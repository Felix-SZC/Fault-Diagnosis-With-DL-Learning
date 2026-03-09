"""
EDL + K 个独立二分类小模型训练脚本。

依次训练 K 个 ResNet2d(num_classes=2)，第 k 个模型只做「是否属于第 k 类」的二分类，
使用 EDL 损失（evidence + Dirichlet），独立保存为 model_0.pth .. model_{K-1}.pth。
与 train_edl_binary.py 区别：本脚本为 K 个独立小模型，无参数共享；后者为单 backbone 多二分类头。
"""
import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到 sys.path，确保可以导入 common、models 等模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config, save_experiment_info
from common.utils.data_loader import RawSignalDataset, NpyIndexDataset, LabeledImageDataset
from common.edl_losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from common.utils.edl_helpers import one_hot_embedding
from models import get_model


def get_dataset(data_config, split='train'):
    """根据配置选择并实例化数据集，与 train_edl_binary 一致：仅已知类，标签映射为 0..K-1。"""
    data_type = data_config.get('type')
    if data_type is None:
        raise ValueError("配置文件中必须指定 data.type")
    openset_config = data_config.get('openset', {})
    known_classes = openset_config.get('known_classes')
    if known_classes is None:
        raise ValueError("配置文件中必须指定 data.openset.known_classes")
    # 根据数据类型加载相应的数据集
    if data_type == 'raw_signal':
        base_dir = data_config.get('raw_signal_output_dir')
        split_dir = os.path.join(base_dir, split)
        return RawSignalDataset(split_dir=split_dir, filter_classes=known_classes)
    elif data_type == 'wpt':
        base_dir = data_config.get('wpt_output_dir')
        split_dir = os.path.join(base_dir, split)
        return NpyIndexDataset(split_dir=split_dir, filter_classes=known_classes)
    elif data_type in ('image', 'stft'):
        # STFT 时频图等图像数据：目录下为 name_label.png，需 img_output_dir
        base_dir = data_config.get('img_output_dir') or data_config.get('stft_output_dir')
        if not base_dir:
            raise ValueError("image/stft 数据需在配置中指定 data.img_output_dir")
        path = os.path.join(base_dir, split)
        transform = transforms.Compose([transforms.ToTensor()])
        return LabeledImageDataset(path=path, transform=transform, filter_classes=known_classes)
    else:
        raise ValueError(f"train_edl_ensemble_binary 支持 'raw_signal'、'wpt'、'image'/'stft'，当前: {data_type}")


def main():
    parser = argparse.ArgumentParser(description='EDL 集成二分类：训练 K 个独立二分类 EDL 小模型')
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    data_config = config['data']
    train_config = config['train']
    model_config = config['model']

    K = model_config.get('num_classes')
    if K is None or K < 1:
        raise ValueError("config model.num_classes 必须为已知类数量 K")

    # 2. 准备数据
    batch_size = train_config['batch_size']
    train_dataset = get_dataset(data_config, split='train')
    val_dataset = get_dataset(data_config, split='val')
    # 设置 snr_db（噪声增强），仅对 RawSignalDataset 生效
    if isinstance(train_dataset, RawSignalDataset):
        snr_db = train_config.get('snr_db', None)
        train_dataset.snr_db = snr_db
        if snr_db is not None:
            print(f"训练集启用高斯噪声增强，SNR: {snr_db} dB")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    print(f"训练集大小: {len(train_dataset)}，验证集大小: {len(val_dataset)}，K={K}")

    # 3. 设备与模型类型
    device_config = train_config.get('device', None)
    cuda_available = torch.cuda.is_available()
    if device_config and device_config.lower() == 'cuda' and not cuda_available:
        device = torch.device('cpu')
        print("警告: 配置请求 CUDA 但不可用，使用 CPU")
    else:
        device = torch.device(device_config or ('cuda' if cuda_available else 'cpu'))
    print(f"使用设备: {device}")

    # 每个小模型为 backbone(num_classes=2)，若配置为 Binary 则强制使用 ResNet18_2d
    backbone_type = model_config.get('type', 'ResNet18_2d')
    if 'Binary' in str(backbone_type):
        backbone_type = 'ResNet18_2d'
    print(f"backbone: {backbone_type}, num_classes=2  per model")

    # 4. EDL 损失与训练超参（edl_annealing_step 设为 0 则关闭 KL 退火）
    edl_loss_type = train_config.get('edl_loss_type', 'mse')
    annealing_step = train_config.get('edl_annealing_step', 10)
    if edl_loss_type == 'mse':
        criterion = edl_mse_loss
    elif edl_loss_type == 'digamma':
        criterion = edl_digamma_loss
    elif edl_loss_type == 'log':
        criterion = edl_log_loss
    else:
        raise ValueError(f"不支持的 EDL 损失: {edl_loss_type}")
    print(f"EDL 损失: {edl_loss_type}, KL 退火: {'关闭' if not annealing_step else f'step={annealing_step}'}")

    learning_rate = float(train_config.get('learning_rate', 0.0001))
    num_epochs = train_config.get('num_epochs', 60)
    checkpoint_dir = train_config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)

    scheduler_conf = train_config.get('scheduler', {})
    use_scheduler = scheduler_conf.get('use_scheduler', False)
    sch_type = scheduler_conf.get('scheduler_type', 'StepLR') if use_scheduler else None
    step_size = scheduler_conf.get('step_size', 40)
    gamma = scheduler_conf.get('gamma', 0.1)

    optimizer_type = train_config.get('optimizer', 'Adam')
    momentum = float(train_config.get('momentum', 0.9))
    weight_decay = float(train_config.get('weight_decay', 1e-4))

    train_start_time = datetime.now()
    save_experiment_info(config, checkpoint_dir, model=None, train_dataset=train_dataset, val_dataset=val_dataset, start_time=train_start_time)

    # 需要 input_channels 的 backbone（如 TimeFreqAttention），仅对 WPT/raw_signal 数据
    model_kw = {'num_classes': 2}
    if backbone_type == 'TimeFreqAttention' and not isinstance(train_dataset, LabeledImageDataset):
        sample, _ = train_dataset[0]
        # WPT: (1, bands, time) -> bands; RawSignal: (1, L) -> 1
        model_kw['input_channels'] = sample.shape[1] if sample.dim() >= 2 else sample.shape[0]

    # 5. 依次训练 K 个二分类模型（k = 0..K-1）
    for k in range(K):
        print(f"\n{'='*60}")
        print(f"训练二分类模型 k = {k} / {K-1}（是否属于第 k 类）")
        print(f"{'='*60}")
        # 统计正/负样本数（全预测「否」时 val_acc = 负类比例，如 6/7≈85.71%）
        def _count_pos_neg(ds, kk):
            if getattr(ds, 'files', None) is not None and hasattr(ds, '_parse_label'):
                labs = [ds.label_map.get(ds._parse_label(f), ds._parse_label(f)) if getattr(ds, 'label_map', None) else ds._parse_label(f) for f in ds.files]
                pos = sum(1 for l in labs if l == kk)
                return pos, len(labs) - pos
            pos = sum(1 for i in range(len(ds)) if ds[i][1] == kk)
            return pos, len(ds) - pos
        train_pos, train_neg = _count_pos_neg(train_dataset, k)
        val_pos, val_neg = _count_pos_neg(val_dataset, k)
        print(f"  训练集: 正类={train_pos}, 负类={train_neg} (全预测否准确率={100.*train_neg/len(train_dataset):.1f}%)")
        print(f"  验证集: 正类={val_pos}, 负类={val_neg} (全预测否准确率={100.*val_neg/len(val_dataset):.1f}%)")
        # 不同 k 使用不同随机种子，增加多样性
        torch.manual_seed(train_config.get('seed', 42) + k)

        model = get_model(backbone_type, **model_kw).to(device)
        if optimizer_type == 'SGD':
            optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        if use_scheduler and sch_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)

        best_val_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if backbone_type == 'TimeFreqAttention' and inputs.dim() == 4:
                    inputs = inputs.squeeze(1)  # (B, 1, bands, time) -> (B, bands, time)
                # 二分类标签：1 表示第 k 类，0 表示非第 k 类
                binary_labels = (labels == k).long()
                optimizer.zero_grad()
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out
                y_binary = one_hot_embedding(binary_labels, 2).float().to(device)
                # 正类加权：正类样本权 = 负/正 比例，缓解「全预测否」导致 val_acc 卡在 85.71%
                n_pos = (binary_labels == 1).sum().item()
                n_neg = (binary_labels == 0).sum().item()
                pos_w = n_neg / max(n_pos, 1)
                sample_w = torch.where(binary_labels == 1, torch.tensor(pos_w, device=device, dtype=torch.float), torch.ones_like(binary_labels, device=device, dtype=torch.float))
                loss = criterion(logits, y_binary, epoch, 2, annealing_step, device, sample_weight=sample_w)
                loss.backward()
                optimizer.step()
                # 用于统计准确率：取「Yes」概率 > 0.5 为预测正类
                with torch.no_grad():
                    evidence = relu_evidence(logits)
                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    preds = (probs[:, 1] > 0.5).long()
                train_loss += loss.item() * inputs.size(0)
                train_correct += (preds == binary_labels).sum().item()
                train_total += inputs.size(0)

            train_loss /= len(train_dataset)
            train_acc = 100.0 * train_correct / train_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if backbone_type == 'TimeFreqAttention' and inputs.dim() == 4:
                        inputs = inputs.squeeze(1)  # (B, 1, bands, time) -> (B, bands, time)
                    binary_labels = (labels == k).long()
                    out = model(inputs)
                    logits = out[0] if isinstance(out, tuple) else out
                    y_binary = one_hot_embedding(binary_labels, 2).float().to(device)
                    n_pos = (binary_labels == 1).sum().item()
                    n_neg = (binary_labels == 0).sum().item()
                    pos_w = n_neg / max(n_pos, 1)
                    sample_w = torch.where(binary_labels == 1, torch.tensor(pos_w, device=device, dtype=torch.float), torch.ones_like(binary_labels, device=device, dtype=torch.float))
                    loss = criterion(logits, y_binary, epoch, 2, annealing_step, device, sample_weight=sample_w)
                    evidence = relu_evidence(logits)
                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    preds = (probs[:, 1] > 0.5).long()
                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (preds == binary_labels).sum().item()
                    val_total += inputs.size(0)

            val_loss /= len(val_dataset)
            val_acc = 100.0 * val_correct / val_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch+1}/{num_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.2f}% val_loss={val_loss:.4f} val_acc={val_acc:.2f}%")

            # 保存当前 k 的最佳模型
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
                torch.save({'state_dict': model.state_dict(), 'k': k, 'best_val_acc': best_val_acc}, path_k)

        # 若从未保存过（如未达到过更优验证准确率），则保存最后一轮
        path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
        if not os.path.exists(path_k):
            torch.save({'state_dict': model.state_dict(), 'k': k, 'best_val_acc': best_val_acc}, path_k)
        print(f"模型 k={k} 已保存: {path_k}, best_val_acc={best_val_acc:.2f}%")

        # 保存该模型的训练曲线
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        fig.suptitle(f'Model k={k} (class k vs rest)')
        ax1.plot(history['train_loss'], label='Train')
        ax1.plot(history['val_loss'], label='Val')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(history['train_acc'], label='Train')
        ax2.plot(history['val_acc'], label='Val')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        fig.tight_layout()
        fig.savefig(os.path.join(checkpoint_dir, f'training_curves_{k}.png'), dpi=150)
        plt.close(fig)

    # 6. 保存实验信息
    train_end_time = datetime.now()
    save_experiment_info(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        history=None,
        best_val_acc=None,
        start_time=train_start_time,
        end_time=train_end_time
    )
    print("\nK 个二分类 EDL 模型训练完成。")

if __name__ == '__main__':
    main()
