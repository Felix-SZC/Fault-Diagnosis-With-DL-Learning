"""
阶段二：NvF 各子模型在冻结 LaoDA backbone 上只训练二分类头 fc2（可选解冻 fc1）。
依赖阶段一 train_lao_da_closed_pretrain.py 产出的 pretrained_full.pth。

保存的 model_k.pth 与 train_nvf.py 格式一致，可直接用 test_NvF.py 加载。

不修改 train_nvf.py / test_NvF.py / LaoDA.py。
"""
import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.utils.helpers import load_config, save_experiment_info
from common.utils.data_loader import NpyPackDataset
from common.utils.data_loader_1d import NpyPackDataset1D
from common.utils.lao_da_pretrained import (
    load_lao_da_backbone_from_multiclass_ckpt,
    set_lao_da_trainable_heads,
    summarize_trainable_params,
)
from common.edl_losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from common.utils.edl_helpers import one_hot_embedding
from models import get_model


def get_dataset(data_config, model_type, split='train'):
    data_dir = data_config.get('data_dir')
    if data_dir is None:
        raise ValueError("配置文件中必须指定 data.data_dir")
    openset_config = data_config.get('openset', {})
    known_classes = openset_config.get('known_classes')
    unknown_classes = openset_config.get('unknown_classes', [])
    if known_classes is None:
        raise ValueError("配置文件中必须指定 data.openset.known_classes")
    dataset_cls = NpyPackDataset1D if model_type == 'LaoDA' else NpyPackDataset
    return dataset_cls(
        data_dir=data_dir,
        split=split,
        filter_classes=known_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
    )


def resolve_model_indices(train_config, K):
    model_indices_cfg = train_config.get('model_indices', 'all')
    if model_indices_cfg is None or model_indices_cfg == 'all':
        return list(range(K))

    if isinstance(model_indices_cfg, int):
        model_indices = [model_indices_cfg]
    elif isinstance(model_indices_cfg, list):
        model_indices = model_indices_cfg
    else:
        raise ValueError("train.model_indices 仅支持 'all'、整数或整数列表")

    normalized = []
    for k in model_indices:
        if not isinstance(k, int):
            raise ValueError("train.model_indices 列表中的元素必须是整数")
        if k < 0 or k >= K:
            raise ValueError(f"train.model_indices 包含越界索引 {k}，应在 [0, {K - 1}]")
        if k not in normalized:
            normalized.append(k)
    if not normalized:
        raise ValueError("train.model_indices 解析后为空，请至少指定一个子模型索引")
    return normalized


def _resolve_pretrained_path(raw_path: str) -> str:
    if os.path.isabs(raw_path):
        return raw_path
    return os.path.normpath(os.path.join(os.getcwd(), raw_path))


def main():
    parser = argparse.ArgumentParser(description='NvF + 冻结 LaoDA 预训练 backbone，只训二分类头')
    parser.add_argument('--config', type=str, default='configs/bench_NvF_LaoDA_frozen_backbone.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config['data']
    train_config = config['train']
    model_config = config['model']

    K = len(data_config['openset']['known_classes'])
    print(f"已知类数量 K={K}")

    backbone_type = model_config.get('type', 'LaoDA')
    if backbone_type != 'LaoDA':
        raise ValueError(f"train_nvf_frozen_backbone 仅支持 LaoDA，当前 model.type={backbone_type}")

    pre_raw = train_config.get('pretrained_multiclass_ckpt')
    if not pre_raw:
        raise ValueError("请在 train.pretrained_multiclass_ckpt 中指定阶段一 pretrained_full.pth 路径")
    pretrained_path = _resolve_pretrained_path(pre_raw)
    print(f"多类预训练权重: {pretrained_path}")

    freeze_fc1 = train_config.get('freeze_fc1', True)
    print(f"freeze_fc1={freeze_fc1} （True 时仅训练 fc2；False 时训练 fc1+fc2）")

    batch_size = train_config.get('batch_size', 32)
    print(f"backbone: {backbone_type}, num_classes=2 per model (frozen backbone)")

    train_dataset = get_dataset(data_config, backbone_type, split='train')
    val_dataset = get_dataset(data_config, backbone_type, split='test')

    device_config = train_config.get('device', None)
    device = torch.device(device_config or ('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"使用设备: {device}")

    edl_loss_type = train_config.get('edl_loss_type', 'mse')
    annealing_step = train_config.get('edl_annealing_step', 10)
    kl_weight = float(train_config.get('kl_weight', 1.0))
    if edl_loss_type == 'mse':
        criterion = edl_mse_loss
    elif edl_loss_type == 'digamma':
        criterion = edl_digamma_loss
    elif edl_loss_type == 'log':
        criterion = edl_log_loss
    else:
        raise ValueError(f"不支持的 EDL 损失: {edl_loss_type}")

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

    ensemble_strategy = train_config.get('ensemble_strategy', 'Normal_vs_Fault_i')
    if ensemble_strategy != 'Normal_vs_Fault_i':
        print(f"警告：仅支持 Normal_vs_Fault_i，配置为 {ensemble_strategy}，仍按 NvF 训练。")
    print("使用集成策略: Normal_vs_Fault_i (NvF) + 冻结预训练 backbone")

    train_start_time = datetime.now()
    save_experiment_info(
        config, checkpoint_dir, model=None,
        train_dataset=train_dataset, val_dataset=val_dataset,
        start_time=train_start_time,
    )

    model_kw = {'num_classes': 2}
    model_indices = resolve_model_indices(train_config, K)
    print(f"本次将训练子模型索引: {model_indices}")

    for k in model_indices:
        print(f"\n{'=' * 60}")
        if k == 0:
            print(f"训练二分类模型 k = 0 / {K - 1}（正常 vs 所有故障）")
            train_subset = train_dataset
            val_subset = val_dataset
            train_pos = np.sum(train_dataset.y == 0)
            train_neg = len(train_dataset) - train_pos
            val_pos = np.sum(val_dataset.y == 0)
            val_neg = len(val_dataset) - val_pos
        else:
            print(f"训练二分类模型 k = {k} / {K - 1}（正常 vs 第 {k} 类故障）")
            train_indices = np.where((train_dataset.y == 0) | (train_dataset.y == k))[0]
            val_indices = np.where((val_dataset.y == 0) | (val_dataset.y == k))[0]
            train_subset = Subset(train_dataset, train_indices)
            val_subset = Subset(val_dataset, val_indices)
            train_pos = np.sum(train_dataset.y[train_indices] == k)
            train_neg = np.sum(train_dataset.y[train_indices] == 0)
            val_pos = np.sum(val_dataset.y[val_indices] == k)
            val_neg = np.sum(val_dataset.y[val_indices] == 0)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False)

        print(f"  训练集: 正类={train_pos}, 负类={train_neg} (全预测否准确率={100. * train_neg / len(train_subset):.1f}%)")
        print(f"  验证集: 正类={val_pos}, 负类={val_neg} (全预测否准确率={100. * val_neg / len(val_subset):.1f}%)")

        torch.manual_seed(train_config.get('seed', 42) + k)

        model = get_model(backbone_type, **model_kw).to(device)
        loaded, skipped = load_lao_da_backbone_from_multiclass_ckpt(model, pretrained_path, device)
        print(f"  已从预训练加载 {len(loaded)} 个张量；跳过 fc2 等: {len(skipped)} 项")
        set_lao_da_trainable_heads(model, train_fc1=not freeze_fc1)
        trainable_names = summarize_trainable_params(model)
        print(f"  可训练参数 ({len(trainable_names)}): {trainable_names}")

        params_train = [p for p in model.parameters() if p.requires_grad]
        if not params_train:
            raise RuntimeError("没有可训练参数，请检查 freeze_fc1 与模型结构")

        if optimizer_type == 'SGD':
            optimizer = optim.SGD(params_train, lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
        else:
            optimizer = optim.Adam(params_train, lr=learning_rate)

        if use_scheduler and sch_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1)

        best_val_acc = 0.0
        best_val_loss = float('inf')
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(num_epochs):
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            train_tp = train_fp = train_fn = 0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                if k == 0:
                    binary_labels = (labels == 0).long()
                else:
                    binary_labels = (labels == k).long()
                optimizer.zero_grad()
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out

                y_binary = one_hot_embedding(binary_labels, 2).float().to(device)
                n_pos = (binary_labels == 1).sum().item()
                n_neg = (binary_labels == 0).sum().item()
                pos_w = n_neg / max(n_pos, 1)
                sample_w = torch.where(
                    binary_labels == 1,
                    torch.tensor(pos_w, device=device, dtype=torch.float),
                    torch.ones_like(binary_labels, device=device, dtype=torch.float),
                )

                loss = criterion(logits, y_binary, epoch, 2, annealing_step, device,
                                 sample_weight=sample_w, kl_weight=kl_weight)
                loss.backward()
                optimizer.step()

                with torch.no_grad():
                    evidence = relu_evidence(logits)
                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    preds = (probs[:, 1] > 0.5).long()

                train_loss += loss.item() * inputs.size(0)
                train_correct += (preds == binary_labels).sum().item()
                train_total += inputs.size(0)
                train_tp += ((preds == 1) & (binary_labels == 1)).sum().item()
                train_fp += ((preds == 1) & (binary_labels == 0)).sum().item()
                train_fn += ((preds == 0) & (binary_labels == 1)).sum().item()

            train_loss /= len(train_subset)
            train_acc = 100.0 * train_correct / train_total
            train_recall_pos = 100.0 * train_tp / max(train_tp + train_fn, 1)
            train_precision_pos = 100.0 * train_tp / max(train_tp + train_fp, 1)

            model.eval()
            val_loss = 0.0
            val_correct = val_total = val_tp = val_fp = val_fn = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    if k == 0:
                        binary_labels = (labels == 0).long()
                    else:
                        binary_labels = (labels == k).long()
                    out = model(inputs)
                    logits = out[0] if isinstance(out, tuple) else out

                    y_binary = one_hot_embedding(binary_labels, 2).float().to(device)
                    n_pos = (binary_labels == 1).sum().item()
                    n_neg = (binary_labels == 0).sum().item()
                    pos_w = n_neg / max(n_pos, 1)
                    sample_w = torch.where(
                        binary_labels == 1,
                        torch.tensor(pos_w, device=device, dtype=torch.float),
                        torch.ones_like(binary_labels, device=device, dtype=torch.float),
                    )

                    loss = criterion(logits, y_binary, epoch, 2, annealing_step, device,
                                     sample_weight=sample_w, kl_weight=kl_weight)
                    evidence = relu_evidence(logits)
                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    preds = (probs[:, 1] > 0.5).long()

                    val_loss += loss.item() * inputs.size(0)
                    val_correct += (preds == binary_labels).sum().item()
                    val_total += inputs.size(0)
                    val_tp += ((preds == 1) & (binary_labels == 1)).sum().item()
                    val_fp += ((preds == 1) & (binary_labels == 0)).sum().item()
                    val_fn += ((preds == 0) & (binary_labels == 1)).sum().item()

            val_loss /= len(val_subset)
            val_acc = 100.0 * val_correct / val_total
            val_recall_pos = 100.0 * val_tp / max(val_tp + val_fn, 1)
            val_precision_pos = 100.0 * val_tp / max(val_tp + val_fp, 1)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_loss)
            else:
                scheduler.step()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"  Epoch [{epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} "
                      f"train_acc={train_acc:.2f}% train_rec_pos={train_recall_pos:.2f}% "
                      f"train_prec_pos={train_precision_pos:.2f}% val_loss={val_loss:.4f} "
                      f"val_acc={val_acc:.2f}% val_rec_pos={val_recall_pos:.2f}% "
                      f"val_prec_pos={val_precision_pos:.2f}%")

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
                torch.save({'state_dict': model.state_dict(), 'k': k, 'best_val_acc': best_val_acc, 'best_val_loss': best_val_loss}, path_k)

        path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
        if not os.path.exists(path_k):
            torch.save({'state_dict': model.state_dict(), 'k': k, 'best_val_acc': best_val_acc}, path_k)
        print(f"模型 k={k} 已保存: {path_k}, best_val_acc={best_val_acc:.2f}%, best_val_loss={best_val_loss:.4f}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        if k == 0:
            fig.suptitle(f'Model k=0 frozen-backbone NvF')
        else:
            fig.suptitle(f'Model k={k} frozen-backbone NvF')
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

    save_experiment_info(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=None,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        history=None,
        best_val_acc=None,
        start_time=train_start_time,
        end_time=datetime.now(),
    )
    print("\n阶段二完成。测试: python test_NvF.py --config configs/bench_NvF_LaoDA_frozen_backbone.yaml")


if __name__ == '__main__':
    main()
