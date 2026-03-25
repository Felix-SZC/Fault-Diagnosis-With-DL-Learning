"""
阶段一：LaoDA 闭集 K 类多分类预训练（EDL），得到共享特征提取器 + K 类 fc2。
产出 pretrained_full.pth，供 train_nvf_frozen_backbone.py 加载 backbone（不含 fc2）。

不修改 train_nvf.py / LaoDA.py。
"""
import argparse
import os
import sys
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.utils.helpers import load_config, save_experiment_info
from common.utils.data_loader_1d import NpyPackDataset1D
from common.edl_losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from common.utils.edl_helpers import one_hot_embedding
from models import get_model


def get_dataset(data_config, split='train'):
    data_dir = data_config.get('data_dir')
    if data_dir is None:
        raise ValueError("配置文件中必须指定 data.data_dir")
    openset_config = data_config.get('openset', {})
    known_classes = openset_config.get('known_classes')
    unknown_classes = openset_config.get('unknown_classes', [])
    if known_classes is None:
        raise ValueError("配置文件中必须指定 data.openset.known_classes")
    return NpyPackDataset1D(
        data_dir=data_dir,
        split=split,
        filter_classes=known_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
    )


def main():
    parser = argparse.ArgumentParser(description='LaoDA 闭集 K 类 EDL 预训练（阶段一）')
    parser.add_argument('--config', type=str, default='configs/bench_LaoDA_closed_pretrain.yaml')
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config['data']
    train_config = config['train']
    model_config = config['model']

    backbone_type = model_config.get('type', 'LaoDA')
    if backbone_type != 'LaoDA':
        raise ValueError(f"本脚本仅支持 LaoDA，当前 model.type={backbone_type}")

    known_classes = data_config['openset']['known_classes']
    K = len(known_classes)
    print(f"闭集类别数 K={K}（含正常），多分类头维度={K}")

    batch_size = train_config.get('batch_size', 32)
    train_dataset = get_dataset(data_config, split='train')
    val_dataset = get_dataset(data_config, split='test')

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

    train_start_time = datetime.now()
    save_experiment_info(
        config, checkpoint_dir, model=None,
        train_dataset=train_dataset, val_dataset=val_dataset,
        start_time=train_start_time,
    )

    model = get_model('LaoDA', num_classes=K).to(device)
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=momentum, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    if use_scheduler and sch_type == 'StepLR':
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=1.0)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    torch.manual_seed(train_config.get('seed', 42))

    best_val_acc = 0.0
    best_val_loss = float('inf')
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        train_tp0 = train_fp0 = train_fn0 = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            y_onehot = one_hot_embedding(labels, K).float().to(device)
            optimizer.zero_grad()
            logits = model(inputs)
            logits = logits[0] if isinstance(logits, tuple) else logits

            loss = criterion(logits, y_onehot, epoch, K, annealing_step, device, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()

            with torch.no_grad():
                evidence = relu_evidence(logits)
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                preds = probs.argmax(dim=1)

            train_loss += loss.item() * inputs.size(0)
            train_correct += (preds == labels).sum().item()
            train_total += inputs.size(0)
            # 与 NvF k=0 一致：将「类别 0（正常）」视为正类，统计二分类 recall/precision
            train_tp0 += ((preds == 0) & (labels == 0)).sum().item()
            train_fp0 += ((preds == 0) & (labels != 0)).sum().item()
            train_fn0 += ((preds != 0) & (labels == 0)).sum().item()

        train_loss /= len(train_dataset)
        train_acc = 100.0 * train_correct / max(train_total, 1)
        train_recall_pos = 100.0 * train_tp0 / max(train_tp0 + train_fn0, 1)
        train_precision_pos = 100.0 * train_tp0 / max(train_tp0 + train_fp0, 1)

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_tp0 = val_fp0 = val_fn0 = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                y_onehot = one_hot_embedding(labels, K).float().to(device)
                logits = model(inputs)
                logits = logits[0] if isinstance(logits, tuple) else logits
                loss = criterion(logits, y_onehot, epoch, K, annealing_step, device, kl_weight=kl_weight)
                evidence = relu_evidence(logits)
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                preds = probs.argmax(dim=1)
                val_loss += loss.item() * inputs.size(0)
                val_correct += (preds == labels).sum().item()
                val_total += inputs.size(0)
                val_tp0 += ((preds == 0) & (labels == 0)).sum().item()
                val_fp0 += ((preds == 0) & (labels != 0)).sum().item()
                val_fn0 += ((preds != 0) & (labels == 0)).sum().item()

        val_loss /= len(val_dataset)
        val_acc = 100.0 * val_correct / max(val_total, 1)
        val_recall_pos = 100.0 * val_tp0 / max(val_tp0 + val_fn0, 1)
        val_precision_pos = 100.0 * val_tp0 / max(val_tp0 + val_fp0, 1)
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()

        # *_rec_pos / *_prec_pos：把「预测为类 0」当作正类，与 NvF 中 k=0（正常 vs 全故障）语义一致
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(
                f"  Epoch [{epoch + 1}/{num_epochs}] train_loss={train_loss:.4f} train_acc={train_acc:.2f}% "
                f"train_rec_pos={train_recall_pos:.2f}% train_prec_pos={train_precision_pos:.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% "
                f"val_rec_pos={val_recall_pos:.2f}% val_prec_pos={val_precision_pos:.2f}%"
            )

        # 预训练阶段，准确率容易饱和。为了获取更充分退火的特征，当准确率相同时，比较 loss。
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            out_path = os.path.join(checkpoint_dir, 'pretrained_full.pth')
            torch.save(
                {
                    'state_dict': model.state_dict(),
                    'num_classes': K,
                    'known_classes': known_classes,
                    'best_val_acc': best_val_acc,
                    'best_val_loss': best_val_loss,
                },
                out_path,
            )
            
        # 始终保存最后一个 epoch 的权重，作为充分退火后的备用选择
        last_out_path = os.path.join(checkpoint_dir, 'pretrained_full_last.pth')
        torch.save(
            {
                'state_dict': model.state_dict(),
                'num_classes': K,
                'known_classes': known_classes,
                'val_acc': val_acc,
                'val_loss': val_loss,
                'epoch': epoch,
            },
            last_out_path,
        )

    out_path = os.path.join(checkpoint_dir, 'pretrained_full.pth')
    if not os.path.exists(out_path):
        torch.save(
            {
                'state_dict': model.state_dict(),
                'num_classes': K,
                'known_classes': known_classes,
                'best_val_acc': best_val_acc,
            },
            out_path,
        )
    print(f"\n预训练权重已保存: {out_path}, best_val_acc={best_val_acc:.2f}%, best_val_loss={best_val_loss:.4f}")
    print(f"最后 epoch 权重已保存: {last_out_path}")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Closed-set K-class Pretrain')
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
    fig.savefig(os.path.join(checkpoint_dir, 'training_curves_closed_pretrain.png'), dpi=150)
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
    print("\n阶段一完成。下一步: python train_nvf_frozen_backbone.py --config configs/bench_NvF_LaoDA_frozen_backbone.yaml")


if __name__ == '__main__':
    main()
