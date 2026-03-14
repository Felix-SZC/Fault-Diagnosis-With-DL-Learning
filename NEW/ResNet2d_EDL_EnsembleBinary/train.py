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

# 确保导入模块正确
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.utils.helpers import load_config, save_experiment_info
from common.utils.data_loader import NpyPackDataset
from common.edl_losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from common.utils.edl_helpers import one_hot_embedding
from models import get_model

def get_dataset(data_config, split='train'):
    """根据配置加载新的 Npy 打包数据集，只保留已知类"""
    data_dir = data_config.get('data_dir')
    if data_dir is None:
        raise ValueError("配置文件中必须指定 data.data_dir")
        
    openset_config = data_config.get('openset', {})
    known_classes = openset_config.get('known_classes')
    unknown_classes = openset_config.get('unknown_classes', [])
    
    if known_classes is None:
        raise ValueError("配置文件中必须指定 data.openset.known_classes")
        
    return NpyPackDataset(
        data_dir=data_dir,
        split=split,
        filter_classes=known_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes
    )

def main():
    parser = argparse.ArgumentParser(description='EDL 集成二分类：训练 K 个独立二分类 EDL 小模型')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config['data']
    train_config = config['train']
    model_config = config['model']

    K = len(data_config['openset']['known_classes'])
    print(f"已知类数量 K={K}")

    batch_size = train_config.get('batch_size', 32)
    
    # 因为打包数据集没有验证集，我们直接用 test 作为 val
    train_dataset = get_dataset(data_config, split='train')
    val_dataset = get_dataset(data_config, split='test')

    device_config = train_config.get('device', None)
    cuda_available = torch.cuda.is_available()
    device = torch.device(device_config or ('cuda' if cuda_available else 'cpu'))
    print(f"使用设备: {device}")

    backbone_type = model_config.get('type', 'ResNet18_2d_Light')
    print(f"backbone: {backbone_type}, num_classes=2 per model")

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
    
    # 从 config 中读取新策略
    ensemble_strategy = train_config.get('ensemble_strategy', 'Normal_vs_Fault_i')
    print(f"使用集成策略: {ensemble_strategy}")
    
    save_experiment_info(config, checkpoint_dir, model=None, train_dataset=train_dataset, val_dataset=val_dataset, start_time=train_start_time)

    model_kw = {'num_classes': 2}

    for k in range(K):
        print(f"\n{'='*60}")
        if ensemble_strategy == 'One_vs_Rest':
            print(f"训练 OvR 模型 k = {k} / {K-1}（类别 {k} vs Rest）")
            train_subset = train_dataset
            val_subset = val_dataset
            train_pos = np.sum(train_dataset.y == k)
            train_neg = len(train_dataset) - train_pos
            val_pos = np.sum(val_dataset.y == k)
            val_neg = len(val_dataset) - val_pos
        else:  # 默认为 'Normal_vs_Fault_i'
            if k == 0:
                print(f"训练二分类模型 k = 0 / {K-1}（正常 vs 所有故障）")
            else:
                print(f"训练二分类模型 k = {k} / {K-1}（正常 vs 第 {k} 类故障）")
            
            if k == 0:
                train_subset = train_dataset
                val_subset = val_dataset
                train_pos = np.sum(train_dataset.y == 0)
                train_neg = len(train_dataset) - train_pos
                val_pos = np.sum(val_dataset.y == 0)
                val_neg = len(val_dataset) - val_pos
            else:
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
        
        print(f"  训练集: 正类={train_pos}, 负类={train_neg} (全预测否准确率={100.*train_neg/len(train_subset):.1f}%)")
        print(f"  验证集: 正类={val_pos}, 负类={val_neg} (全预测否准确率={100.*val_neg/len(val_subset):.1f}%)")
        
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
                
                # 根据策略动态生成二分类标签
                if ensemble_strategy == 'One_vs_Rest':
                    binary_labels = (labels == k).long()  # 1 表示类别 k，0 表示 Rest
                else: # 'Normal_vs_Fault_i'
                    if k == 0:
                        binary_labels = (labels == 0).long() # 1 表示正常(0)，0 表示故障(>0)
                    else:
                        binary_labels = (labels == k).long() # 1 表示故障(k)，0 表示正常(0)
                optimizer.zero_grad()
                out = model(inputs)
                logits = out[0] if isinstance(out, tuple) else out
                
                y_binary = one_hot_embedding(binary_labels, 2).float().to(device)
                n_pos = (binary_labels == 1).sum().item()
                n_neg = (binary_labels == 0).sum().item()
                pos_w = n_neg / max(n_pos, 1)
                sample_w = torch.where(binary_labels == 1, torch.tensor(pos_w, device=device, dtype=torch.float), torch.ones_like(binary_labels, device=device, dtype=torch.float))
                
                loss = criterion(logits, y_binary, epoch, 2, annealing_step, device, sample_weight=sample_w)
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

            train_loss /= len(train_subset)
            train_acc = 100.0 * train_correct / train_total

            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    
                    if ensemble_strategy == 'One_vs_Rest':
                        binary_labels = (labels == k).long()
                    else: # 'Normal_vs_Fault_i'
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

            val_loss /= len(val_subset)
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

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
                torch.save({'state_dict': model.state_dict(), 'k': k, 'best_val_acc': best_val_acc}, path_k)

        path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
        if not os.path.exists(path_k):
            torch.save({'state_dict': model.state_dict(), 'k': k, 'best_val_acc': best_val_acc}, path_k)
        print(f"模型 k={k} 已保存: {path_k}, best_val_acc={best_val_acc:.2f}%")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        if ensemble_strategy == 'One_vs_Rest':
            fig.suptitle(f'Model k={k} (Class {k} vs Rest)')
        else: # 'Normal_vs_Fault_i'
            if k == 0:
                fig.suptitle(f'Model k=0 (Normal vs All Faults)')
            else:
                fig.suptitle(f'Model k={k} (Normal vs Fault {k})')
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
