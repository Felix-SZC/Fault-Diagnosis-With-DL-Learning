"""
EDL模型微调脚本
实现"冻结骨干+解耦训练"策略来恢复证据量

核心策略：
1. 加载预训练模型权重（含Center Loss训练）
2. 冻结backbone参数，保持特征空间不变
3. 仅训练分类器头部
4. 使用纯EDL MSE损失（不含Center Loss）进行微调
"""

import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import numpy as np

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config, save_experiment_info, count_parameters
from common.utils.data_loader import RawSignalDataset, NpyIndexDataset
from common.edl_losses import edl_mse_loss, relu_evidence
from common.utils.edl_helpers import one_hot_embedding
from models import get_model


def get_dataset(data_config, split='train'):
    """根据配置选择并实例化数据集"""
    data_type = data_config.get('type')
    if data_type is None:
        raise ValueError("配置文件中必须指定 data.type")
    
    openset_config = data_config.get('openset', {})
    known_classes = openset_config.get('known_classes')
    if known_classes is None:
        raise ValueError("配置文件中必须指定 data.openset.known_classes")
    
    if data_type == 'raw_signal':
        base_dir = data_config.get('raw_signal_output_dir')
        split_dir = os.path.join(base_dir, split)
        return RawSignalDataset(split_dir=split_dir, filter_classes=known_classes)
    elif data_type == 'wpt':
        base_dir = data_config.get('wpt_output_dir')
        split_dir = os.path.join(base_dir, split)
        return NpyIndexDataset(split_dir=split_dir, filter_classes=known_classes)
    else:
        raise ValueError(f"微调脚本目前仅支持 'raw_signal' 和 'wpt' 数据类型，当前类型: {data_type}")


def freeze_backbone(model):
    """
    冻结backbone参数，只保留分类器可训练
    
    Args:
        model: PyTorch模型实例
    
    Returns:
        tuple: (可训练参数数量, 冻结参数数量)
    """
    # 识别backbone组件（根据ResNet结构）
    backbone_modules = []
    
    # 检查模型结构并识别backbone
    if hasattr(model, 'conv1'):
        backbone_modules.append(model.conv1)
    if hasattr(model, 'bn1'):
        backbone_modules.append(model.bn1)
    if hasattr(model, 'relu'):
        backbone_modules.append(model.relu)
    if hasattr(model, 'maxpool'):
        backbone_modules.append(model.maxpool)
    if hasattr(model, 'layer1'):
        backbone_modules.append(model.layer1)
    if hasattr(model, 'layer2'):
        backbone_modules.append(model.layer2)
    if hasattr(model, 'layer3'):
        backbone_modules.append(model.layer3)
    if hasattr(model, 'layer4'):
        backbone_modules.append(model.layer4)
    if hasattr(model, 'avgpool'):
        backbone_modules.append(model.avgpool)
    
    # 冻结backbone参数
    frozen_params = 0
    for module in backbone_modules:
        for param in module.parameters():
            param.requires_grad = False
            frozen_params += param.numel()
    
    # 确保分类器可训练
    trainable_params = 0
    if hasattr(model, 'fc'):
        for param in model.fc.parameters():
            param.requires_grad = True
            trainable_params += param.numel()
    else:
        # 如果模型结构不同，尝试找到最后一个线性层
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear) and 'fc' in name.lower():
                for param in module.parameters():
                    param.requires_grad = True
                    trainable_params += param.numel()
    
    return trainable_params, frozen_params


def compute_evidence_stats(outputs, device):
    """
    计算证据量统计信息
    
    Args:
        outputs: 模型输出的logits，形状为 [batch_size, num_classes]
        device: 计算设备
    
    Returns:
        dict: 包含平均证据值、不确定性等统计信息
    """
    evidence = relu_evidence(outputs)
    alpha = evidence + 1
    
    # 计算浓度参数总和
    S = torch.sum(alpha, dim=1)
    
    # 计算不确定性：u = K / S，其中K是类别数
    K = alpha.shape[1]
    uncertainty = K / S
    
    # 计算平均证据值
    mean_evidence = torch.mean(evidence, dim=0)
    total_evidence = torch.sum(evidence, dim=1)
    
    stats = {
        'mean_evidence_per_class': mean_evidence.cpu().numpy(),
        'mean_total_evidence': torch.mean(total_evidence).item(),
        'mean_uncertainty': torch.mean(uncertainty).item(),
        'std_uncertainty': torch.std(uncertainty).item(),
    }
    
    return stats


def main():
    parser = argparse.ArgumentParser(description='EDL Fine-tuning Script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to pretrained model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for fine-tuned model (default: checkpoint_dir/finetune)')
    parser.add_argument('--num_epochs', type=int, default=None, help='Number of fine-tuning epochs (default: from config or 20)')
    parser.add_argument('--learning_rate', type=float, default=None, help='Learning rate for fine-tuning (default: original_lr / 10)')
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    data_config = config['data']
    train_config = config['train']
    model_config = config['model']
    
    # 2. 准备数据
    batch_size = train_config['batch_size']
    
    train_dataset = get_dataset(data_config, split='train')
    val_dataset = get_dataset(data_config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 3. 准备设备
    device_config = train_config.get('device', None)
    cuda_available = torch.cuda.is_available()
    
    if device_config:
        requested_device = device_config.lower()
        if requested_device == 'cuda' and not cuda_available:
            print(f"警告: 配置文件中指定使用 CUDA，但 PyTorch 未编译 CUDA 支持，将使用 CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(requested_device)
    else:
        device = torch.device('cuda' if cuda_available else 'cpu')
    print(f"使用设备: {device}")
    
    # 4. 加载模型
    num_classes = model_config.get('num_classes')
    model = get_model(model_config.get('type'), num_classes=num_classes).to(device)
    
    # 加载预训练权重
    print(f"\n加载预训练模型: {args.checkpoint}")
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"检查点文件不存在: {args.checkpoint}")
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    # 处理不同的checkpoint格式
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint
    
    # 加载权重（允许部分匹配）
    model.load_state_dict(state_dict, strict=False)
    print("预训练权重加载完成")
    
    # 5. 冻结backbone，只训练分类器
    print("\n冻结backbone参数...")
    trainable_params, frozen_params = freeze_backbone(model)
    print(f"可训练参数数量: {trainable_params:,}")
    print(f"冻结参数数量: {frozen_params:,}")
    print(f"可训练参数占比: {100.0 * trainable_params / (trainable_params + frozen_params):.2f}%")
    
    # 验证冻结状态
    trainable_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"验证：实际可训练参数: {trainable_count:,}")
    
    # 6. 配置优化器和损失函数
    original_lr = float(train_config['learning_rate'])
    finetune_lr = 0.01

    print(f"\n原始学习率: {original_lr}")
    print(f"微调学习率: {finetune_lr}")
    
    # 只优化可训练参数（分类器）
    trainable_params_list = [p for p in model.parameters() if p.requires_grad]
    optimizer_type = 'Adam'
    
    if optimizer_type == 'SGD':
        momentum = float(train_config.get('momentum', 0.9))
        weight_decay = float(train_config.get('weight_decay', 1e-4))
        optimizer = optim.SGD(
            trainable_params_list,
            lr=finetune_lr,
            momentum=momentum,
            weight_decay=weight_decay
        )
        print(f"使用优化器: SGD (momentum={momentum}, weight_decay={weight_decay})")
    else:
        optimizer = optim.Adam(trainable_params_list, lr=finetune_lr)
        print(f"使用优化器: Adam")
    
    # EDL损失函数配置
    annealing_step = 1e-6
    criterion = edl_mse_loss
    print(f"使用EDL损失函数: MSE")
    print(f"KL退火步数: {annealing_step}")
    
    # 学习率调度器
    scheduler_conf = train_config.get('scheduler', {})
    if scheduler_conf.get('use_scheduler', False):
        sch_type = scheduler_conf.get('scheduler_type', 'StepLR')
        if sch_type == 'StepLR':
            scheduler = optim.lr_scheduler.StepLR(
                optimizer,
                step_size=scheduler_conf.get('step_size', 10),
                gamma=scheduler_conf.get('gamma', 0.1)
            )
        else:
            scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    else:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)
    
    # 7. 设置输出目录
    if args.output_dir:
        checkpoint_dir = args.output_dir
    else:
        # 默认：从checkpoint路径自动提取目录，在其下创建finetune子目录
        # 例如：checkpoints/ResNet2d_EDL/run11/best_model.pth -> checkpoints/ResNet2d_EDL/run11/finetune
        checkpoint_path = os.path.abspath(args.checkpoint)
        checkpoint_parent_dir = os.path.dirname(checkpoint_path)
        checkpoint_dir = os.path.join(checkpoint_parent_dir, 'finetune')
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"\n微调模型保存目录: {checkpoint_dir}")
    
    # 8. 训练循环
    num_epochs = args.num_epochs if args.num_epochs is not None else train_config.get('num_epochs', 20)
    print(f"\n开始微调，总轮数: {num_epochs}")
    
    best_val_acc = 0.0
    best_evidence = 0.0  # 记录最佳平均证据值
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'train_evidence': [],
        'train_uncertainty': [],
        'val_evidence': [],
        'val_uncertainty': []
    }
    
    train_start_time = datetime.now()
    
    # 实时绘图初始化
    plt.ion()
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    fig.suptitle('Fine-tuning Progress')
    ax_loss = axes[0, 0]
    ax_acc = axes[0, 1]
    ax_evidence = axes[1, 0]
    ax_uncertainty = axes[1, 1]
    
    def update_plot():
        """更新实时训练曲线"""
        epochs = range(1, len(history['train_loss']) + 1)
        
        ax_loss.clear()
        ax_loss.plot(epochs, history['train_loss'], label='Train Loss', marker='o')
        ax_loss.plot(epochs, history['val_loss'], label='Val Loss', marker='s')
        ax_loss.set_xlabel('Epoch')
        ax_loss.set_ylabel('Loss')
        ax_loss.set_title('Loss')
        ax_loss.legend()
        ax_loss.grid(True)
        
        ax_acc.clear()
        ax_acc.plot(epochs, history['train_acc'], label='Train Acc', marker='o')
        ax_acc.plot(epochs, history['val_acc'], label='Val Acc', marker='s')
        ax_acc.set_xlabel('Epoch')
        ax_acc.set_ylabel('Accuracy (%)')
        ax_acc.set_title('Accuracy')
        ax_acc.legend()
        ax_acc.grid(True)
        
        ax_evidence.clear()
        ax_evidence.plot(epochs, history['train_evidence'], label='Train Evidence', marker='o')
        ax_evidence.plot(epochs, history['val_evidence'], label='Val Evidence', marker='s')
        ax_evidence.set_xlabel('Epoch')
        ax_evidence.set_ylabel('Mean Total Evidence')
        ax_evidence.set_title('Evidence')
        ax_evidence.legend()
        ax_evidence.grid(True)
        
        ax_uncertainty.clear()
        ax_uncertainty.plot(epochs, history['train_uncertainty'], label='Train Uncertainty', marker='o')
        ax_uncertainty.plot(epochs, history['val_uncertainty'], label='Val Uncertainty', marker='s')
        ax_uncertainty.set_xlabel('Epoch')
        ax_uncertainty.set_ylabel('Mean Uncertainty')
        ax_uncertainty.set_title('Uncertainty')
        ax_uncertainty.legend()
        ax_uncertainty.grid(True)
        
        fig.tight_layout()
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            pass
    
    for epoch in range(num_epochs):
        print(f"\n{'='*60}")
        print(f"轮次 [{epoch+1}/{num_epochs}]")
        print(f"{'='*60}")
        
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr:.6f}")
        
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        train_evidence_list = []
        train_uncertainty_list = []
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            
            # EDL训练逻辑
            y = one_hot_embedding(labels, num_classes)
            y = y.to(device)
            
            # 前向传播（不需要return_features，因为不计算Center Loss）
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            _, preds = torch.max(outputs, 1)
            
            # 计算EDL损失（仅EDL，不含Center Loss）
            loss = criterion(
                outputs, y.float(), epoch, num_classes, annealing_step, device
            )
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item() * inputs.size(0)
            train_correct += torch.sum(preds == labels.data).item()
            train_total += labels.size(0)
            
            # 计算证据统计（每10个batch计算一次，减少计算开销）
            if batch_idx % 10 == 0:
                with torch.no_grad():
                    stats = compute_evidence_stats(outputs, device)
                    train_evidence_list.append(stats['mean_total_evidence'])
                    train_uncertainty_list.append(stats['mean_uncertainty'])
        
        train_loss = train_loss / len(train_dataset)
        train_acc = 100.0 * train_correct / train_total
        train_mean_evidence = np.mean(train_evidence_list) if train_evidence_list else 0.0
        train_mean_uncertainty = np.mean(train_uncertainty_list) if train_uncertainty_list else 0.0
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        val_evidence_list = []
        val_uncertainty_list = []
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                y = one_hot_embedding(labels, num_classes)
                y = y.to(device)
                
                outputs = model(inputs)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                
                _, preds = torch.max(outputs, 1)
                
                loss = criterion(
                    outputs, y.float(), epoch, num_classes, annealing_step, device
                )
                
                val_loss += loss.item() * inputs.size(0)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)
                
                # 计算证据统计
                stats = compute_evidence_stats(outputs, device)
                val_evidence_list.append(stats['mean_total_evidence'])
                val_uncertainty_list.append(stats['mean_uncertainty'])
        
        val_loss = val_loss / len(val_dataset)
        val_acc = 100.0 * val_correct / val_total
        val_mean_evidence = np.mean(val_evidence_list) if val_evidence_list else 0.0
        val_mean_uncertainty = np.mean(val_uncertainty_list) if val_uncertainty_list else 0.0
        
        # 更新学习率
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
        
        # 记录历史
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['train_evidence'].append(train_mean_evidence)
        history['train_uncertainty'].append(train_mean_uncertainty)
        history['val_evidence'].append(val_mean_evidence)
        history['val_uncertainty'].append(val_mean_uncertainty)
        
        # 打印结果
        print(f"\n训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"        平均证据值: {train_mean_evidence:.4f}, 平均不确定性: {train_mean_uncertainty:.4f}")
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
        print(f"        平均证据值: {val_mean_evidence:.4f}, 平均不确定性: {val_mean_uncertainty:.4f}")
        
        # 更新实时曲线
        update_plot()
        
        # 保存训练曲线
        try:
            curve_path = os.path.join(checkpoint_dir, 'finetuning_curves.png')
            fig.savefig(curve_path, dpi=150)
        except Exception as e:
            print(f"保存训练曲线时出错: {e}")
        
        # 保存最佳模型（基于验证准确率和证据值）
        if val_acc > best_val_acc or (val_acc == best_val_acc and val_mean_evidence > best_evidence):
            if val_acc > best_val_acc:
                best_val_acc = val_acc
            if val_mean_evidence > best_evidence:
                best_evidence = val_mean_evidence
            
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"✓ 保存最佳模型: 验证准确率={val_acc:.2f}%, 平均证据值={val_mean_evidence:.4f}")
    
    train_end_time = datetime.now()
    
    # 9. 保存最终结果
    print(f"\n{'='*60}")
    print("微调完成！")
    print(f"{'='*60}")
    
    # 保存最终训练曲线
    try:
        curve_path = os.path.join(checkpoint_dir, 'finetuning_curves.png')
        fig.savefig(curve_path, dpi=150)
        print(f"训练曲线已保存至: {curve_path}")
    except Exception as e:
        print(f"保存训练曲线时出错: {e}")
    
    plt.ioff()
    try:
        plt.close(fig)
    except Exception:
        pass
    
    # 保存实验信息
    finetune_config = config.copy()
    finetune_config['train'] = train_config.copy()
    finetune_config['train']['checkpoint_dir'] = checkpoint_dir
    finetune_config['train']['learning_rate'] = finetune_lr
    finetune_config['train']['num_epochs'] = num_epochs
    finetune_config['train']['notes'] = f"Fine-tuning from {args.checkpoint}. Frozen backbone, only classifier trained."
    
    save_experiment_info(
        config=finetune_config,
        checkpoint_dir=checkpoint_dir,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        history=history,
        best_val_acc=best_val_acc,
        start_time=train_start_time,
        end_time=train_end_time
    )
    
    print(f"\n最终结果:")
    print(f"  最佳验证准确率: {best_val_acc:.2f}%")
    print(f"  最佳平均证据值: {best_evidence:.4f}")
    print(f"  最终验证不确定性: {history['val_uncertainty'][-1]:.4f}")
    print(f"\n模型已保存至: {checkpoint_dir}")


if __name__ == '__main__':
    main()
