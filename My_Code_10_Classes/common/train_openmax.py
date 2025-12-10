import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from datetime import datetime
import pickle

# 添加项目根目录到 sys.path，确保可以导入 common, models 等模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config, save_experiment_info
from common.utils.data_loader import RawSignalDataset, NpyIndexDataset
from common.utils.trainer import train_one_epoch, validate
from common.openmax import compute_mavs, fit_weibull
from models import get_model

def get_dataset(data_config, split='train'):
    """根据配置选择并实例化数据集 (Open-set 版本)"""
    # OpenMax 训练脚本要求配置中必须明确指定数据类型和已知类别
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
    else:
        raise ValueError(f"OpenMax 训练脚本目前仅支持 'raw_signal' 和 'wpt' 数据类型，当前类型: {data_type}")

def main():
    parser = argparse.ArgumentParser(description='Unified Training Script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
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
    
    # 设置 snr_db（噪声增强），仅对 RawSignalDataset 生效
    if isinstance(train_dataset, RawSignalDataset):
        snr_db = train_config.get('snr_db', None)
        train_dataset.snr_db = snr_db
        if snr_db is not None:
            print(f"训练集启用高斯噪声增强，SNR: {snr_db} dB")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 3. 准备模型
    # 设备选择：优先使用配置文件中的设置，但需要验证 CUDA 是否真的可用
    device_config = train_config.get('device', None)
    cuda_available = torch.cuda.is_available()
    
    if device_config:
        requested_device = device_config.lower()
        print(f"从配置文件读取设备: {requested_device}")
        
        # 如果请求使用 CUDA 但 CUDA 不可用，则回退到 CPU
        if requested_device == 'cuda' and not cuda_available:
            print(f"警告: 配置文件中指定使用 CUDA，但 PyTorch 未编译 CUDA 支持，将使用 CPU")
            device = torch.device('cpu')
        else:
            device = torch.device(requested_device)
    else:
        # 自动检测 CUDA 可用性
        print(f"CUDA 可用性检测: {cuda_available}")
        if cuda_available:
            print(f"检测到 {torch.cuda.device_count()} 个 GPU")
            print(f"GPU 名称: {torch.cuda.get_device_name(0)}")
        device = torch.device('cuda' if cuda_available else 'cpu')
    print(f"使用设备: {device}")
    
    model = get_model(model_config.get('type'), num_classes=model_config.get('num_classes')).to(device)

    # 4. 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    learning_rate = float(train_config['learning_rate'])
    
    # 优化器配置
    optimizer_type = train_config.get('optimizer', 'Adam') # 默认为 Adam
    if optimizer_type == 'SGD':
        # 确保数值类型正确（YAML可能将科学计数法解析为字符串）
        momentum = float(train_config.get('momentum', 0.9))
        weight_decay = float(train_config.get('weight_decay', 1e-4))
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=momentum, 
            weight_decay=weight_decay
        )
        print(f"使用优化器: SGD (momentum={momentum}, weight_decay={weight_decay})")
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    print(f"使用优化器: {optimizer_type}")

    # 学习率调度器
    scheduler_conf = train_config.get('scheduler', {})
    if scheduler_conf.get('use_scheduler', False):
        sch_type = scheduler_conf.get('scheduler_type')
        
        if sch_type == 'ReduceLROnPlateau':
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, mode='min', 
                factor=scheduler_conf.get('gamma', 0.1), 
                patience=5
            )
        elif sch_type == 'MultiStepLR':
            scheduler = optim.lr_scheduler.MultiStepLR(
                optimizer,
                milestones=scheduler_conf.get('milestones', [40, 80]),
                gamma=scheduler_conf.get('gamma', 0.1)
            )
        else: # 默认 StepLR
            scheduler = optim.lr_scheduler.StepLR(
                optimizer, 
                step_size=scheduler_conf.get('step_size', 10), 
                gamma=scheduler_conf.get('gamma', 0.1)
            )
    else:
        # 简单的默认调度器
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

    # 5. 训练循环
    checkpoint_dir = train_config['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    num_epochs = train_config['num_epochs']
    best_val_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    train_start_time = datetime.now()
    
    # 初始保存实验信息
    save_experiment_info(config, checkpoint_dir, model=model, train_dataset=train_dataset, val_dataset=val_dataset, start_time=train_start_time)

    # 实时绘图初始化
    plt.ion()
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    fig.suptitle('Training Progress')
    
    def update_plot():
        """更新实时训练曲线"""
        epochs = range(1, len(history['train_loss']) + 1)
        ax1.clear()
        ax2.clear()
        ax1.plot(epochs, history['train_loss'], label='Train Loss')
        ax1.plot(epochs, history['val_loss'], label='Val Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        ax2.plot(epochs, history['train_acc'], label='Train Acc')
        ax2.plot(epochs, history['val_acc'], label='Val Acc')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        fig.tight_layout()
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            pass

    for epoch in range(num_epochs):
        print(f"\n轮次 [{epoch+1}/{num_epochs}]")
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"当前学习率: {current_lr}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
            
        print(f"训练集 - 损失: {train_loss:.4f}, 准确率: {train_acc:.2f}%")
        print(f"验证集 - 损失: {val_loss:.4f}, 准确率: {val_acc:.2f}%")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # 更新实时曲线
        update_plot()
        
        # 每个 epoch 后保存一次曲线图（防止训练中断丢失）
        try:
            curve_path = os.path.join(checkpoint_dir, 'training_curves.png')
            fig.savefig(curve_path, dpi=150)
        except Exception as e:
            print(f"保存训练曲线时出错: {e}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, 'best_model.pth'))
            print(f"保存最佳模型: {val_acc:.2f}%")
            
    train_end_time = datetime.now()
    
    # 6. 保存最终结果
    # 保存训练曲线图
    try:
        curve_path = os.path.join(checkpoint_dir, 'training_curves.png')
        fig.savefig(curve_path, dpi=150)
        print(f"训练曲线已保存至: {curve_path}")
    except Exception as e:
        print(f"保存训练曲线时出错: {e}")
    
    # 关闭交互模式
    plt.ioff()
    try:
        plt.close(fig)
    except Exception:
        pass
    
    # 更新实验信息
    save_experiment_info(
        config=config,
        checkpoint_dir=checkpoint_dir,
        model=model,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        history=history,
        best_val_acc=best_val_acc,
        start_time=train_start_time,
        end_time=train_end_time
    )
    print("\n训练完成！")

    # --- OpenMax 后处理 ---
    print("\n开始 OpenMax 后处理...")
    
    # 1. 加载最佳模型
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    print("已加载最佳模型进行 OpenMax 计算。")

    # 2. 重新创建训练数据加载器（不打乱，用于 MAV 计算）
    openset_train_dataset = get_dataset(data_config, 'train')
    openset_train_loader = DataLoader(openset_train_dataset, batch_size=batch_size, shuffle=False)

    # 3. 计算 MAVs 和 Weibull 模型
    num_known_classes = len(data_config['openset']['known_classes'])
    
    print("正在计算 MAVs (使用 Deep Features)...")
    mavs, all_features, correct_labels = compute_mavs(model, openset_train_loader, num_known_classes, device)
    
    print("正在拟合 Weibull 模型 (使用 Deep Features)...")
    # fit_weibull 现在使用 features 而不是 logits
    weibull_models = fit_weibull(mavs, all_features, correct_labels)
    
    # 4. 保存 OpenMax 所需文件
    openmax_dir = os.path.join(checkpoint_dir, 'openmax_files')
    os.makedirs(openmax_dir, exist_ok=True)
    
    torch.save(mavs, os.path.join(openmax_dir, 'mavs.pth'))
    
    with open(os.path.join(openmax_dir, 'weibull_models.pkl'), 'wb') as f:
        pickle.dump(weibull_models, f)
        
    print(f"OpenMax MAVs 和 Weibull 模型已保存至: {openmax_dir}")


if __name__ == '__main__':
    main()
