import argparse
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime

# 添加项目根目录到 sys.path，确保可以导入 common, models 等模块
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config, save_experiment_info
from common.utils.data_loader import LabeledImageDataset, RawSignalDataset, NpyIndexDataset
from common.utils.trainer import train_one_epoch, validate
from models import get_model

def get_dataset(data_config, split='train'):
    """根据配置选择并实例化数据集"""
    data_type = data_config.get('type', 'unknown')
    
    # 自动推断类型（兼容旧配置）
    if data_type == 'unknown':
        if 'img_output_dir' in data_config:
            data_type = 'image'
        elif 'raw_signal_output_dir' in data_config:
            data_type = 'raw_signal'
        elif 'wpt_output_dir' in data_config:
            data_type = 'wpt'
    
    if data_type == 'image':
        base_dir = data_config.get('img_output_dir')
        path = os.path.join(base_dir, split)
        transform = transforms.Compose([transforms.ToTensor()])
        return LabeledImageDataset(path=path, transform=transform)
        
    elif data_type == 'raw_signal':
        base_dir = data_config.get('raw_signal_output_dir')
        split_dir = os.path.join(base_dir, split)
        # 仅训练集支持加噪
        snr_db = None
        # 注意：这里需要从外部传入 snr_db，目前简化处理，后续优化
        # 实际逻辑应在 main 函数中处理
        return RawSignalDataset(split_dir=split_dir, snr_db=None) 
        
    elif data_type == 'wpt':
        base_dir = data_config.get('wpt_output_dir')
        split_dir = os.path.join(base_dir, split)
        return NpyIndexDataset(split_dir=split_dir)
    
    else:
        raise ValueError(f"Unknown data type: {data_type} or insufficient config to infer type.")

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
    
    # 特殊处理：RawSignalDataset 的 snr_db 参数
    train_dataset = get_dataset(data_config, split='train')
    if isinstance(train_dataset, RawSignalDataset):
        snr_db = train_config.get('snr_db', None)
        train_dataset.snr_db = snr_db
        if snr_db is not None:
            print(f"训练集启用高斯噪声增强，SNR: {snr_db} dB")
            
    val_dataset = get_dataset(data_config, split='val')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"训练集大小: {len(train_dataset)}")
    print(f"验证集大小: {len(val_dataset)}")

    # 3. 准备模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 获取输入通道数（针对 TimeFreqAttention 等动态输入模型）
    if isinstance(train_dataset, NpyIndexDataset):
        sample, _ = train_dataset[0]
        input_channels = sample.shape[0]
        # 传递给 create_model
        model = get_model(model_config.get('type'), input_channels=input_channels, **model_config).to(device)
    else:
        # 其他模型通常只需要 num_classes
        model = get_model(model_config.get('type'), num_classes=model_config.get('num_classes')).to(device)

    # 4. 优化器与损失函数
    criterion = nn.CrossEntropyLoss()
    learning_rate = train_config['learning_rate']
    
    # 优化器配置
    optimizer_type = train_config.get('optimizer', 'Adam') # 默认为 Adam
    if optimizer_type == 'SGD':
        optimizer = optim.SGD(
            model.parameters(), 
            lr=learning_rate, 
            momentum=train_config.get('momentum', 0.9), 
            weight_decay=train_config.get('weight_decay', 1e-4)
        )
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
        ax2.set_ylabel('Acc (%)')
        ax2.legend()
        ax2.grid(True)
        fig.tight_layout()
        try:
            fig.canvas.draw()
            fig.canvas.flush_events()
        except Exception:
            pass

    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        
        # 打印当前学习率
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current Learning Rate: {current_lr}")
        
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 更新学习率
        if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_loss)
        else:
            scheduler.step()
            
        print(f"Train - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
        print(f"Val   - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")
        
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

if __name__ == '__main__':
    main()
