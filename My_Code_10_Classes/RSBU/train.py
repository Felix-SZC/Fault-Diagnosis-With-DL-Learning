import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm 
import matplotlib
import matplotlib.pyplot as plt
from datetime import datetime

from utils.data_loader import RawSignalDataset
from utils.helpers import load_config, save_experiment_info
from models import get_model

config = load_config('configs/config.yaml')

data_config = config['data']
train_config = config['train']
model_config = config['model']

# 使用原始信号数据目录
raw_signal_output_dir = data_config.get('raw_signal_output_dir', '../data/RawSignals')
train_dir = os.path.join(raw_signal_output_dir, 'train')
val_dir = os.path.join(raw_signal_output_dir, 'val')

batch_size = train_config['batch_size']
num_epochs = train_config['num_epochs']
checkpoint_dir = train_config['checkpoint_dir']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'使用设备:{device}')

os.makedirs(checkpoint_dir, exist_ok=True)
print(f'模型将保存至: {checkpoint_dir}')

print(f"从预处理目录读取: {raw_signal_output_dir}")
# 从配置读取噪声参数
add_noise = train_config.get('add_noise', False)
noise_std = train_config.get('noise_std', 0.05)
# 训练集添加噪声（如果启用），验证集不加噪声
train_dataset = RawSignalDataset(split_dir=train_dir, add_noise=add_noise, noise_std=noise_std)
val_dataset = RawSignalDataset(split_dir=val_dir, add_noise=False)
if add_noise:
    print(f"训练集已启用高斯噪声增强，噪声标准差: {noise_std}")
else:
    print("训练集未启用噪声增强")

train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

# 根据配置自动选择和实例化模型
model = get_model(model_config).to(device)
model_type = model_config.get('type', 'Unknown')
print(f'使用模型: {model_type}')
criterion = nn.CrossEntropyLoss()


def get_lr_for_epoch(epoch: int) -> float:
    """Return staged learning rate: [0-39]=0.1, [40-79]=0.01, [80-99]=0.001."""
    if epoch < 20:
        return 0.001
    if epoch < 30:
        return 0.005
    return 0.0001


def set_optimizer_lr(optimizer: optim.Optimizer, lr: float) -> None:
    """Update optimizer param groups with the provided lr."""
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


initial_lr = get_lr_for_epoch(0)
optimizer = optim.Adam(model.parameters(), lr=initial_lr)

# 记录训练开始时间
train_start_time = datetime.now()

# 训练开始前保存实验信息（配置信息）
print("\n保存实验配置信息...")
save_experiment_info(
    config=config,
    checkpoint_dir=checkpoint_dir,
    model=model,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    start_time=train_start_time
)

def train_one_epoch(model, train_loader, criterion, optimizer, device): 
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='训练中')

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

        progress_bar.set_postfix({'loss': loss.item()})
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = 100 * correct / total

    return epoch_loss, epoch_acc

def validate(model, val_loader, criterion, device):
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        progress_bar = tqdm(val_loader, desc='验证中')
        
        for images, labels in progress_bar:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            progress_bar.set_postfix({'loss': loss.item()})

        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

print("\n开始训练...\n")
print(f"训练集大小: {len(train_dataset)}")
print(f"验证集大小: {len(val_dataset)}")
print(f"Batch Size: {batch_size}")
print(f"总Epochs: {num_epochs}\n")

best_val_acc = 0.0

# 指标记录
history = {
    'train_loss': [],
    'train_acc': [],
    'val_loss': [],
    'val_acc': []
}

# 实时绘图初始化（若环境不支持可忽略）
plt.ion()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle('Training Progress')

def update_plot():
    epochs = range(1, len(history['train_loss']) + 1)
    ax1.clear(); ax2.clear()
    ax1.plot(epochs, history['train_loss'], label='train_loss')
    ax1.plot(epochs, history['val_loss'], label='val_loss')
    ax1.set_xlabel('Epoch'); ax1.set_ylabel('Loss'); ax1.legend(); ax1.grid(True)
    ax2.plot(epochs, history['train_acc'], label='train_acc')
    ax2.plot(epochs, history['val_acc'], label='val_acc')
    ax2.set_xlabel('Epoch'); ax2.set_ylabel('Acc (%)'); ax2.legend(); ax2.grid(True)
    fig.tight_layout()
    try:
        fig.canvas.draw()
        fig.canvas.flush_events()
    except Exception:
        pass

for epoch in range(num_epochs):
    print(f"\n{'='*50}")
    print(f"Epoch [{epoch+1}/{num_epochs}]")
    print(f"{'='*50}")

    current_lr = get_lr_for_epoch(epoch)
    set_optimizer_lr(optimizer, current_lr)
    print(f"当前学习率: {current_lr}")
    
    train_loss, train_acc = train_one_epoch(model=model, train_loader=train_loader, 
                                criterion=criterion, optimizer = optimizer, device=device)
    val_loss, val_acc = validate(model=model, val_loader=val_loader, 
                            criterion=criterion, device=device)
    print(f"\n训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%")
    print(f"验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%")

    # 记录历史并更新曲线
    history['train_loss'].append(train_loss)
    history['train_acc'].append(train_acc)
    history['val_loss'].append(val_loss)
    history['val_acc'].append(val_acc)
    update_plot()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
        torch.save(model.state_dict(), best_model_path)
        print(f"保存最佳模型，验证准确率: {val_acc:.2f}%")
# 记录训练结束时间
train_end_time = datetime.now()

print("\n" + "="*50)
print("训练完成!")
print(f"最佳验证准确率: {best_val_acc:.2f}%")
print("="*50)

# 训练结束后，保存最终曲线图
final_curve_path = os.path.join(checkpoint_dir, 'training_curves.png')
fig.savefig(final_curve_path, dpi=150)
plt.ioff()
try:
    plt.close(fig)
except Exception:
    pass

# 训练结束后更新实验信息（包含完整训练结果）
print("\n更新实验信息（包含训练结果）...")
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