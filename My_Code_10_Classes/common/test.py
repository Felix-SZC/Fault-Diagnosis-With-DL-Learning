import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config
from common.utils.data_loader import LabeledImageDataset, RawSignalDataset, NpyIndexDataset
from models import get_model

def get_dataset(data_config, split='test'):
    """复用 train.py 中的逻辑，但针对测试集"""
    data_type = data_config.get('type', 'unknown')
    
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
        return RawSignalDataset(split_dir=split_dir) # 测试集不加噪
        
    elif data_type == 'wpt':
        base_dir = data_config.get('wpt_output_dir')
        split_dir = os.path.join(base_dir, split)
        return NpyIndexDataset(split_dir=split_dir)
    
    else:
        raise ValueError(f"Unknown data type: {data_type}")

def main():
    parser = argparse.ArgumentParser(description='Unified Testing Script')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    train_config = config['train'] # 需要读取 checkpoint_dir

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 加载数据
    test_dataset = get_dataset(data_config, split='test')
    test_loader = DataLoader(test_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
    print(f"测试集大小: {len(test_dataset)}")

    # 加载模型
    if isinstance(test_dataset, NpyIndexDataset):
        sample, _ = test_dataset[0]
        input_channels = sample.shape[0]
        model = get_model(model_config.get('type'), input_channels=input_channels, **model_config).to(device)
    else:
        model = get_model(model_config.get('type'), num_classes=model_config.get('num_classes')).to(device)

    # 加载权重
    checkpoint_dir = train_config['checkpoint_dir']
    model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    
    if not os.path.exists(model_path):
        print(f"错误: 未找到模型文件 {model_path}")
        return
        
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    print(f"已加载模型: {model_path}")

    # 推理
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # 报告
    print("\n测试完成！分类报告：")
    report = classification_report(all_labels, all_preds)
    print(report)
    
    # 保存报告
    with open(os.path.join(checkpoint_dir, 'test_report.txt'), 'w') as f:
        f.write(report)

    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(os.path.join(checkpoint_dir, 'test_confusion_matrix.png'))
    print(f"结果已保存至 {checkpoint_dir}")

if __name__ == '__main__':
    main()

