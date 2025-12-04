import yaml
import os
from datetime import datetime
import torch

def load_config(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def save_config(config, file_path):
    """
    将配置字典保存到 YAML 文件。
    """
    # 确保目录存在
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f)

def count_parameters(model):
    """
    计算模型的参数量。
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total_params,
        'trainable': trainable_params,
        'non_trainable': total_params - trainable_params
    }

def save_experiment_info(config, checkpoint_dir, model=None, train_dataset=None, 
                         val_dataset=None, test_dataset=None, history=None, 
                         best_val_acc=None, start_time=None, end_time=None):
    """
    保存实验信息到 YAML 和文本文件。
    
    Args:
        config: 配置字典
        checkpoint_dir: 保存目录
        model: 模型实例（可选，用于获取参数量）
        train_dataset: 训练数据集（可选，用于获取数据集大小）
        val_dataset: 验证数据集（可选）
        test_dataset: 测试数据集（可选）
        history: 训练历史字典（可选，包含 loss 和 acc）
        best_val_acc: 最佳验证准确率（可选）
        start_time: 训练开始时间（可选）
        end_time: 训练结束时间（可选）
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # 收集实验信息
    experiment_info = {
        'experiment': {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S') if start_time else None,
            'end_time': end_time.strftime('%Y-%m-%d %H:%M:%S') if end_time else None,
            'duration': None,
        },
        'config': config,
        'model': {},
        'data': {},
        'training': {},
        'results': {}
    }
    
    # 计算训练时长
    if start_time and end_time:
        duration = end_time - start_time
        experiment_info['experiment']['duration'] = str(duration)
        experiment_info['experiment']['duration_seconds'] = duration.total_seconds()
    
    # 模型信息
    if model is not None:
        model_config = config.get('model', {})
        experiment_info['model']['type'] = model_config.get('type', 'Unknown')
        experiment_info['model']['num_classes'] = model_config.get('num_classes', 'Unknown')
        experiment_info['model']['parameters'] = count_parameters(model)
        experiment_info['model']['device'] = str(next(model.parameters()).device)
    
    # 数据集信息
    dataset_info = {}
    if train_dataset is not None:
        dataset_info['train_size'] = len(train_dataset)
    if val_dataset is not None:
        dataset_info['val_size'] = len(val_dataset)
    if test_dataset is not None:
        dataset_info['test_size'] = len(test_dataset)
    
    data_config = config.get('data', {})
    dataset_info['raw_signal_output_dir'] = data_config.get('raw_signal_output_dir', 'Unknown')
    dataset_info['split_ratios'] = data_config.get('split_ratios', 'Unknown')
    experiment_info['data'] = dataset_info
    
    # 训练信息
    train_config = config.get('train', {})
    training_info = {
        'num_epochs': train_config.get('num_epochs', 'Unknown'),
        'batch_size': train_config.get('batch_size', 'Unknown'),
        'learning_rate': train_config.get('learning_rate', 'Unknown'),
        'optimizer': 'Adam',  # 当前硬编码为 Adam
        'criterion': 'CrossEntropyLoss',
        'checkpoint_dir': train_config.get('checkpoint_dir', 'Unknown'),
        'notes': train_config.get('notes', '')
    }
    experiment_info['training'] = training_info
    
    # 训练结果
    if history is not None:
        experiment_info['results']['history'] = {
            'train_loss': history.get('train_loss', []),
            'train_acc': history.get('train_acc', []),
            'val_loss': history.get('val_loss', []),
            'val_acc': history.get('val_acc', [])
        }
    
    if best_val_acc is not None:
        experiment_info['results']['best_val_acc'] = best_val_acc
    
    # 生成并保存文本文件（易读格式）
    txt_path = os.path.join(checkpoint_dir, 'experiment_info.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("实验信息记录\n")
        f.write("=" * 80 + "\n\n")
        
        # 实验基本信息
        f.write("【实验基本信息】\n")
        f.write("-" * 80 + "\n")
        if start_time:
            f.write(f"开始时间: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if end_time:
            f.write(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        if start_time and end_time:
            duration = end_time - start_time
            hours, remainder = divmod(duration.total_seconds(), 3600)
            minutes, seconds = divmod(remainder, 60)
            f.write(f"训练时长: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒\n")
        f.write("\n")
        
        # 模型信息
        if experiment_info['model']:
            f.write("【模型信息】\n")
            f.write("-" * 80 + "\n")
            f.write(f"模型类型: {experiment_info['model'].get('type', 'Unknown')}\n")
            f.write(f"类别数: {experiment_info['model'].get('num_classes', 'Unknown')}\n")
            if 'parameters' in experiment_info['model']:
                params = experiment_info['model']['parameters']
                f.write(f"总参数量: {params['total']:,}\n")
                f.write(f"可训练参数: {params['trainable']:,}\n")
                f.write(f"不可训练参数: {params['non_trainable']:,}\n")
            if 'device' in experiment_info['model']:
                f.write(f"训练设备: {experiment_info['model']['device']}\n")
            f.write("\n")
        
        # 数据集信息
        if experiment_info['data']:
            f.write("【数据集信息】\n")
            f.write("-" * 80 + "\n")
            data_info = experiment_info['data']
            if 'train_size' in data_info:
                f.write(f"训练集大小: {data_info['train_size']:,}\n")
            if 'val_size' in data_info:
                f.write(f"验证集大小: {data_info['val_size']:,}\n")
            if 'test_size' in data_info:
                f.write(f"测试集大小: {data_info['test_size']:,}\n")
            if 'split_ratios' in data_info and data_info['split_ratios'] != 'Unknown':
                ratios = data_info['split_ratios']
                f.write(f"数据集划分比例: {ratios}\n")
            f.write(f"数据目录: {data_info.get('raw_signal_output_dir', 'Unknown')}\n")
            f.write("\n")
        
        # 训练配置
        if experiment_info['training']:
            f.write("【训练配置】\n")
            f.write("-" * 80 + "\n")
            train_info = experiment_info['training']
            f.write(f"训练轮数: {train_info.get('num_epochs', 'Unknown')}\n")
            f.write(f"批次大小: {train_info.get('batch_size', 'Unknown')}\n")
            f.write(f"学习率: {train_info.get('learning_rate', 'Unknown')}\n")
            f.write(f"优化器: {train_info.get('optimizer', 'Unknown')}\n")
            f.write(f"损失函数: {train_info.get('criterion', 'Unknown')}\n")
            f.write(f"保存目录: {train_info.get('checkpoint_dir', 'Unknown')}\n")
            notes = train_info.get('notes', '').strip()
            if notes:
                f.write(f"\n实验说明:\n{notes}\n")
            f.write("\n")
        
        # 训练结果
        if experiment_info['results']:
            f.write("【训练结果】\n")
            f.write("-" * 80 + "\n")
            if 'best_val_acc' in experiment_info['results']:
                f.write(f"最佳验证准确率: {experiment_info['results']['best_val_acc']:.2f}%\n")
            if 'history' in experiment_info['results']:
                hist = experiment_info['results']['history']
                if hist.get('train_acc'):
                    final_train_acc = hist['train_acc'][-1] if hist['train_acc'] else None
                    final_val_acc = hist['val_acc'][-1] if hist['val_acc'] else None
                    if final_train_acc is not None:
                        f.write(f"最终训练准确率: {final_train_acc:.2f}%\n")
                    if final_val_acc is not None:
                        f.write(f"最终验证准确率: {final_val_acc:.2f}%\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
    
    print(f"\n实验信息已保存至: {txt_path}")