"""
辅助函数模块
提供设备选择、one-hot编码等工具函数
"""
import torch


def get_device():
    """
    自动检测并返回可用的计算设备（GPU或CPU）
    
    Returns:
        torch.device: 如果检测到CUDA则返回 'cuda:0'，否则返回 'cpu'
    """
    use_cuda = torch.cuda.is_available()  # 检查是否有可用的CUDA设备
    device = torch.device("cuda:0" if use_cuda else "cpu")
    return device


def one_hot_embedding(labels, num_classes=10):
    """
    将整数标签转换为one-hot编码向量
    
    Args:
        labels: 整数标签张量，形状为 [batch_size] 或 [batch_size, 1]
        num_classes: 类别总数，默认为10（MNIST有10个类别）
    
    Returns:
        torch.Tensor: one-hot编码张量，形状为 [batch_size, num_classes]
                     例如：标签2 -> [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]
    
    示例:
        >>> labels = torch.tensor([0, 2, 4])
        >>> one_hot = one_hot_embedding(labels, num_classes=10)
        >>> # 返回形状为 [3, 10] 的one-hot编码
    """
    # 创建单位矩阵，每一行对应一个类别的one-hot编码
    # device=labels.device 确保在正确的设备上创建（CPU或GPU）
    y = torch.eye(num_classes, device=labels.device) 
    # 使用标签作为索引，从单位矩阵中选择对应的行
    return y[labels]
