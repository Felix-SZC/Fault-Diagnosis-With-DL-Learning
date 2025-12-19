"""
LeNet网络模型定义
经典的卷积神经网络，用于MNIST手写数字分类
"""
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


class LeNet(nn.Module):
    """
    LeNet网络结构
    
    网络架构：
    - 输入: 1×28×28 灰度图像
    - Conv1: 1 -> 20 通道，5×5卷积核
    - Conv2: 20 -> 50 通道，5×5卷积核
    - FC1: 20000 -> 500 全连接层
    - FC2: 500 -> 10 输出层（10个类别）
    
    注意：在不确定性模式下，最后一层输出的是logits（未经过softmax），
         会通过relu_evidence转换为证据，再加1得到Dirichlet参数alpha
    """
    
    def __init__(self, dropout=False):
        """
        初始化LeNet网络
        
        Args:
            dropout: 是否在全连接层使用dropout正则化
        """
        super().__init__()
        self.use_dropout = dropout
        
        # 第一层卷积：1个输入通道 -> 20个输出通道，5×5卷积核
        # 输入: [batch, 1, 28, 28] -> 输出: [batch, 20, 24, 24]
        self.conv1 = nn.Conv2d(1, 20, kernel_size=5)
        
        # 第二层卷积：20个输入通道 -> 50个输出通道，5×5卷积核
        # 输入: [batch, 20, 12, 12] -> 输出: [batch, 50, 8, 8]
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5)
        
        # 第一个全连接层：20000 -> 500
        # 20000 = 50 * 20 * 20（经过池化后的特征图尺寸）
        self.fc1 = nn.Linear(20000, 500)
        
        # 输出层：500 -> 10（10个类别）
        # 注意：这里输出的是logits，不是概率
        self.fc2 = nn.Linear(500, 10)

    def forward(self, x):
        """
        前向传播
        
        Args:
            x: 输入图像张量，形状为 [batch_size, 1, 28, 28]
        
        Returns:
            torch.Tensor: 输出logits，形状为 [batch_size, 10]
        """
        # 第一层：卷积 -> ReLU激活 -> 最大池化
        # 注意：这里池化大小是1，实际上没有进行下采样
        x = F.relu(F.max_pool2d(self.conv1(x), 1))
        
        # 第二层：卷积 -> ReLU激活 -> 最大池化
        x = F.relu(F.max_pool2d(self.conv2(x), 1))
        
        # 展平：将特征图展平为一维向量
        # [batch, 50, 20, 20] -> [batch, 20000]
        x = x.view(x.size()[0], -1)
        
        # 第一个全连接层 + ReLU激活
        x = F.relu(self.fc1(x))
        
        # 可选的dropout正则化（仅在训练时生效）
        if self.use_dropout:
            x = F.dropout(x, training=self.training)
        
        # 输出层：生成10个类别的logits
        # 在不确定性模式下，这些logits会转换为证据
        x = self.fc2(x)
        return x
