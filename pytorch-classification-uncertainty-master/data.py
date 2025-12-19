"""
数据加载模块
负责下载、加载和预处理MNIST数据集
"""
import torch
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# 加载MNIST训练集
# download=True: 如果数据不存在则自动下载
# train=True: 使用训练集（60000张图片）
# transform: 将PIL图像转换为张量，并归一化到[0,1]
data_train = MNIST("./data/mnist",
                   download=True,
                   train=True,
                   transform=transforms.Compose([transforms.ToTensor()]))

# 加载MNIST验证/测试集
# train=False: 使用测试集（10000张图片）
data_val = MNIST("./data/mnist",
                 train=False,
                 download=True,
                 transform=transforms.Compose([transforms.ToTensor()]))

# 创建训练数据加载器
# batch_size=1000: 每批处理1000张图片
# shuffle=True: 每个epoch随机打乱数据顺序
# num_workers=0: Windows系统建议设为0，避免多进程问题
dataloader_train = DataLoader(
    data_train, batch_size=1000, shuffle=True, num_workers=0)

# 创建验证数据加载器
# shuffle=False: 验证集不需要打乱
dataloader_val = DataLoader(data_val, batch_size=1000, num_workers=0)

# 将数据加载器组织成字典，方便在训练和验证时使用
dataloaders = {
    "train": dataloader_train,
    "val": dataloader_val,
}

# 从验证集中提取一张数字"1"的图片，用于后续的旋转实验
# 索引5通常对应一张数字"1"的图片
digit_one, _ = data_val[5]
