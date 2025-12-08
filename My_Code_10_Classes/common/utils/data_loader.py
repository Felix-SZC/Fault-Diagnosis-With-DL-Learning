import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image

class LabeledImageDataset(Dataset):
    """
    通用的图像数据集类。
    从文件名 'name_label.ext' 中解析标签。
    适用于 Handcraft_ResNet, Handcraft_MYCNN 等项目。
    """
    def __init__(self, path, transform=None):
        self.path = path
        self.files = [f for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_name = os.path.join(self.path, self.files[idx])
        
        # 尝试以灰度图打开，如果失败则以RGB模式打开
        try:
            image = Image.open(img_name).convert('L')
        except IOError:
            image = Image.open(img_name).convert('RGB')
        
        label = int(self.files[idx].split('_')[-1].split('.')[0])
        
        if self.transform:
            image = self.transform(image)
            
        return image, label

class RawSignalDataset(Dataset):
    """
    读取预处理的原始1D振动信号数据：
    - split_dir 下应包含 index.csv（两列：file,label）
    - 每一行 file 指向同目录下的一个 .npy 文件
    - .npy 内容为 (time_steps,) 的一维数组（例如 (1024,)）
    - __getitem__ 返回 (Tensor[1, time_steps], int_label)
    
    适用于 RSBU, RSBU_OpenMax 等项目。
    
    可选参数：
    - snr_db: 信噪比 (dB)。如果提供，则会为每个样本添加高斯噪声以达到此信噪比。
      如果为 None，则不添加噪声。(默认: None)
    """
    def __init__(self, split_dir: str, snr_db: float = None):
        index_path = os.path.join(split_dir, 'index.csv')
        if not os.path.exists(index_path):
            # 兼容处理：如果没有 index.csv，可能是旧的目录结构或者数据未生成
            raise FileNotFoundError(f'未找到索引文件: {index_path}')
        self.split_dir = split_dir
        self.index_df = pd.read_csv(index_path)
        if 'file' not in self.index_df.columns or 'label' not in self.index_df.columns:
            raise ValueError('index.csv 需包含列: file,label')
        self.snr_db = snr_db

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        path = os.path.join(self.split_dir, str(row['file']))
        # 加载单个样本（原始1D信号）
        arr = np.load(path)  # numpy.ndarray, 形状：(time_steps,)
        # 转换为tensor
        tensor = torch.from_numpy(arr).float()
        
        # 如果需要添加高斯噪声
        if self.snr_db is not None:
            # 1. 计算信号功率
            signal_power = torch.mean(tensor.pow(2))
            
            # 2. 从信噪比(dB)计算噪声功率
            snr = 10**(self.snr_db / 10.0)
            noise_power = signal_power / snr
            
            # 3. 计算噪声标准差
            noise_std = torch.sqrt(noise_power)
            
            # 4. 生成并添加噪声
            noise = torch.randn_like(tensor) * noise_std
            tensor = tensor + noise

        # 添加通道维度: (time_steps,) -> (1, time_steps)
        tensor = tensor.unsqueeze(0)
        
        label = int(row['label'])
        return tensor, label

class NpyIndexDataset(Dataset):
    """
    读取预计算的 WPT 数据：
    - split_dir 下应包含 index.csv（两列：file,label）
    - 每一行 file 指向同目录下的一个 .npy 文件
    - .npy 内容为 (bands, time) 的浮点矩阵（例如 (16, 1024)）
    - __getitem__ 返回 (Tensor[bands, time], int_label)
    
    适用于 TimeFreqAttention_Model 等项目。
    """
    def __init__(self, split_dir: str):
        index_path = os.path.join(split_dir, 'index.csv')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f'未找到索引文件: {index_path}')
        self.split_dir = split_dir
        self.index_df = pd.read_csv(index_path)
        if 'file' not in self.index_df.columns or 'label' not in self.index_df.columns:
            raise ValueError('index.csv 需包含列: file,label')

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        path = os.path.join(self.split_dir, str(row['file']))
        # 加载单个样本（小波包系数矩阵）
        arr = np.load(path)               # numpy.ndarray, 形状：(bands, time)
        tensor = torch.from_numpy(arr).float()
        label = int(row['label'])
        return tensor, label

