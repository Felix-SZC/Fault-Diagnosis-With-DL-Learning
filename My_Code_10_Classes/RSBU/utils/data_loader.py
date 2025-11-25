import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class LabeledImageDataset(Dataset):
    """
    通用的图像数据集类。
    从文件名 'name_label.ext' 中解析标签。
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
    """
    def __init__(self, split_dir: str):
        index_path = os.path.join(split_dir, 'index.csv')
        if not os.path.exists(index_path):
            raise FileNotFoundError(f'未找到索引文件: {index_path}，请先运行 make_raw_signal_dataset.py')
        self.split_dir = split_dir
        self.index_df = pd.read_csv(index_path)
        if 'file' not in self.index_df.columns or 'label' not in self.index_df.columns:
            raise ValueError('index.csv 需包含列: file,label')

    def __len__(self):
        return len(self.index_df)

    def __getitem__(self, idx):
        row = self.index_df.iloc[idx]
        path = os.path.join(self.split_dir, str(row['file']))
        # 加载单个样本（原始1D信号）
        arr = np.load(path)  # numpy.ndarray, 形状：(time_steps,)
        # 转换为tensor并添加通道维度：(time_steps,) -> (1, time_steps)
        tensor = torch.from_numpy(arr).float()
        # 添加通道维度
        tensor = tensor.unsqueeze(0)  # (time_steps,) -> (1, time_steps)
        label = int(row['label'])
        return tensor, label