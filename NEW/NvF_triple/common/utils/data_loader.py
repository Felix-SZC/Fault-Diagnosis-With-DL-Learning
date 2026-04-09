import os
import torch
import numpy as np
from torch.utils.data import Dataset
from .augmentation import SignalAugmenter

class NpyPackDataset(Dataset):
    """
    针对打包的 npy 数据集：
    - 读取 data_dir 下的 X_{split}.npy 和 y_{split}.npy
    - X: 形状为 (N, 64, 64) 的 numpy 数组
    - y: 形状为 (N,) 的字符串标签或数值标签的 numpy 数组
    - 输出 (Tensor[1, 64, 64], int_label) 以匹配 2D 卷积
    """
    def __init__(self, data_dir: str, split: str = 'train', 
                 filter_classes: list = None, 
                 known_classes: list = None,
                 unknown_classes: list = None,
                 augment_config: dict = None):
        actual_split = 'test' if split == 'val' else split
        
        self.split = split
        self.augmenter = SignalAugmenter(augment_config) if augment_config else None
        
        x_path = os.path.join(data_dir, f'X_{actual_split}.npy')
        y_path = os.path.join(data_dir, f'y_{actual_split}.npy')
        
        if not os.path.exists(x_path) or not os.path.exists(y_path):
            # 兼容大写 Y_
            y_path_upper = os.path.join(data_dir, f'Y_{actual_split}.npy')
            if os.path.exists(y_path_upper):
                y_path = y_path_upper
            else:
                raise FileNotFoundError(f"找不到数据文件：{x_path} 或 {y_path}")
            
        self.X = np.load(x_path)
        self.y = np.load(y_path)
        
        # 建立全局标签映射 (保证已知类占据 0..K-1，未知类占据后面)
        all_classes = (known_classes or []) + (unknown_classes or [])
        # 去重并保持顺序（已知类在前）
        unique_all_classes = []
        for c in all_classes:
            if c not in unique_all_classes:
                unique_all_classes.append(c)
                
        # 建立全局从原始类别 -> 整数的映射
        self.global_label_map = {orig: i for i, orig in enumerate(unique_all_classes)}
        
        # 如果 self.y 中存在未知于 global_label_map 的标签（比如多余的类），为了稳妥起见我们给它们分配一个新的ID
        current_max_id = len(unique_all_classes)
        
        if filter_classes is not None:
            print(f"[{split}] 正在筛选数据集，只保留类别: {filter_classes}")
            mask = np.isin(self.y, filter_classes)
            self.X = self.X[mask]
            self.y = self.y[mask]
            
        # 将所有的字符串/原始标签映射为整数
        mapped_y = []
        for label in self.y:
            # y 可能读出来是 np.str_，需要提取 Python 类型
            lbl_val = label.item() if hasattr(label, 'item') else label
            if lbl_val not in self.global_label_map:
                self.global_label_map[lbl_val] = current_max_id
                current_max_id += 1
            mapped_y.append(self.global_label_map[lbl_val])
            
        self.y = np.array(mapped_y, dtype=np.int64)
        
        # 为了兼容测试脚本中对 label_map 的提取（它假设 ID 类的映射）
        self.label_map = {orig: self.global_label_map[orig] for orig in (known_classes or [])}
        print(f"[{split}] 数据集大小: {len(self.y)}, 已知类标签映射: {self.label_map}")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_sample = self.X[idx]
        # (64, 64) -> (1, 64, 64) 转换为 2D CNN 输入所需的单通道
        tensor = torch.from_numpy(x_sample).float().unsqueeze(0)
        
        if self.split == 'train' and self.augmenter is not None:
            tensor = self.augmenter(tensor)
            
        label = self.y[idx]
        return tensor, label
