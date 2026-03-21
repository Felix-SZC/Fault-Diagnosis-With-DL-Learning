import os
import numpy as np
import torch
from torch.utils.data import Dataset


class NpyPackDataset1D(Dataset):
    """
    读取打包 npy 数据并输出 1D Conv 输入：
    - X: 期望 (N, 64, 64)
    - y: 形状 (N,)
    - 输出: Tensor[64, 64]，解释为 [C, L]
    """

    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        filter_classes: list = None,
        known_classes: list = None,
        unknown_classes: list = None,
    ):
        actual_split = "test" if split == "val" else split
        x_path = os.path.join(data_dir, f"X_{actual_split}.npy")
        y_path = os.path.join(data_dir, f"y_{actual_split}.npy")

        if not os.path.exists(x_path) or not os.path.exists(y_path):
            y_path_upper = os.path.join(data_dir, f"Y_{actual_split}.npy")
            if os.path.exists(y_path_upper):
                y_path = y_path_upper
            else:
                raise FileNotFoundError(f"找不到数据文件：{x_path} 或 {y_path}")

        self.X = np.load(x_path)
        self.y = np.load(y_path)

        all_classes = (known_classes or []) + (unknown_classes or [])
        unique_all_classes = []
        for c in all_classes:
            if c not in unique_all_classes:
                unique_all_classes.append(c)
        self.global_label_map = {orig: i for i, orig in enumerate(unique_all_classes)}
        current_max_id = len(unique_all_classes)

        if filter_classes is not None:
            print(f"[{split}] 正在筛选数据集，只保留类别: {filter_classes}")
            mask = np.isin(self.y, filter_classes)
            self.X = self.X[mask]
            self.y = self.y[mask]

        mapped_y = []
        for label in self.y:
            lbl_val = label.item() if hasattr(label, "item") else label
            if lbl_val not in self.global_label_map:
                self.global_label_map[lbl_val] = current_max_id
                current_max_id += 1
            mapped_y.append(self.global_label_map[lbl_val])
        self.y = np.array(mapped_y, dtype=np.int64)
        self.label_map = {orig: self.global_label_map[orig] for orig in (known_classes or [])}

        if self.X.ndim != 3:
            raise ValueError(f"1D数据加载器期望 X 为三维 (N, C, L)，实际 shape={self.X.shape}")
        if self.X.shape[1] != 64 or self.X.shape[2] != 64:
            print(f"[{split}] 警告: 当前 1D 输入 shape={self.X.shape[1:]}, 非预期 (64,64)")

        print(
            f"[{split}] 数据集大小: {len(self.y)}, 输入shape(单样本)={self.X.shape[1:]}, "
            f"已知类标签映射: {self.label_map}"
        )

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        x_sample = self.X[idx]  # [C, L]
        tensor = torch.from_numpy(x_sample).float()
        label = self.y[idx]
        return tensor, label

