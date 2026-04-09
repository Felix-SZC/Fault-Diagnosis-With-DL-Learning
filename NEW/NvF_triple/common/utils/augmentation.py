import numpy as np
import torch
import random

class SignalAugmenter:
    def __init__(self, config=None):
        """
        初始化数据增强器
        :param config: dict, 包含增强配置开关和参数
        """
        if config is None:
            config = {}
        
        self.enable = config.get("enable", False)
        self.max_shift_ratio = config.get("max_shift_ratio", 0.0)
        self.min_max_scale = config.get("min_max_scale", False)
        self.noise_std_ratio = config.get("noise_std_ratio", 0.0)
        self.max_mask_ratio = config.get("max_mask_ratio", 0.0)
        self.structure_noise_scale = config.get("structure_noise_scale", 0.0)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        对单个样本应用数据增强
        :param x: 形状为 (1, 64, 64) 或 (64, 64) 的 Tensor
        :return: 增强后的 Tensor
        """
        if not self.enable:
            return x

        # 转换为 numpy 处理比较方便，然后再转回 torch
        x_np = x.numpy()
        original_shape = x_np.shape
        
        # 将数据展平为 1D 序列进行时序相关的增强，再 reshape 回去
        seq_len = np.prod(original_shape)
        x_flat = x_np.reshape(-1)

        # 1. 降低幅值能量偏置 (Min-max_scale / Random Amplitude Scaling)
        if self.min_max_scale and random.random() < 0.5:
            # 随机幅度缩放，例如 0.8 到 1.2 之间
            scale = random.uniform(0.8, 1.2)
            x_flat = x_flat * scale

        # 2. 降低采样相位偏置 (max_shift_ratio)
        if self.max_shift_ratio > 0 and random.random() < 0.5:
            max_shift = int(seq_len * self.max_shift_ratio)
            if max_shift > 0:
                shift = random.randint(-max_shift, max_shift)
                x_flat = np.roll(x_flat, shift)

        # 3. 提升抗噪鲁棒性 (noise_std_ratio)
        if self.noise_std_ratio > 0 and random.random() < 0.5:
            std = np.std(x_flat)
            if std > 0:
                noise = np.random.normal(0, std * self.noise_std_ratio, size=x_flat.shape)
                x_flat = x_flat + noise

        # 4. 抑制对偶然局部噪点或局部伪特征的依赖 (max_mask_ratio)
        if self.max_mask_ratio > 0 and random.random() < 0.5:
            max_mask_len = int(seq_len * self.max_mask_ratio)
            if max_mask_len > 0:
                mask_len = random.randint(1, max_mask_len)
                mask_start = random.randint(0, seq_len - mask_len)
                # 将掩码区域置为该样本的均值或 0 (这里使用 0)
                x_flat[mask_start : mask_start + mask_len] = 0

        # 5. 降低结构偏移偏置 (structure_noise_scale)
        if self.structure_noise_scale > 0 and random.random() < 0.5:
            # 构造低频基线漂移（例如半个正弦波）
            t = np.linspace(0, np.pi, seq_len)
            std = np.std(x_flat)
            if std > 0:
                baseline = np.sin(t) * self.structure_noise_scale * std
                # 随机翻转漂移方向
                if random.random() < 0.5:
                    baseline = -baseline
                x_flat = x_flat + baseline

        x_aug = x_flat.reshape(original_shape)
        return torch.from_numpy(x_aug).float()
