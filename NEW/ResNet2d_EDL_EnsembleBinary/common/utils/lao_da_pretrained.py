"""
LaoDA 两阶段训练工具：从多类预训练 checkpoint 加载 backbone（不含 fc2），并在二分类 NvF 模型上设置冻结策略。
不修改 models/LaoDA.py；供 train_nvf_frozen_backbone.py 使用。
"""
import os
from typing import Dict, List, Tuple

import torch
import torch.nn as nn


def _extract_state_dict(ckpt) -> Dict[str, torch.Tensor]:
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            return ckpt["state_dict"]
        if "model_state_dict" in ckpt:
            return ckpt["model_state_dict"]
    if isinstance(ckpt, dict) and all(isinstance(v, torch.Tensor) for v in ckpt.values()):
        return ckpt
    raise ValueError("checkpoint 中未找到 state_dict / model_state_dict")


def load_lao_da_backbone_from_multiclass_ckpt(
    model_binary: nn.Module,
    ckpt_path: str,
    device: torch.device,
    strict_backbone: bool = False,
) -> Tuple[List[str], List[str]]:
    """
    将多类 LaoDA 预训练权重加载到 num_classes=2 的 LaoDA 上：复制除 fc2 外所有同名同形状参数；fc2 保持随机初始化。

    Returns:
        (loaded_keys, skipped_keys)
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"预训练权重不存在: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    src = _extract_state_dict(ckpt)
    dst = model_binary.state_dict()

    to_load = {}
    loaded_keys = []
    skipped_keys = []

    for k, v in src.items():
        if k.startswith("fc2."):
            skipped_keys.append(k)
            continue
        if k not in dst:
            skipped_keys.append(k)
            continue
        if v.shape != dst[k].shape:
            skipped_keys.append(f"{k} (shape {tuple(v.shape)} != {tuple(dst[k].shape)})")
            continue
        to_load[k] = v
        loaded_keys.append(k)

    missing_in_src = [k for k in dst.keys() if k not in to_load and not k.startswith("fc2.")]
    model_binary.load_state_dict(to_load, strict=False)

    if strict_backbone and missing_in_src:
        raise RuntimeError(f"预训练 ckpt 缺少以下 backbone 键: {missing_in_src}")

    return loaded_keys, skipped_keys


def set_lao_da_trainable_heads(model: nn.Module, train_fc1: bool = False) -> List[str]:
    """
    冻结 conv/bn/mamba；按 train_fc1 决定是否训练 fc1；fc2 始终可训练（NvF 二分类头）。
    """
    trainable = []
    for name, p in model.named_parameters():
        p.requires_grad = False

    for name, p in model.named_parameters():
        if name.startswith("fc2."):
            p.requires_grad = True
            trainable.append(name)
        elif train_fc1 and name.startswith("fc1."):
            p.requires_grad = True
            trainable.append(name)

    return sorted(trainable)


def summarize_trainable_params(model: nn.Module) -> List[str]:
    return sorted([n for n, p in model.named_parameters() if p.requires_grad])
