"""
评估阶段一闭集 K 类 LaoDA 预训练权重（pretrained_full.pth）：准确率 + 混淆矩阵。

用法（在 NEW/ResNet2d_EDL_EnsembleBinary 根目录）:
  python tools/test_lao_da_closed_pretrain.py --config configs/bench_LaoDA_closed_pretrain.yaml
  python tools/test_lao_da_closed_pretrain.py --config configs/bench_LaoDA_closed_pretrain.yaml \\
      --ckpt checkpoints/LaoDA/closed_pretrain/seed2/pretrained_full.pth \\
      --output_dir checkpoints/LaoDA/closed_pretrain/seed2/test_eval
"""
from __future__ import annotations

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _root)

from common.edl_losses import relu_evidence  # noqa: E402
from common.utils.data_loader_1d import NpyPackDataset1D  # noqa: E402
from common.utils.helpers import load_config  # noqa: E402
from models import get_model  # noqa: E402


def get_dataset(data_config, split: str = "test"):
    data_dir = data_config.get("data_dir")
    if data_dir is None:
        raise ValueError("配置文件中必须指定 data.data_dir")
    openset_config = data_config.get("openset", {})
    known_classes = openset_config.get("known_classes")
    unknown_classes = openset_config.get("unknown_classes", [])
    if known_classes is None:
        raise ValueError("配置文件中必须指定 data.openset.known_classes")
    return NpyPackDataset1D(
        data_dir=data_dir,
        split=split,
        filter_classes=known_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
    )


def _load_checkpoint(path: str, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    parser = argparse.ArgumentParser(description="LaoDA 闭集预训练 checkpoint 评估（混淆矩阵）")
    parser.add_argument("--config", type=str, default="configs/bench_LaoDA_closed_pretrain.yaml")
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="pretrained_full.pth；默认使用 train.checkpoint_dir/pretrained_full.pth",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="保存图与报告；默认 checkpoint_dir/test_closed_eval",
    )
    parser.add_argument("--split", type=str, default="test", choices=["train", "test"])
    parser.add_argument("--batch_size", type=int, default=None, help="默认读取 train.batch_size")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    train_config = config["train"]
    model_config = config["model"]

    if model_config.get("type", "LaoDA") != "LaoDA":
        raise ValueError("本脚本仅支持 model.type=LaoDA")

    known_classes = data_config["openset"]["known_classes"]
    K = len(known_classes)
    checkpoint_dir = train_config["checkpoint_dir"]
    ckpt_path = args.ckpt or os.path.join(checkpoint_dir, "pretrained_full.pth")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到权重: {ckpt_path}")

    output_dir = args.output_dir or os.path.join(checkpoint_dir, "test_closed_eval")
    os.makedirs(output_dir, exist_ok=True)

    batch_size = args.batch_size or int(train_config.get("batch_size", 32))
    device_str = train_config.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_str)

    dataset = get_dataset(data_config, split=args.split)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    ckpt = _load_checkpoint(ckpt_path, device)
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        state_dict = ckpt["state_dict"]
        ckpt_k = ckpt.get("num_classes", None)
        if ckpt_k is not None and int(ckpt_k) != K:
            print(f"警告: ckpt num_classes={ckpt_k} 与当前配置 K={K} 不一致，仍尝试 strict 加载")
    else:
        state_dict = ckpt

    model = get_model("LaoDA", num_classes=K).to(device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    all_labels = []
    all_preds = []
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            logits = model(inputs)
            logits = logits[0] if isinstance(logits, tuple) else logits
            evidence = relu_evidence(logits)
            alpha = evidence + 1
            S = torch.sum(alpha, dim=1, keepdim=True)
            probs = alpha / S
            preds = probs.argmax(dim=1)
            all_labels.append(labels.cpu().numpy())
            all_preds.append(preds.cpu().numpy())

    y_true = np.concatenate(all_labels, axis=0)
    y_pred = np.concatenate(all_preds, axis=0)
    acc = 100.0 * accuracy_score(y_true, y_pred)

    print(f"split={args.split} 样本数={len(y_true)}  accuracy={acc:.2f}%")
    print(classification_report(y_true, y_pred, labels=np.arange(K), target_names=[str(c) for c in known_classes], zero_division=0))

    cm = confusion_matrix(y_true, y_pred, labels=np.arange(K))
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sum > 0, cm.astype(float) / row_sum * 100, 0)
    annot = np.array([[f"{cm_pct[i, j]:.1f}%" for j in range(cm_pct.shape[1])] for i in range(cm_pct.shape[0])])

    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(max(8, K * 1.2), max(6, K * 1.0)))
    sns.heatmap(
        cm_pct,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=known_classes,
        yticklabels=known_classes,
    )
    plt.title(f"Closed-set K-class Pretrain Confusion Matrix")
    plt.ylabel("True")
    plt.xlabel("Pred")
    plt.tight_layout()
    fig_path = os.path.join(output_dir, f"confusion_matrix_closed_pretrain_{args.split}.png")
    plt.savefig(fig_path, dpi=150)
    plt.close()
    print(f"已保存: {fig_path}")

    report_path = os.path.join(output_dir, f"eval_closed_pretrain_{args.split}.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"ckpt={ckpt_path}\nconfig={args.config}\nsplit={args.split}\n")
        f.write(f"accuracy={acc:.4f}%\n\n")
        f.write(
            classification_report(
                y_true, y_pred, labels=np.arange(K), target_names=[str(c) for c in known_classes], zero_division=0
            )
        )
        f.write("\n\nconfusion_matrix (counts):\n")
        f.write(np.array2string(cm))
    print(f"已保存: {report_path}")


if __name__ == "__main__":
    main()
