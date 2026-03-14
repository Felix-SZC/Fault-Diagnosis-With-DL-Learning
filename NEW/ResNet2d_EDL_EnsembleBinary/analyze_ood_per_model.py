import argparse
import os
import sys
import csv

import numpy as np
import torch
from torch.utils.data import DataLoader
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.utils.helpers import load_config  # noqa: E402
from common.utils.data_loader import NpyPackDataset  # noqa: E402
from common.edl_losses import relu_evidence  # noqa: E402
from models import get_model  # noqa: E402


def plot_ood_results(csv_path, output_dir, K, known_classes=None, max_ood_subplots=6, ensemble_strategy=None):
    """从 CSV 读取 OOD 每模型 logits/不确定性，并画图保存到 output_dir。
    ensemble_strategy: 'One_vs_Rest' | 'Normal_vs_Fault_i' | None（仅用 known_classes 短名）
    """
    known_classes = known_classes or [f"Model {k}" for k in range(K)]

    # 读取 CSV
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row["global_index"] = int(row["global_index"])
            row["model_idx"] = int(row["model_idx"])
            row["logit_neg"] = float(row["logit_neg"])
            row["logit_pos"] = float(row["logit_pos"])
            row["uncertainty_edl_2_over_S"] = float(row["uncertainty_edl_2_over_S"])
            if "prob_neg" in row and str(row.get("prob_neg", "")).strip() != "":
                try:
                    row["prob_neg"] = float(row["prob_neg"])
                except (ValueError, TypeError):
                    row["prob_neg"] = None
            else:
                row["prob_neg"] = None
            if "prob_pos" in row and str(row.get("prob_pos", "")).strip() != "":
                try:
                    row["prob_pos"] = float(row["prob_pos"])
                except (ValueError, TypeError):
                    row["prob_pos"] = None
            else:
                row["prob_pos"] = None
            if "max_p_yes" in row and str(row.get("max_p_yes", "")).strip() != "":
                try:
                    row["max_p_yes"] = float(row["max_p_yes"])
                except (ValueError, TypeError):
                    row["max_p_yes"] = None
            else:
                row["max_p_yes"] = None
            rows.append(row)

    if not rows:
        print("CSV 无数据，跳过画图。")
        return

    # 按 global_index 分组，每个 index 对应一个 OOD 样本
    by_index = {}
    for r in rows:
        idx = r["global_index"]
        if idx not in by_index:
            by_index[idx] = []
        by_index[idx].append(r)

    ood_indices = sorted(by_index.keys())
    n_ood = len(ood_indices)

    try:
        plt.rcParams["font.sans-serif"] = ["SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        pass

    # 根据集成策略生成模型标签（NvF / OvR）
    if ensemble_strategy == "Normal_vs_Fault_i" and len(known_classes) >= K:
        model_labels = ["0:NvsAll"]
        for k in range(1, K):
            short = str(known_classes[k])[:10] if k < len(known_classes) else f"F{k}"
            model_labels.append(f"{k}:Nvs{short}")
    elif ensemble_strategy == "One_vs_Rest" and len(known_classes) >= K:
        model_labels = [str(c)[:12] for c in known_classes]
    else:
        model_labels = [str(known_classes[k])[:12] for k in range(K)] if len(known_classes) >= K else [f"M{k}" for k in range(K)]
    x = np.arange(K)
    width = 0.35

    # 图1：前 max_ood_subplots 个 OOD 样本，每个样本一子图（logits 分组柱状 + 不确定性）
    n_plot = min(max_ood_subplots, n_ood)
    n_cols = 2
    n_rows = (n_plot + n_cols - 1) // n_cols
    fig1, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 4 * n_rows))
    if n_rows == 1 and n_cols == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()

    has_probs = rows and rows[0].get("prob_neg") is not None and rows[0].get("prob_pos") is not None

    for i in range(n_plot):
        ax = axes[i]
        idx = ood_indices[i]
        recs = by_index[idx]
        recs = sorted(recs, key=lambda r: r["model_idx"])
        logit_neg = [r["logit_neg"] for r in recs]
        logit_pos = [r["logit_pos"] for r in recs]
        u_vals = [r["uncertainty_edl_2_over_S"] for r in recs]

        ax2 = ax.twinx()
        bars1 = ax.bar(x - width / 2, logit_neg, width, label="logit_neg (Rest)", color="steelblue", alpha=0.8)
        bars2 = ax.bar(x + width / 2, logit_pos, width, label="logit_pos (Class k)", color="coral", alpha=0.8)
        ax2.plot(x, u_vals, "o-", color="green", linewidth=2, markersize=8, label="u=2/S")
        if has_probs:
            prob_neg = [r["prob_neg"] for r in recs]
            prob_pos = [r["prob_pos"] for r in recs]
            ax2.plot(x, prob_neg, "s--", color="navy", linewidth=1.5, markersize=5, alpha=0.8, label="P(Rest)")
            ax2.plot(x, prob_pos, "^-.", color="darkred", linewidth=1.5, markersize=5, alpha=0.8, label="P(Class k)")
        ax.set_ylabel("Logits")
        ax2.set_ylabel("Uncertainty u / Prob (0-1)")
        ax2.set_ylim(0, 1.05)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_title(f"OOD sample (global_index={idx})")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

    for j in range(n_plot, len(axes)):
        axes[j].set_visible(False)
    fig1.tight_layout()
    fig1.savefig(os.path.join(output_dir, "ood_per_model_logits_uncertainty_samples.png"), dpi=150)
    plt.close(fig1)
    print(f"已保存: {os.path.join(output_dir, 'ood_per_model_logits_uncertainty_samples.png')}")

    # 图2：全体 OOD 汇总 — 每个模型上 logit_neg / logit_pos 的均值（柱状）
    logit_neg_per_model = [[] for _ in range(K)]
    logit_pos_per_model = [[] for _ in range(K)]
    u_per_model = [[] for _ in range(K)]
    prob_neg_per_model = [[] for _ in range(K)]
    prob_pos_per_model = [[] for _ in range(K)]
    for idx in ood_indices:
        recs = by_index[idx]
        for r in recs:
            k = r["model_idx"]
            logit_neg_per_model[k].append(r["logit_neg"])
            logit_pos_per_model[k].append(r["logit_pos"])
            u_per_model[k].append(r["uncertainty_edl_2_over_S"])
            if r.get("prob_neg") is not None:
                prob_neg_per_model[k].append(r["prob_neg"])
            if r.get("prob_pos") is not None:
                prob_pos_per_model[k].append(r["prob_pos"])

    mean_neg = [np.mean(logit_neg_per_model[k]) for k in range(K)]
    mean_pos = [np.mean(logit_pos_per_model[k]) for k in range(K)]
    std_neg = [np.std(logit_neg_per_model[k]) for k in range(K)]
    std_pos = [np.std(logit_pos_per_model[k]) for k in range(K)]

    fig2, ax = plt.subplots(1, 1, figsize=(8, 5))
    ax.bar(x - width / 2, mean_neg, width, yerr=std_neg, label="logit_neg (Rest) mean±std", color="steelblue", alpha=0.8, capsize=3)
    ax.bar(x + width / 2, mean_pos, width, yerr=std_pos, label="logit_pos (Class k) mean±std", color="coral", alpha=0.8, capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(model_labels)
    ax.set_ylabel("Logit value")
    ax.set_title("OOD samples: mean logits per OvR model")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    fig2.tight_layout()
    fig2.savefig(os.path.join(output_dir, "ood_per_model_logits_mean.png"), dpi=150)
    plt.close(fig2)
    print(f"已保存: {os.path.join(output_dir, 'ood_per_model_logits_mean.png')}")

    # 图3：全体 OOD 上每个模型的不确定性分布（箱线图）
    fig3, ax = plt.subplots(1, 1, figsize=(8, 5))
    try:
        bp = ax.boxplot([u_per_model[k] for k in range(K)], tick_labels=model_labels, patch_artist=True)
    except TypeError:
        # 兼容旧版本 Matplotlib：使用 labels 参数
        bp = ax.boxplot([u_per_model[k] for k in range(K)], labels=model_labels, patch_artist=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("lightgreen")
        patch.set_alpha(0.7)
    ax.set_ylabel("Uncertainty u = 2/S")
    ax.set_title("OOD samples: uncertainty per OvR model")
    fig3.tight_layout()
    fig3.savefig(os.path.join(output_dir, "ood_per_model_uncertainty_boxplot.png"), dpi=150)
    plt.close(fig3)
    print(f"已保存: {os.path.join(output_dir, 'ood_per_model_uncertainty_boxplot.png')}")

    # 图4：全体 OOD 汇总 — 每个模型上 prob_neg / prob_pos 的均值（柱状）
    if all(len(prob_neg_per_model[k]) > 0 and len(prob_pos_per_model[k]) > 0 for k in range(K)):
        mean_pneg = [np.mean(prob_neg_per_model[k]) for k in range(K)]
        mean_ppos = [np.mean(prob_pos_per_model[k]) for k in range(K)]
        std_pneg = [np.std(prob_neg_per_model[k]) for k in range(K)]
        std_ppos = [np.std(prob_pos_per_model[k]) for k in range(K)]
        fig4, ax = plt.subplots(1, 1, figsize=(8, 5))
        ax.bar(x - width / 2, mean_pneg, width, yerr=std_pneg, label="P(Rest) mean±std", color="steelblue", alpha=0.8, capsize=3)
        ax.bar(x + width / 2, mean_ppos, width, yerr=std_ppos, label="P(Class k) mean±std", color="coral", alpha=0.8, capsize=3)
        ax.set_xticks(x)
        ax.set_xticklabels(model_labels)
        ax.set_ylabel("Probability")
        ax.set_ylim(0, 1.05)
        ax.set_title("OOD samples: mean positive/negative probability per OvR model")
        ax.legend()
        fig4.tight_layout()
        fig4.savefig(os.path.join(output_dir, "ood_per_model_probs_mean.png"), dpi=150)
        plt.close(fig4)
        print(f"已保存: {os.path.join(output_dir, 'ood_per_model_probs_mean.png')}")

    # 图5：每个 OOD 样本的 max_p_yes 分布（直方图）+ max_p_yes vs 均值u 散点
    max_p_yes_per_sample = []
    mean_u_per_sample = []
    for idx in ood_indices:
        recs = by_index[idx]
        mp = recs[0].get("max_p_yes")
        if mp is None and recs and recs[0].get("prob_pos") is not None:
            mp = max(r["prob_pos"] for r in recs)
        if mp is not None:
            max_p_yes_per_sample.append(mp)
            mean_u_per_sample.append(np.mean([r["uncertainty_edl_2_over_S"] for r in recs]))
    if max_p_yes_per_sample:
        fig5, (ax5a, ax5b) = plt.subplots(1, 2, figsize=(10, 5))
        ax5a.hist(max_p_yes_per_sample, bins=25, color="steelblue", alpha=0.8, edgecolor="white")
        ax5a.set_xlabel("max P(Class k) over models")
        ax5a.set_ylabel("Count")
        ax5a.set_title("OOD samples: distribution of max_p_yes")
        ax5a.axvline(0.5, color="red", linestyle="--", label="0.5")
        ax5a.legend()
        ax5b.scatter(max_p_yes_per_sample, mean_u_per_sample, alpha=0.6, s=15)
        ax5b.set_xlabel("max P(Class k)")
        ax5b.set_ylabel("Mean u (2/S) over models")
        ax5b.set_title("OOD: max_p_yes vs mean uncertainty")
        fig5.tight_layout()
        fig5.savefig(os.path.join(output_dir, "ood_per_model_max_p_yes_dist.png"), dpi=150)
        plt.close(fig5)
        print(f"已保存: {os.path.join(output_dir, 'ood_per_model_max_p_yes_dist.png')}")


def get_dataset(data_config, split="test", filter_classes=None):
    data_dir = data_config.get("data_dir")
    if data_dir is None:
        raise ValueError("配置文件中必须指定 data.data_dir")

    openset_config = data_config.get("openset", {})
    known_classes = openset_config.get("known_classes")
    unknown_classes = openset_config.get("unknown_classes", [])

    return NpyPackDataset(
        data_dir=data_dir,
        split=split,
        filter_classes=filter_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "分析测试集中每个 OOD 样本在 OvR 每个模型中的不确定性和 logits 输出；"
            "结果保存为 CSV，便于后续检查。"
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config_OvR.yaml",
        help="配置文件路径（默认: config_OvR.yaml；NvF 请用 --config config_NvF.yaml）",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="checkpoint 目录（含 model_0.pth..model_{K-1}.pth）；"
        "未指定则从 config 的 checkpoint_dir 读取",
    )
    parser.add_argument(
        "--output_csv",
        type=str,
        default=None,
        help="输出 CSV 路径；未指定则为 checkpoint_dir/test/ood_per_model_logits_and_uncertainty.csv",
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="生成并保存 OOD 每模型 logits/不确定性 图到 test 目录",
    )
    parser.add_argument(
        "--max_ood_subplots",
        type=int,
        default=6,
        help="画图时最多展示多少个 OOD 样本的子图（默认 6）",
    )
    parser.add_argument(
        "--plot_only",
        action="store_true",
        help="仅从已有 CSV 画图，不加载模型不写 CSV；需指定 --output_csv 且 CSV 存在",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        choices=["OvR", "NvF"],
        help="仅 --plot_only 时有效：标签风格 OvR(One-vs-Rest) 或 NvF(Normal vs Fault)；非 plot_only 时从 config 读取",
    )
    args = parser.parse_args()

    # 仅画图模式：只读 CSV 并画图
    if args.plot_only:
        csv_path = args.output_csv
        if not csv_path or not os.path.isfile(csv_path):
            print("错误: --plot_only 需指定已存在的 --output_csv 路径。")
            sys.exit(1)
        output_dir = os.path.dirname(csv_path)
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        if not rows:
            print("CSV 无数据。")
            sys.exit(1)
        by_idx = {}
        for r in rows:
            idx = int(r["global_index"])
            by_idx.setdefault(idx, []).append(int(r["model_idx"]))
        K = max(len(by_idx[k]) for k in by_idx)
        known_classes = [f"Model {k}" for k in range(K)]
        strat = "One_vs_Rest" if args.strategy == "OvR" else ("Normal_vs_Fault_i" if args.strategy == "NvF" else None)
        plot_ood_results(csv_path, output_dir, K, known_classes=known_classes, max_ood_subplots=args.max_ood_subplots, ensemble_strategy=strat)
        return

    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    if args.checkpoint:
        checkpoint_dir = os.path.abspath(args.checkpoint)
    else:
        checkpoint_dir = train_config["checkpoint_dir"]
    if not os.path.isdir(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir)

    output_dir = os.path.join(checkpoint_dir, "test")
    os.makedirs(output_dir, exist_ok=True)
    output_csv = args.output_csv or os.path.join(
        output_dir, "ood_per_model_logits_and_uncertainty.csv"
    )

    known_classes = data_config["openset"]["known_classes"]
    unknown_classes = data_config["openset"].get("unknown_classes", [])
    num_known_classes = len(known_classes)
    K = num_known_classes
    ensemble_strategy = train_config.get("ensemble_strategy", "Normal_vs_Fault_i")

    print(f"已知类 K={K}, 已知类名称: {known_classes}")
    print(f"未知类标签列表（仅用于信息输出）: {unknown_classes}")
    print(f"集成策略: {ensemble_strategy}")

    def _infer_backbone_from_state(state):
        """根据 checkpoint 的 conv1 通道数推断 backbone：32 -> Light，64 -> 完整 ResNet18。"""
        if not state or "conv1.weight" not in state:
            return None
        ch = state["conv1.weight"].shape[0]
        if ch == 32:
            return "ResNet18_2d_Light"
        if ch == 64:
            return "ResNet18_2d"
        return None

    backbone_type = model_config.get("type", "ResNet18_2d_Light")
    models = []
    for k in range(K):
        path_k = os.path.join(checkpoint_dir, f"model_{k}.pth")
        if not os.path.exists(path_k):
            raise FileNotFoundError(f"未找到 {path_k}，请先运行 train.py")
        ckpt = torch.load(path_k, map_location=device, weights_only=True)
        state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        model_k = get_model(backbone_type, num_classes=2).to(device)
        try:
            model_k.load_state_dict(state, strict=True)
        except RuntimeError as e:
            inferred = _infer_backbone_from_state(state)
            if inferred and inferred != backbone_type:
                backbone_type = inferred
                model_k = get_model(backbone_type, num_classes=2).to(device)
                model_k.load_state_dict(state, strict=True)
                print(f"根据 checkpoint 推断 backbone={backbone_type}，已用其重新加载 model_{k}.pth")
            else:
                raise e
        model_k.eval()
        models.append(model_k)
    print(f"已加载 K={K} 个模型，backbone={backbone_type}")

    # 读取测试集（包括已知 + 未知）
    test_dataset = get_dataset(data_config, split="test", filter_classes=None)
    test_loader = DataLoader(
        test_dataset,
        batch_size=train_config.get("batch_size", 32),
        shuffle=False,
    )
    print(f"测试集大小: {len(test_dataset)}")

    # OOD 样本定义：标签 >= K
    labels_np = test_dataset.y
    ood_indices = [i for i, y in enumerate(labels_np) if int(y) >= K]
    if not ood_indices:
        print("测试集中没有 OOD 样本（label >= K），不生成 CSV。")
        return
    print(f"OOD 样本数量: {len(ood_indices)}")

    # 为了在 DataLoader 中方便筛选 OOD，记录全局索引
    # NpyPackDataset 默认会返回 (x, y)，我们这里手动跟踪样本全局 idx。
    # 我们按顺序遍历 test_loader，维护一个全局计数器 global_idx。
    fieldnames = [
        "global_index",
        "true_label",
        "is_ood",
        "model_idx",
        "logit_neg",
        "logit_pos",
        "prob_neg",
        "prob_pos",
        "max_p_yes",
        "uncertainty_edl_2_over_S",
    ]

    with open(output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        global_idx = 0
        with torch.no_grad():
            for inputs, labels in test_loader:
                batch_size = labels.size(0)
                inputs = inputs.to(device)

                labels_np_batch = labels.cpu().numpy()
                for b in range(batch_size):
                    y_true = int(labels_np_batch[b])
                    is_ood = int(y_true >= K)
                    if not is_ood:
                        global_idx += 1
                        continue

                    x_b = inputs[b : b + 1]
                    row_buf = []
                    for k in range(K):
                        model_k = models[k]
                        logits_k = model_k(x_b)
                        if isinstance(logits_k, (tuple, list)):
                            logits_k = logits_k[0]
                        logits_k = logits_k.squeeze(0)

                        evidence_k = relu_evidence(logits_k.unsqueeze(0))
                        alpha_k = evidence_k + 1.0
                        S_k = torch.sum(alpha_k, dim=1)
                        u_k = (2.0 / S_k).item()
                        prob_neg = (alpha_k[0, 0] / S_k[0]).item()
                        prob_pos = (alpha_k[0, 1] / S_k[0]).item()
                        row_buf.append({
                            "global_index": global_idx,
                            "true_label": y_true,
                            "is_ood": is_ood,
                            "model_idx": k,
                            "logit_neg": float(logits_k[0].item()),
                            "logit_pos": float(logits_k[1].item()),
                            "prob_neg": float(prob_neg),
                            "prob_pos": float(prob_pos),
                            "uncertainty_edl_2_over_S": float(u_k),
                        })
                    max_p_yes = max(r["prob_pos"] for r in row_buf)
                    for r in row_buf:
                        r["max_p_yes"] = float(max_p_yes)
                        writer.writerow(r)
                    global_idx += 1

                # 对于 batch 中非 OOD 的样本，global_idx 也已经在循环里自增

    print(f"分析完成，结果已保存至: {output_csv}")

    if args.plot:
        plot_ood_results(
            output_csv,
            output_dir,
            K,
            known_classes=known_classes,
            max_ood_subplots=args.max_ood_subplots,
            ensemble_strategy=ensemble_strategy,
        )


if __name__ == "__main__":
    main()

