import argparse
import os

import matplotlib.pyplot as plt
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="分析独立 K 个二分类 EDL 集成的样本级详细信息，并可视化每个子模型的 logits/p_yes/u_k 与 u_mean。"
    )
    parser.add_argument(
        "--detail_path",
        type=str,
        required=True,
        help="test_edl_ensemble_binary.py 保存的 detail_raw.npz 路径",
    )
    parser.add_argument(
        "--target_class",
        type=int,
        default=None,
        help="关注的预测类别（如 8）。若为空则不按预测类别筛选。",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=100,
        help="最多可视化的样本数量",
    )
    parser.add_argument(
        "--only_ood_misclassified",
        action="store_true",
        help="仅选择 OOD 且被误判为 target_class 的样本（is_ood==1 且 pred_label==target_class）",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    detail_path = os.path.abspath(args.detail_path)
    if not os.path.isfile(detail_path):
        raise FileNotFoundError(f"detail_raw 文件不存在: {detail_path}")

    data = np.load(detail_path, allow_pickle=True)
    sample_idx = data["sample_idx"]
    true_label = data["true_label"]
    pred_label = data["pred_label"]
    is_ood = data["is_ood"]
    logits_yes = data["logits_yes"]
    logits_no = data["logits_no"]
    p_yes = data["p_yes"]
    u_k = data["u"]
    u_mean = data["u_mean"]

    num_samples, num_models = p_yes.shape

    # 构造筛选条件
    mask = np.ones(num_samples, dtype=bool)
    if args.target_class is not None:
        mask &= pred_label == args.target_class
    if args.only_ood_misclassified:
        mask &= is_ood.astype(bool)

    candidate_indices = np.where(mask)[0]
    if candidate_indices.size == 0:
        print("根据当前筛选条件未找到任何样本，请放宽条件或检查 target_class。")
        return

    selected_indices = candidate_indices[: args.max_samples]
    print(
        f"从 {candidate_indices.size} 个候选样本中选择 {selected_indices.size} 个进行可视化。"
    )

    out_dir = os.path.join(os.path.dirname(detail_path), "samples")
    os.makedirs(out_dir, exist_ok=True)

    for idx in selected_indices:
        sid = int(sample_idx[idx])
        t = int(true_label[idx])
        p = int(pred_label[idx])
        ood_flag = bool(is_ood[idx])

        logits_yes_k = logits_yes[idx] if logits_yes is not None else None
        logits_no_k = logits_no[idx] if logits_no is not None else None
        p_yes_k = p_yes[idx]
        u_k_vec = u_k[idx]
        u_mean_val = float(u_mean[idx])

        winner = int(np.argmax(p_yes_k))
        models = np.arange(num_models)

        fig, axes = plt.subplots(1, 3, figsize=(14, 4))

        # 子图 1：P_yes
        ax0 = axes[0]
        ax0.bar(models, p_yes_k, color="C0")
        ax0.bar(
            winner,
            p_yes_k[winner],
            color="C3",
            label=f"winner (model {winner})",
        )
        ax0.set_xlabel("Model k")
        ax0.set_ylabel("P_k(yes)")
        ax0.set_title("Per-model P_yes")
        ax0.set_ylim(0.0, 1.0)
        ax0.legend()

        # 子图 2：u_k 与 u_mean
        ax1 = axes[1]
        ax1.bar(models, u_k_vec, color="C1")
        ax1.axhline(u_mean_val, color="k", linestyle="--", label=f"u_mean={u_mean_val:.3f}")
        ax1.set_xlabel("Model k")
        ax1.set_ylabel("u_k (2/S_k)")
        ax1.set_title("Per-model uncertainty u_k")
        ax1.legend()

        # 子图 3：logits_yes（可选）
        ax2 = axes[2]
        if logits_yes_k is not None:
            ax2.bar(models, logits_yes_k, color="C2")
            ax2.set_ylabel("logits_yes")
        else:
            ax2.text(0.5, 0.5, "logits 未保存", ha="center", va="center")
        ax2.set_xlabel("Model k")
        ax2.set_title("Per-model logits_yes")

        fig.suptitle(
            f"sample_idx={sid}, true={t}, pred={p}, is_ood={ood_flag}, u_mean={u_mean_val:.3f}"
        )
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        out_path = os.path.join(out_dir, f"sample_{sid}.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"已保存样本可视化: {out_path}")


if __name__ == "__main__":
    main()

