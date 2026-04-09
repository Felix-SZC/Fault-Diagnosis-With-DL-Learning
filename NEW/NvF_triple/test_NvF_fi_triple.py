# Fi-only 三态测试（无 model0）；不修改 test_NvF.py
from __future__ import annotations

import argparse
import os
import shutil
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from torch.utils.data import DataLoader

sns.set_theme(style="whitegrid", font="SimHei", font_scale=1.0)
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 150

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.edl_losses import relu_evidence
from common.utils.helpers import load_config
from common.utils.data_loader import NpyPackDataset
from common.utils.data_loader_1d import NpyPackDataset1D
from models import get_model

from nvf_fi_triple_common import (
    discover_epochs_fi,
    fi_triple_preds_and_ood,
)


def parse_test_epochs_spec(spec: str) -> list[int]:
    """
    解析 --test_epochs / test_infer.test_epochs 字符串。
    支持：单 epoch「80」；闭区间「10-50」（含端点）；逗号列表「1,5,80」；混合「1,10-12,90」。
    """
    s = (spec or "").strip()
    if not s:
        raise ValueError("test_epochs 为空")
    out: list[int] = []
    for part in s.replace(" ", "").split(","):
        if not part:
            continue
        if "-" in part:
            a, _, b = part.partition("-")
            if not a or not b or "-" in b:
                raise ValueError(f"无法解析区间: {part!r}（请使用如 10-50）")
            lo, hi = int(a), int(b)
            if hi < lo:
                lo, hi = hi, lo
            out.extend(range(lo, hi + 1))
        else:
            out.append(int(part))
    if not out:
        raise ValueError("test_epochs 未解析出任何 epoch")
    return sorted(set(out))


def filter_discovered_epochs(discovered: list[int], wanted: list[int] | None) -> tuple[list[int], list[int]]:
    """若 wanted 为 None 则返回全部 discovered；否则求交集，并返回 (选用列表, 请求但未找到的 epoch)。"""
    if not discovered:
        return [], wanted or []
    if wanted is None:
        return list(discovered), []
    dset = set(discovered)
    use = sorted(dset & set(wanted))
    missing = sorted(set(wanted) - dset)
    return use, missing


def get_dataset(data_config, model_type, split="test", filter_classes=None):
    data_dir = data_config.get("data_dir")
    openset_config = data_config.get("openset", {})
    known_classes = openset_config.get("known_classes")
    unknown_classes = openset_config.get("unknown_classes", [])
    dataset_cls = NpyPackDataset1D if model_type == "LaoDA" else NpyPackDataset
    return dataset_cls(
        data_dir=data_dir,
        split=split,
        filter_classes=filter_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes,
    )


def load_models_fi(checkpoint_dir, K, backbone_type, device, epoch=None):
    models = []
    for k in range(1, K):
        if epoch is None:
            path_k = os.path.join(checkpoint_dir, f"model_{k}.pth")
        else:
            path_k = os.path.join(checkpoint_dir, "epochs", str(epoch), f"model_{k}.pth")
        if not os.path.exists(path_k):
            raise FileNotFoundError(path_k)
        m = get_model(backbone_type, num_classes=2).to(device)
        ckpt = torch.load(path_k, map_location=device, weights_only=True)
        state = ckpt.get("state_dict") or ckpt.get("model_state_dict") or ckpt
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)
    return models


def compute_uncertainty_threshold_iqr_fi(val_loader, models, device, K, tau_normal, tau_fault):
    unc_list = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            p0s, p1s, us = [], [], []
            for m in models:
                logits = m(inputs)
                logits = logits[0] if isinstance(logits, tuple) else logits
                ev = relu_evidence(logits)
                alpha = ev + 1
                S = torch.sum(alpha, dim=1)
                probs = alpha / S.unsqueeze(1)
                p0s.append(probs[:, 0])
                p1s.append(probs[:, 1])
                us.append(2.0 / S)
            p0_stack = torch.stack(p0s, dim=1)
            p1_stack = torch.stack(p1s, dim=1)
            u_stack = torch.stack(us, dim=1)
            _p, score = fi_triple_preds_and_ood(p0_stack, p1_stack, u_stack, tau_normal, tau_fault)
            unc_list.append(score.cpu().numpy())
    all_u = np.concatenate(unc_list)
    q1, q3 = np.percentile(all_u, [25, 75])
    return float(q3 + 1.5 * (q3 - q1))


def run_test_loop_fi(
    models,
    test_loader,
    device,
    K,
    uncertainty_threshold,
    tau_normal,
    tau_fault,
):
    M = K - 1
    all_labels = []
    all_preds = []
    # 仅保留 logic 分数（与 uncertainty_distribution_mean_nvf.png 对应）
    all_uncertainties = []
    binary_rows = []
    closed_true = []
    closed_pred = []
    p_yes_rows = []
    u_rows = []
    lp_rows = []
    ln_rows = []
    w_pred = []
    w_pyes = []
    w_lpos = []
    w_lneg = []
    sample_idx = []
    cursor = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            p0s, p1s, us, lpos, lneg = [], [], [], [], []
            for m in models:
                logits = m(inputs)
                logits = logits[0] if isinstance(logits, tuple) else logits
                ev = relu_evidence(logits)
                alpha = ev + 1
                S = torch.sum(alpha, dim=1)
                probs = alpha / S.unsqueeze(1)
                p0s.append(probs[:, 0])
                p1s.append(probs[:, 1])
                us.append(2.0 / S)
                lpos.append(logits[:, 1])
                lneg.append(logits[:, 0])
            p0_stack = torch.stack(p0s, dim=1)
            p1_stack = torch.stack(p1s, dim=1)
            u_stack = torch.stack(us, dim=1)
            lp_stack = torch.stack(lpos, dim=1)
            ln_stack = torch.stack(lneg, dim=1)

            preds_closed, ood_logic = fi_triple_preds_and_ood(
                p0_stack, p1_stack, u_stack, tau_normal, tau_fault
            )

            preds_closed = preds_closed.cpu().numpy()
            B = inputs.size(0)

            binary_rows.append((p1_stack > 0.5).cpu().numpy().astype(np.int32))
            p_yes_rows.append(p1_stack.cpu().numpy())
            u_rows.append(u_stack.cpu().numpy())
            lp_rows.append(lp_stack.cpu().numpy())
            ln_rows.append(ln_stack.cpu().numpy())

            all_uncertainties.extend(ood_logic.cpu().numpy().tolist())

            for i in range(B):
                lbl = int(labels[i].item())
                if lbl < K:
                    closed_true.append(lbl)
                    closed_pred.append(int(preds_closed[i]))

                u_val = float(ood_logic[i].item())
                if u_val > uncertainty_threshold:
                    all_preds.append(-1)
                else:
                    all_preds.append(int(preds_closed[i]))
                all_labels.append(lbl)

                pc = int(preds_closed[i])
                w_pred.append(pc)
                if pc > 0:
                    j = pc - 1
                    w_pyes.append(float(p1_stack[i, j].item()))
                    w_lpos.append(float(lp_stack[i, j].item()))
                    w_lneg.append(float(ln_stack[i, j].item()))
                else:
                    w_pyes.append(float(p1_stack[i].mean().item()))
                    w_lpos.append(float(lp_stack[i].mean().item()))
                    w_lneg.append(float(ln_stack[i].mean().item()))

            sample_idx.append(np.arange(cursor, cursor + B, dtype=np.int64))
            cursor += B

    # 对齐 evaluate 接口：仅保留 logic 一项
    final_unc_dict = {"logic": np.array(all_uncertainties)}

    return (
        np.array(all_labels),
        np.array(all_preds),
        final_unc_dict,
        np.concatenate(binary_rows, axis=0),
        np.array(closed_true),
        np.array(closed_pred),
        np.concatenate(p_yes_rows, axis=0),
        np.concatenate(u_rows, axis=0),
        np.concatenate(lp_rows, axis=0),
        np.concatenate(ln_rows, axis=0),
        np.array(w_pred, dtype=np.int64),
        np.array(w_pyes),
        np.array(w_lpos),
        np.array(w_lneg),
        np.concatenate(sample_idx),
    )


def compute_class_model_stats_fi(values_matrix, all_labels, K, has_unknown):
    M = K - 1
    class_indices = list(range(K))
    if has_unknown:
        class_indices.append(K)
    stat = np.zeros((len(class_indices), M), dtype=np.float32)
    row_labels = [f"Class {i}" for i in range(K)]
    if has_unknown:
        row_labels.append("OOD")
    for row_idx, c in enumerate(class_indices):
        mask = (all_labels == c) if c < K else (all_labels >= K)
        if mask.sum() == 0:
            continue
        stat[row_idx] = values_matrix[mask].mean(axis=0)
    return stat, row_labels


def plot_class_model_heatmap_fi(
    stat_matrix,
    row_labels,
    M,
    title,
    cbar_label,
    filename,
    output_dir,
    fmt=".3f",
    cmap="mako",
    annotate=True,
    value_scale=1.0,
    vmin=None,
    vmax=None,
):
    """与 test_NvF.plot_class_model_heatmap 相同风格；列对应 Model 1..M（无 model0）。"""
    to_plot = stat_matrix * value_scale
    figsize_w = max(9, 1.15 * M + 4)
    figsize_h = max(4.5, 0.8 * len(row_labels) + 2.4)
    plt.figure(figsize=(figsize_w, figsize_h))
    ax = sns.heatmap(
        to_plot,
        annot=annotate,
        fmt=fmt,
        cmap=cmap,
        xticklabels=[f"Model {i}" for i in range(1, M + 1)],
        yticklabels=row_labels,
        cbar_kws={"label": cbar_label},
        linewidths=0.4,
        linecolor="white",
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel("Sub-model")
    ax.set_ylabel("True Class")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=180)
    plt.close()


def evaluate_fi_outputs(
    all_labels,
    all_preds,
    uncertainties_dict,
    binary_preds_matrix,
    closed_set_true,
    closed_set_pred,
    p_yes_matrix,
    u_k_matrix,
    logits_pos_matrix,
    logits_neg_matrix,
    known_classes,
    K,
    has_unknown_samples,
    uncertainty_threshold,
    output_dir,
    save_sample_level_csv=True,
):
    os.makedirs(output_dir, exist_ok=True)
    known_mask = all_labels < K
    unknown_mask = ~known_mask

    mapped_known_labels = all_labels[known_mask]
    known_preds_mapped = all_preds[known_mask]
    if len(mapped_known_labels) == 0:
        raise RuntimeError("没有已知类样本")
    accuracy = accuracy_score(mapped_known_labels, known_preds_mapped)

    known_idx = np.where(known_mask)[0]
    binary_accuracies = []
    binary_conf_info = []
    for k in range(1, K):
        sub = (all_labels[known_idx] == 0) | (all_labels[known_idx] == k)
        idx_k = known_idx[sub]
        if len(idx_k) == 0:
            continue
        true_k = (all_labels[idx_k] == k).astype(np.int32)
        pred_k = binary_preds_matrix[idx_k, k - 1]
        binary_accuracies.append(accuracy_score(true_k, pred_k))
        binary_conf_info.append(
            {
                "model_idx": k,
                "true": true_k,
                "pred": pred_k,
                "title": f"Model {k} (Normal vs Fault {k} [{known_classes[k]}])",
            }
        )

    if binary_conf_info:
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        n_models = len(binary_conf_info)
        n_cols = min(3, n_models)
        n_rows = int(np.ceil(n_models / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        for idx, info in enumerate(binary_conf_info):
            r = idx // n_cols
            c = idx % n_cols
            ax = axes[r, c]
            cm_bin = confusion_matrix(info["true"], info["pred"], labels=[0, 1])
            rs = cm_bin.sum(axis=1, keepdims=True)
            cm_pct = np.where(rs > 0, cm_bin.astype(float) / rs * 100, 0)
            annot = np.array([[f"{cm_pct[i, j]:.1f}%" for j in range(2)] for i in range(2)])
            sns.heatmap(
                cm_pct,
                annot=annot,
                fmt="",
                cmap="Blues",
                xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"],
                ax=ax,
                cbar=(idx == 0),
            )
            ax.set_title(info["title"])
            ax.set_ylabel("True")
            ax.set_xlabel("Pred")
        for idx in range(len(binary_conf_info), n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            axes[r, c].axis("off")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_binary_all_models_nvf.png"), dpi=150)
        plt.close()

    if len(closed_set_true) > 0:
        cm_closed = confusion_matrix(closed_set_true, closed_set_pred, labels=np.arange(K))
        rs = cm_closed.sum(axis=1, keepdims=True)
        cm_pct = np.where(rs > 0, cm_closed.astype(float) / rs * 100, 0)
        annot = np.array([[f"{cm_pct[i, j]:.1f}%" for j in range(K)] for i in range(K)])
        plt.rcParams["font.sans-serif"] = ["SimHei"]
        plt.rcParams["axes.unicode_minus"] = False
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_pct,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=known_classes,
            yticklabels=known_classes,
        )
        plt.title("Confusion Matrix (Closed-set, %)")
        plt.ylabel("True Label")
        plt.xlabel("Predicted Label")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "confusion_matrix_closed.png"), dpi=150)
        plt.close()

    label_names = [str(c) for c in known_classes] + ["Unknown"]
    cm_labels = []
    cm_preds = []
    for tl, pl in zip(all_labels, all_preds):
        cm_labels.append(int(tl) if int(tl) < K else K)
        cm_preds.append(int(pl) if 0 <= int(pl) < K else K)
    cm = confusion_matrix(np.array(cm_labels), np.array(cm_preds), labels=np.arange(K + 1))
    rs = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(rs > 0, cm.astype(float) / rs * 100, 0)
    annot = np.array([[f"{cm_pct[i, j]:.1f}%" for j in range(K + 1)] for i in range(K + 1)])
    plt.rcParams["font.sans-serif"] = ["SimHei"]
    plt.rcParams["axes.unicode_minus"] = False
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=annot, fmt="", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.title("Confusion Matrix (method=edl_mean, OOD-Assisted, %)")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"), dpi=150)
    plt.close()

    f1 = auroc = far = mar = None
    if has_unknown_samples:
        true_u = (~known_mask).astype(int)
        pred_u = (all_preds == -1).astype(int)
        unc_logic = uncertainties_dict["logic"]
        f1 = f1_score(true_u, pred_u)
        auroc = roc_auc_score(true_u, unc_logic)
        tp = np.sum((true_u == 1) & (pred_u == 1))
        fp = np.sum((true_u == 0) & (pred_u == 1))
        tn = np.sum((true_u == 0) & (pred_u == 0))
        fn = np.sum((true_u == 1) & (pred_u == 0))
        far = fp / (fp + tn) * 100 if (fp + tn) > 0 else 0.0
        mar = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0.0
        fpr, tpr, _ = roc_curve(true_u, unc_logic)
        plt.figure()
        plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (area = {auroc:.2f})")
        plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (EDL NvF)")
        plt.legend(loc="lower right")
        plt.savefig(os.path.join(output_dir, "roc_curve_nvf.png"), dpi=150)
        plt.close()

    # 仅 logic 不确定度评价（分布图 + 文本）
    results_path = os.path.join(output_dir, "binary_test_results_fi.txt")
    with open(results_path, "w", encoding="utf-8") as f:
        f.write(f"Closed-set Accuracy: {accuracy * 100:.2f}%\n")
        f.write(f"OOD Threshold (logic): {uncertainty_threshold:.6f}\n")
        f.write(f"Avg binary acc (over heads): {np.mean(binary_accuracies) * 100:.2f}%\n" if binary_accuracies else "")
        if f1 is not None:
            f.write(f"F1 OOD (logic score): {f1:.4f}\nAUROC (logic): {auroc:.4f}\nFAR: {far:.2f}%\nMAR: {mar:.2f}%\n")
        if has_unknown_samples:
            unc = uncertainties_dict["logic"]
            true_u = (~known_mask).astype(int)
            a_m = roc_auc_score(true_u, unc)
            f.write(f"\nAUROC (logic): {a_m:.4f}\n")

            id_unc = unc[known_mask]
            ood_unc = unc[unknown_mask]
            with plt.style.context("default"):
                plt.figure(figsize=(8, 6))
                plt.hist(id_unc, bins=50, alpha=0.6, label="ID", density=True)
                if ood_unc is not None and len(ood_unc) > 0:
                    plt.hist(ood_unc, bins=50, alpha=0.6, label="OOD", density=True)
                plt.xlabel("OOD score / Uncertainty")
                plt.ylabel("Density")
                umax = float(np.nanmax(unc))
                if umax <= 1.01:
                    plt.xlim(0.0, 1.0)
                    plt.xticks(np.arange(0, 1.05, 0.05))
                plt.title(f"Uncertainty Distribution (logic) AUROC={a_m:.4f}")
                plt.legend()
                plt.grid(True, linestyle="--", alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, "uncertainty_distribution_mean_nvf.png"), dpi=150)
                plt.close()

    has_ood_row = np.any(all_labels >= K)
    M = K - 1
    if p_yes_matrix.size > 0:
        hard_preds = (p_yes_matrix > 0.5).astype(np.float32)
        pref_matrix, pref_rows = compute_class_model_stats_fi(hard_preds, all_labels, K, has_ood_row)
        plot_class_model_heatmap_fi(
            pref_matrix,
            pref_rows,
            M,
            title="Per-model Positive Prediction Ratio (NvF, %)",
            cbar_label="Ratio (%)",
            filename="per_model_class_preference_matrix_nvf.png",
            output_dir=output_dir,
            fmt=".1f",
            cmap="Blues",
            annotate=True,
            value_scale=100.0,
            vmin=0.0,
            vmax=100.0,
        )
    if u_k_matrix.size > 0:
        uk_stats, uk_rows = compute_class_model_stats_fi(u_k_matrix, all_labels, K, has_ood_row)
        unc_vals = uk_stats.reshape(-1)
        if unc_vals.size > 0:
            unc_vmin = float(np.percentile(unc_vals, 5))
            unc_vmax = float(np.percentile(unc_vals, 95))
            if unc_vmax <= unc_vmin:
                unc_vmin, unc_vmax = None, None
        else:
            unc_vmin, unc_vmax = None, None
        plot_class_model_heatmap_fi(
            uk_stats,
            uk_rows,
            M,
            title="Class-Model Mean EDL Uncertainty u_k (NvF)",
            cbar_label="mean u_k",
            filename="per_model_uncertainty_uk_nvf.png",
            output_dir=output_dir,
            fmt=".3f",
            cmap="mako",
            annotate=True,
            vmin=unc_vmin,
            vmax=unc_vmax,
        )
    if logits_pos_matrix.size > 0 and logits_neg_matrix.size > 0:
        logits_pos_stats, logits_rows = compute_class_model_stats_fi(
            logits_pos_matrix, all_labels, K, has_ood_row
        )
        logits_neg_stats, _ = compute_class_model_stats_fi(
            logits_neg_matrix, all_labels, K, has_ood_row
        )
        logits_range = np.concatenate(
            [logits_pos_stats.reshape(-1), logits_neg_stats.reshape(-1)], axis=0
        )
        if logits_range.size > 0:
            logits_vmin = float(np.percentile(logits_range, 5))
            logits_vmax = float(np.percentile(logits_range, 95))
            if logits_vmax <= logits_vmin:
                logits_vmin, logits_vmax = None, None
        else:
            logits_vmin, logits_vmax = None, None
        plot_class_model_heatmap_fi(
            logits_pos_stats,
            logits_rows,
            M,
            title="Class-Model Mean Positive Logits (NvF)",
            cbar_label="mean logits_pos",
            filename="per_model_logits_pos_nvf.png",
            output_dir=output_dir,
            fmt=".2f",
            cmap="rocket",
            annotate=True,
            vmin=logits_vmin,
            vmax=logits_vmax,
        )
        plot_class_model_heatmap_fi(
            logits_neg_stats,
            logits_rows,
            M,
            title="Class-Model Mean Negative Logits (NvF)",
            cbar_label="mean logits_neg",
            filename="per_model_logits_neg_nvf.png",
            output_dir=output_dir,
            fmt=".2f",
            cmap="rocket",
            annotate=True,
            vmin=logits_vmin,
            vmax=logits_vmax,
        )

    return {
        "accuracy": accuracy * 100.0,
        "avg_binary_acc": float(np.mean(binary_accuracies) * 100) if binary_accuracies else 0.0,
        "f1_score": f1,
        "auroc": auroc,
        "far": far,
        "mar": mar,
        "results_path": results_path,
    }


def main():
    parser = argparse.ArgumentParser(description="Fi-only 三态 NvF 测试")
    parser.add_argument("--config", type=str, default="configs/bench_NvF_LaoDA_fi_triple.yaml")
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--test_all_epochs", action="store_true")
    parser.add_argument(
        "--uncertainty_threshold",
        "--uncertainty",
        type=float,
        default=0.5,
        help="OOD 分数阈值（超过则判 Unknown）；--uncertainty 同义",
    )
    parser.add_argument("--threshold_from_val", action="store_true")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument(
        "--test_epochs",
        type=str,
        default=None,
        help="仅测 checkpoint/epochs/ 下指定轮次：单轮 80；区间 10-50；列表 1,5,90；混合 1,10-12。与配置 test_infer.test_epochs 二选一可叠加（命令行优先）。"
        "指定后会进入逐 epoch 测试流程（不必再开 --test_all_epochs）。",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config["data"]
    model_config = config["model"]
    train_config = config["train"]
    test_infer = train_config.get("test_infer", {})
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tau_normal = float(test_infer.get("fi_tau_normal", 0.5))
    tau_fault = float(test_infer.get("fi_tau_fault", 0.5))
    test_all_epochs = bool(args.test_all_epochs or test_infer.get("test_all_epochs", False))
    epoch_save_csv = bool(test_infer.get("epoch_save_sample_level_csv", False))

    epochs_wanted: list[int] | None = None
    spec_cli = getattr(args, "test_epochs", None)
    spec_yaml = test_infer.get("test_epochs")
    spec = (spec_cli or (str(spec_yaml).strip() if spec_yaml is not None else None) or "").strip()
    if spec:
        try:
            epochs_wanted = parse_test_epochs_spec(spec)
        except ValueError as ex:
            raise SystemExit(f"test_epochs 解析失败: {ex}") from ex

    run_per_epoch = bool(test_all_epochs or epochs_wanted is not None)

    checkpoint_dir = os.path.abspath(args.checkpoint or train_config["checkpoint_dir"])
    if not os.path.isdir(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    output_dir = args.output_dir or os.path.join(checkpoint_dir, "test_fi")
    os.makedirs(output_dir, exist_ok=True)

    known_classes = data_config["openset"]["known_classes"]
    K = len(known_classes)
    backbone_type = model_config.get("type", "LaoDA")
    batch_size = int(train_config.get("batch_size", 32))

    test_dataset = get_dataset(data_config, backbone_type, split="test", filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    has_unknown = np.any(test_dataset.y >= K)
    print(f"Fi-only 测试 K={K}, M={K - 1}, device={device}")

    if run_per_epoch:
        discovered = discover_epochs_fi(checkpoint_dir, K)
        epochs, missing = filter_discovered_epochs(discovered, epochs_wanted)
        if missing:
            print(
                f"警告: 以下 epoch 在 checkpoint 中不存在或未集齐 model_1..model_{K - 1}，已跳过: {missing}"
            )
        if not epochs:
            print(
                f"没有可测试的 epoch（已发现 {discovered or '无'}；"
                f"请求 {epochs_wanted if epochs_wanted is not None else '全部'}）"
            )
            return
        if epochs_wanted is not None:
            print(f"按指定测试 epoch: {epochs}")
        else:
            print(f"测试全部已保存 epoch: {epochs}")
        all_rows = []
        plot_root = os.path.join(output_dir, "epochs")
        os.makedirs(plot_root, exist_ok=True)
        if args.threshold_from_val:
            val_ds = get_dataset(data_config, backbone_type, split="test", filter_classes=known_classes)
            val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
            models0 = None
            try:
                models0 = load_models_fi(checkpoint_dir, K, backbone_type, device, epoch=None)
            except FileNotFoundError:
                e0 = epochs[0]
                models0 = load_models_fi(checkpoint_dir, K, backbone_type, device, epoch=e0)
                print(
                    f"提示: checkpoint 根目录无 model_1..pth，IQR 阈值改用 epochs/{e0}/ 权重估计"
                )
            thr = compute_uncertainty_threshold_iqr_fi(val_loader, models0, device, K, tau_normal, tau_fault)
            del models0
        else:
            thr = args.uncertainty_threshold
        for e in epochs:
            models_e = load_models_fi(checkpoint_dir, K, backbone_type, device, epoch=e)
            packs = run_test_loop_fi(models_e, test_loader, device, K, thr, tau_normal, tau_fault)
            ep_dir = os.path.join(plot_root, str(e))
            m = evaluate_fi_outputs(
                *packs[:10],
                known_classes,
                K,
                has_unknown,
                thr,
                ep_dir,
                save_sample_level_csv=epoch_save_csv,
            )
            all_rows.append((e, m["accuracy"], m.get("f1_score"), m.get("auroc"), m.get("far"), m.get("mar")))
            del models_e
        with open(os.path.join(output_dir, "test_results_all_epochs_fi.csv"), "w", encoding="utf-8") as f:
            f.write("epoch,accuracy,f1_score,auroc,far,mar\n")
            for r in all_rows:
                f.write(
                    f"{r[0]},{r[1]:.2f},{'' if r[2] is None else f'{r[2]:.4f}'},"
                    f"{'' if r[3] is None else f'{r[3]:.4f}'},"
                    f"{'' if r[4] is None else f'{r[4]:.2f}'},"
                    f"{'' if r[5] is None else f'{r[5]:.2f}'}\n"
                )
        best = max(all_rows, key=lambda x: (x[1], x[3] or -1))
        best_src = os.path.join(plot_root, str(best[0]))
        best_dst = os.path.join(output_dir, "best_epoch")
        if os.path.isdir(best_dst):
            shutil.rmtree(best_dst)
        shutil.copytree(best_src, best_dst)
        with open(os.path.join(best_dst, "selected_epoch.txt"), "w", encoding="utf-8") as ef:
            ef.write(
                f"{best[0]}\n"
                f"# 与 test_results_all_epochs_fi.csv 中按 closed accuracy 优先、AUROC 次之选出的最优轮一致\n"
            )
        print(f"best_epoch={best[0]} -> {best_dst}（已写入 {os.path.join(best_dst, 'selected_epoch.txt')}）")
        return

    models = load_models_fi(checkpoint_dir, K, backbone_type, device, epoch=None)
    if args.threshold_from_val:
        val_ds = get_dataset(data_config, backbone_type, split="test", filter_classes=known_classes)
        val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        thr = compute_uncertainty_threshold_iqr_fi(val_loader, models, device, K, tau_normal, tau_fault)
        print(f"IQR 阈值: {thr:.4f}")
    else:
        thr = args.uncertainty_threshold
        print(f"阈值: {thr}")

    packs = run_test_loop_fi(models, test_loader, device, K, thr, tau_normal, tau_fault)
    m = evaluate_fi_outputs(
        *packs[:10],
        known_classes,
        K,
        has_unknown,
        thr,
        output_dir,
        save_sample_level_csv=True,
    )
    print(f"Closed accuracy: {m['accuracy']:.2f}% 结果: {m['results_path']}")


if __name__ == "__main__":
    main()
