import argparse
import os
import sys
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style='whitegrid', font='SimHei', font_scale=1.0)
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.utils.helpers import load_config
from common.utils.data_loader import NpyPackDataset
from common.utils.data_loader_1d import NpyPackDataset1D
from common.edl_losses import relu_evidence
from models import get_model


def get_dataset(data_config, model_type, split='test', filter_classes=None):
    data_dir = data_config.get('data_dir')
    if data_dir is None:
        raise ValueError("配置文件中必须指定 data.data_dir")

    openset_config = data_config.get('openset', {})
    known_classes = openset_config.get('known_classes')
    unknown_classes = openset_config.get('unknown_classes', [])

    dataset_cls = NpyPackDataset1D if model_type == 'LaoDA' else NpyPackDataset
    return dataset_cls(
        data_dir=data_dir,
        split=split,
        filter_classes=filter_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes
    )


def discover_epochs(checkpoint_dir, K):
    epochs_dir = os.path.join(checkpoint_dir, 'epochs')
    if not os.path.isdir(epochs_dir):
        return []
    epochs = []
    for name in os.listdir(epochs_dir):
        if not name.isdigit():
            continue
        e = int(name)
        sub = os.path.join(epochs_dir, name)
        if os.path.isdir(sub) and all(os.path.exists(os.path.join(sub, f'model_{k}.pth')) for k in range(K)):
            epochs.append(e)
    return sorted(epochs)


def load_models(checkpoint_dir, K, backbone_type, device, epoch=None):
    models = []
    for k in range(K):
        if epoch is None:
            path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
        else:
            path_k = os.path.join(checkpoint_dir, 'epochs', str(epoch), f'model_{k}.pth')
        if not os.path.exists(path_k):
            raise FileNotFoundError(f"未找到 {path_k}")
        model_k = get_model(backbone_type, num_classes=2).to(device)
        ckpt = torch.load(path_k, map_location=device, weights_only=True)
        state = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
        model_k.load_state_dict(state, strict=True)
        model_k.eval()
        models.append(model_k)
    return models


def compute_uncertainty_threshold_iqr(
    val_loader,
    models,
    device,
    K,
    fault_fusion='legacy',
    fusion_tau=1.0,
    ood_score_mode='mean_all',
    ood_lambda=0.5,
    class_decision='nvf_fusion',
):
    for m in models:
        m.eval()
    uncertainties_list = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            u_k_list = []
            p_yes_list = []
            for k in range(K):
                logits_k = models[k](inputs)
                evidence_k = relu_evidence(logits_k)
                alpha_k = evidence_k + 1
                S_k = torch.sum(alpha_k, dim=1)
                u_k = 2.0 / S_k
                probs_k = alpha_k / S_k.unsqueeze(1)
                p_yes_k = probs_k[:, 1]
                u_k_list.append(u_k)
                p_yes_list.append(p_yes_k)

            u_k_stack = torch.stack(u_k_list, dim=1)  # [B, K]
            if class_decision == 'min_u_gated':
                _, score = gated_min_u_preds_and_ood(u_k_stack, p_yes_list, K)
            else:
                _, winner_fault_idx = build_nvf_final_probs_and_winner(
                    p_yes_list, K, fault_fusion=fault_fusion, fusion_tau=fusion_tau
                )
                score = compute_batch_ood_score(
                    u_k_stack, winner_fault_idx, ood_score_mode=ood_score_mode, ood_lambda=ood_lambda
                )
            uncertainties_list.append(score.cpu().numpy())
    all_u = np.concatenate(uncertainties_list)
    q1, q3 = np.percentile(all_u, [25, 75])
    iqr = q3 - q1
    return float(q3 + 1.5 * iqr)


def gated_min_u_preds_and_ood(u_k_stack, p_yes_list, K):
    """
    门控 + 最小不确定度闭集决策（与 NvF 模型0语义一致）：
    - 模型0：p_yes_list[0] 为「正常」正类概率；p_yes_0 > 0.5 → 预测类别 0，OOD=u_0。
    - 否则（故障分支）：在模型 1..K-1 中，仅考虑 p_yes_k > 0.5（子模型判为故障）的索引，
      在这些索引上取 u_k=2/S_k 最小者为故障类（argmin 平局取第一个最小值）。
    - 若无一子模型 p_yes>0.5：闭集类取 1..K-1 上 p_yes 最大者，OOD 分数=1.0（满分不确定度）。

    Args:
        u_k_stack: [B, K]
        p_yes_list: 长度 K 的 list，每项 [B]
        K: 已知类数

    Returns:
        preds: LongTensor [B]，取值 0..K-1
        ood_score: FloatTensor [B]
    """
    device = u_k_stack.device
    dtype = u_k_stack.dtype
    B = u_k_stack.shape[0]
    p0 = p_yes_list[0]
    normal_mask = p0 > 0.5

    if K <= 1:
        preds = torch.zeros(B, dtype=torch.long, device=device)
        ood_score = u_k_stack[:, 0]
        return preds, ood_score

    u_fault = u_k_stack[:, 1:K]  # [B, K-1]
    p_fault = torch.stack(p_yes_list[1:K], dim=1)
    valid = p_fault > 0.5
    has_any = valid.any(dim=1)

    inf = torch.tensor(float('inf'), device=device, dtype=dtype)
    u_masked = torch.where(valid, u_fault, inf)
    k_rel_minu = u_masked.argmin(dim=1)
    fault_pred_minu = (k_rel_minu + 1).long()

    k_rel_maxp = p_fault.argmax(dim=1)
    fault_pred_fallback = (k_rel_maxp + 1).long()

    fault_preds = torch.where(has_any, fault_pred_minu, fault_pred_fallback)

    ood_minu = u_k_stack.gather(1, fault_pred_minu.unsqueeze(1)).squeeze(1)
    ood_fallback = torch.ones(B, device=device, dtype=dtype)
    ood_fault = torch.where(has_any, ood_minu, ood_fallback)

    preds = torch.where(
        normal_mask,
        torch.zeros(B, dtype=torch.long, device=device),
        fault_preds,
    )
    ood_score = torch.where(normal_mask, u_k_stack[:, 0], ood_fault)
    return preds, ood_score


def build_nvf_final_probs_and_winner(p_yes_list, K, fault_fusion='legacy', fusion_tau=1.0):
    """
    根据 NvF 逻辑构造最终类别分数并返回故障winner索引（绝对类别索引）。
    winner_fault_idx 仅在故障子模型(1..K-1)内竞争得到。
    """
    p_normal = p_yes_list[0]
    p_fault_total = 1.0 - p_normal
    fault_branch_scores = None

    if K <= 1:
        final_probs = p_normal.unsqueeze(1)
        winner_fault_idx = torch.zeros_like(p_normal, dtype=torch.long)
        return final_probs, winner_fault_idx

    fault_q = torch.stack(p_yes_list[1:], dim=1)  # [B, K-1]
    if fault_fusion == 'softmax':
        tau = max(float(fusion_tau), 1e-6)
        fault_w = F.softmax(fault_q / tau, dim=1)
        fault_branch_scores = fault_w
        fault_final = p_fault_total.unsqueeze(1) * fault_w
    else:
        fault_branch_scores = fault_q
        fault_final = p_fault_total.unsqueeze(1) * fault_q

    final_probs = torch.cat([p_normal.unsqueeze(1), fault_final], dim=1)
    winner_fault_rel = fault_branch_scores.argmax(dim=1)
    winner_fault_idx = winner_fault_rel + 1
    return final_probs, winner_fault_idx


def compute_batch_ood_score(u_k_stack, winner_fault_idx, ood_score_mode='mean_all', ood_lambda=0.5):
    if ood_score_mode == 'winner_only':
        return u_k_stack.gather(1, winner_fault_idx.view(-1, 1)).squeeze(1)
    if ood_score_mode == 'key_models':
        lam = float(ood_lambda)
        lam = min(max(lam, 0.0), 1.0)
        u0 = u_k_stack[:, 0]
        u_winner = u_k_stack.gather(1, winner_fault_idx.view(-1, 1)).squeeze(1)
        return lam * u0 + (1.0 - lam) * u_winner
    return u_k_stack.mean(dim=1)


def run_test_loop(
    models,
    test_loader,
    device,
    K,
    uncertainty_threshold,
    fault_fusion='legacy',
    fusion_tau=1.0,
    ood_score_mode='mean_all',
    ood_lambda=0.5,
    class_decision='nvf_fusion',
):
    all_labels = []
    all_preds = []
    all_uncertainties = []
    binary_preds_list = []
    closed_set_true = []
    closed_set_pred = []
    p_yes_rows = []
    u_k_rows = []
    logits_pos_rows = []
    logits_neg_rows = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            p_yes_list = []
            u_k_list = []
            logits_pos_list = []
            logits_neg_list = []
            for k in range(K):
                logits_k = models[k](inputs)
                evidence_k = relu_evidence(logits_k)
                alpha_k = evidence_k + 1
                S_k = torch.sum(alpha_k, dim=1)
                probs_k = alpha_k / S_k.unsqueeze(1)
                p_yes_k = probs_k[:, 1]
                u_k_k = 2.0 / S_k

                p_yes_list.append(p_yes_k)
                u_k_list.append(u_k_k)

                logits_pos_list.append(logits_k[:, 1])
                logits_neg_list.append(logits_k[:, 0])

            if class_decision == 'min_u_gated':
                u_k_stack_early = torch.stack(u_k_list, dim=1)
                preds_t, ood_t = gated_min_u_preds_and_ood(u_k_stack_early, p_yes_list, K)
                preds_local = preds_t.cpu().numpy()
                winner_fault_idx = None  # 仅 nvf_fusion 路径使用
            else:
                final_probs, winner_fault_idx = build_nvf_final_probs_and_winner(
                    p_yes_list, K, fault_fusion=fault_fusion, fusion_tau=fusion_tau
                )
                preds_local = final_probs.argmax(dim=1).cpu().numpy()

            binary_preds = []
            binary_preds.append((p_yes_list[0] > 0.5).cpu().numpy().astype(np.int32))
            for k in range(1, K):
                binary_preds.append((p_yes_list[k] > 0.5).cpu().numpy().astype(np.int32))
            binary_preds_list.append(np.stack(binary_preds, axis=1))

            # 记录每个样本在各模型下的正类概率和 u_k 以及 logits
            p_yes_stack = torch.stack(p_yes_list, dim=1)       # [B, K]
            u_k_stack = torch.stack(u_k_list, dim=1)           # [B, K]
            logits_pos_stack = torch.stack(logits_pos_list, dim=1)  # [B, K]
            logits_neg_stack = torch.stack(logits_neg_list, dim=1)  # [B, K]

            p_yes_rows.append(p_yes_stack.cpu().numpy())
            u_k_rows.append(u_k_stack.cpu().numpy())
            logits_pos_rows.append(logits_pos_stack.cpu().numpy())
            logits_neg_rows.append(logits_neg_stack.cpu().numpy())

            for i in range(len(labels)):
                if int(labels[i]) < K:
                    closed_set_true.append(int(labels[i]))
                    closed_set_pred.append(int(preds_local[i]))

            if class_decision == 'min_u_gated':
                uncertainty = ood_t.cpu().numpy()
            else:
                uncertainty = compute_batch_ood_score(
                    u_k_stack, winner_fault_idx, ood_score_mode=ood_score_mode, ood_lambda=ood_lambda
                ).cpu().numpy()

            for i in range(len(labels)):
                u_val = float(uncertainty[i])
                all_uncertainties.append(u_val)
                if u_val > uncertainty_threshold:
                    all_preds.append(-1)
                else:
                    pl = int(preds_local[i])
                    all_preds.append(pl if pl < K else -1)
            all_labels.extend(labels.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_uncertainties = np.array(all_uncertainties)
    binary_preds_matrix = np.concatenate(binary_preds_list, axis=0)
    closed_set_true = np.array(closed_set_true)
    closed_set_pred = np.array(closed_set_pred)
    p_yes_matrix = np.concatenate(p_yes_rows, axis=0)
    u_k_matrix = np.concatenate(u_k_rows, axis=0)
    logits_pos_matrix = np.concatenate(logits_pos_rows, axis=0)
    logits_neg_matrix = np.concatenate(logits_neg_rows, axis=0)
    return (
        all_labels,
        all_preds,
        all_uncertainties,
        binary_preds_matrix,
        closed_set_true,
        closed_set_pred,
        p_yes_matrix,
        u_k_matrix,
        logits_pos_matrix,
        logits_neg_matrix,
    )


def compute_class_model_stats(values_matrix, all_labels, K, has_unknown, reduce='mean'):
    """
    将 [N, K] 的 per-sample, per-model 指标聚合为 [C, K] 的类-模型统计矩阵。
    C=K(+1)，额外一行为 OOD（若存在未知类样本）。
    """
    if reduce not in ['mean', 'median']:
        raise ValueError(f"不支持的 reduce={reduce}，仅支持 mean/median")

    class_indices = list(range(K))
    if has_unknown:
        class_indices.append(K)

    stat_matrix = np.zeros((len(class_indices), K), dtype=np.float32)
    row_labels = [f'Class {i}' for i in range(K)]
    if has_unknown:
        row_labels.append('OOD')

    for row_idx, c in enumerate(class_indices):
        if c < K:
            mask = (all_labels == c)
        else:
            mask = (all_labels >= K)
        if not np.any(mask):
            continue
        subset = values_matrix[mask]  # [Nc, K]
        if reduce == 'mean':
            stat_matrix[row_idx] = subset.mean(axis=0)
        else:
            stat_matrix[row_idx] = np.median(subset, axis=0)
    return stat_matrix, row_labels


def plot_class_model_heatmap(
    stat_matrix,
    row_labels,
    K,
    title,
    cbar_label,
    filename,
    output_dir,
    fmt='.3f',
    cmap='mako',
    annotate=True,
    value_scale=1.0,
    vmin=None,
    vmax=None,
):
    """绘制统一风格的“故障类型 × 子模型”热力图。"""
    to_plot = stat_matrix * value_scale
    figsize_w = max(9, 1.15 * K + 4)
    figsize_h = max(4.5, 0.8 * len(row_labels) + 2.4)
    plt.figure(figsize=(figsize_w, figsize_h))
    ax = sns.heatmap(
        to_plot,
        annot=annotate,
        fmt=fmt,
        cmap=cmap,
        xticklabels=[f'Model {i}' for i in range(K)],
        yticklabels=row_labels,
        cbar_kws={'label': cbar_label},
        linewidths=0.4,
        linecolor='white',
        vmin=vmin,
        vmax=vmax,
    )
    ax.set_xlabel('Sub-model')
    ax.set_ylabel('True Class')
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=180)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='EDL 集成二分类 (NvF)：仅使用 EDL mean 不确定度的测试脚本')
    parser.add_argument('--config', type=str, default='configs/bench_NvF_LaoDA.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test_all_epochs', '--all', action='store_true', dest='test_all_epochs')
    parser.add_argument('--uncertainty_threshold', '--uncertainty', type=float, default=0.5, dest='uncertainty_threshold')
    parser.add_argument('--threshold_from_val', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--fault_fusion', type=str, default=None, choices=['legacy', 'softmax'])
    parser.add_argument('--fusion_tau', type=float, default=None)
    parser.add_argument('--ood_score_mode', type=str, default=None, choices=['mean_all', 'key_models', 'winner_only'])
    parser.add_argument('--ood_lambda', type=float, default=None)
    parser.add_argument('--class_decision', type=str, default=None, choices=['nvf_fusion', 'min_u_gated'])
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']

    ensemble_strategy = train_config.get('ensemble_strategy', 'Normal_vs_Fault_i')
    if ensemble_strategy != 'Normal_vs_Fault_i':
        print(f"警告：配置中的 ensemble_strategy={ensemble_strategy}，但 test_NvF.py 仅支持 Normal_vs_Fault_i，将按该策略解释结果。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    test_infer_cfg = train_config.get('test_infer', {})
    fault_fusion = args.fault_fusion if args.fault_fusion is not None else test_infer_cfg.get('fault_fusion', 'legacy')
    fusion_tau = args.fusion_tau if args.fusion_tau is not None else float(test_infer_cfg.get('fusion_tau', 1.0))
    ood_score_mode = args.ood_score_mode if args.ood_score_mode is not None else test_infer_cfg.get('ood_score_mode', 'mean_all')
    ood_lambda = args.ood_lambda if args.ood_lambda is not None else float(test_infer_cfg.get('ood_lambda', 0.5))
    class_decision = args.class_decision if args.class_decision is not None else test_infer_cfg.get('class_decision', 'nvf_fusion')
    if class_decision not in ('nvf_fusion', 'min_u_gated'):
        raise ValueError(f"不支持的 class_decision={class_decision}，应为 nvf_fusion 或 min_u_gated")
    if fusion_tau <= 0:
        raise ValueError("fusion_tau 必须 > 0")
    ood_lambda = min(max(float(ood_lambda), 0.0), 1.0)
    print(
        f"测试策略: class_decision={class_decision}, fault_fusion={fault_fusion}, fusion_tau={fusion_tau:.4f}, "
        f"ood_score_mode={ood_score_mode}, ood_lambda={ood_lambda:.4f}"
    )
    if class_decision == 'min_u_gated':
        print(
            "  [min_u_gated] 闭集：模型0 判正常→类0；否则仅在 p_yes>0.5 的故障子模型中取 u 最小者为故障类；"
            "若全无 p_yes>0.5 则 argmax(p_yes) 定类且 OOD=1.0；其余 OOD=决策子模型 u。"
            "（忽略 ood_score_mode/ood_lambda；fault_fusion/fusion_tau 不参与类别决策。）"
        )

    if args.checkpoint:
        checkpoint_dir = os.path.abspath(args.checkpoint)
    else:
        checkpoint_dir = train_config['checkpoint_dir']
    if not os.path.isdir(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    output_dir = args.output_dir or os.path.join(checkpoint_dir, 'test')
    os.makedirs(output_dir, exist_ok=True)

    known_classes = data_config['openset']['known_classes']
    unknown_classes = data_config['openset'].get('unknown_classes', [])
    K = len(known_classes)

    backbone_type = model_config.get('type', 'ResNet18_2d_Light')

    test_dataset = get_dataset(data_config, backbone_type, split='test', filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
    has_unknown_samples = any([v >= K for v in test_dataset.y])
    print(f"测试集大小: {len(test_dataset)}")

    if args.test_all_epochs:
        epochs = discover_epochs(checkpoint_dir, K)
        if not epochs:
            print("未找到任何 epoch 权重（需 checkpoint_dir/epochs/*/ 下 model_k.pth），请先训练并开启 save_every_epoch。")
            return
        print(f"全测试模式：共 {len(epochs)} 个 epoch，将依次测试并汇总到 test_results_all_epochs_nvf.csv")
        models = load_models(checkpoint_dir, K, backbone_type, device, epoch=None)
        if args.threshold_from_val:
            val_dataset = get_dataset(data_config, backbone_type, split='test', filter_classes=known_classes)
            val_loader = DataLoader(val_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
            uncertainty_threshold = compute_uncertainty_threshold_iqr(
                val_loader,
                models,
                device,
                K,
                fault_fusion=fault_fusion,
                fusion_tau=fusion_tau,
                ood_score_mode=ood_score_mode,
                ood_lambda=ood_lambda,
                class_decision=class_decision,
            )
        else:
            uncertainty_threshold = args.uncertainty_threshold
        del models
        all_epoch_rows = []
        for e in epochs:
            models_e = load_models(checkpoint_dir, K, backbone_type, device, epoch=e)
            all_labels_e, all_preds_e, all_uncertainties_e, binary_preds_matrix_e, closed_set_true_e, closed_set_pred_e, *_ = run_test_loop(
                models_e,
                test_loader,
                device,
                K,
                uncertainty_threshold,
                fault_fusion=fault_fusion,
                fusion_tau=fusion_tau,
                ood_score_mode=ood_score_mode,
                ood_lambda=ood_lambda,
                class_decision=class_decision,
            )
            known_mask_e = all_labels_e < K
            mapped_known_e = all_labels_e[known_mask_e]
            known_preds_e = all_preds_e[known_mask_e]
            acc_e = accuracy_score(mapped_known_e, known_preds_e) * 100.0
            f1_e = auroc_e = far_e = mar_e = None
            if has_unknown_samples:
                true_unk = (~known_mask_e).astype(int)
                pred_unk = (all_preds_e == -1).astype(int)
                f1_e = f1_score(true_unk, pred_unk)
                auroc_e = roc_auc_score(true_unk, all_uncertainties_e)
                tp = np.sum((true_unk == 1) & (pred_unk == 1))
                fp = np.sum((true_unk == 0) & (pred_unk == 1))
                tn = np.sum((true_unk == 0) & (pred_unk == 0))
                fn = np.sum((true_unk == 1) & (pred_unk == 0))
                far_e = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
                mar_e = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
            all_epoch_rows.append({
                'epoch': e, 'accuracy': acc_e, 'f1_score': f1_e, 'auroc': auroc_e, 'far': far_e, 'mar': mar_e
            })
            del models_e
        csv_path = os.path.join(output_dir, 'test_results_all_epochs_nvf.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('epoch,accuracy,f1_score,auroc,far,mar\n')
            for r in all_epoch_rows:
                f1_s = '' if r['f1_score'] is None else f"{r['f1_score']:.4f}"
                auroc_s = '' if r['auroc'] is None else f"{r['auroc']:.4f}"
                far_s = '' if r['far'] is None else f"{r['far']:.2f}"
                mar_s = '' if r['mar'] is None else f"{r['mar']:.2f}"
                f.write(f"{r['epoch']},{r['accuracy']:.2f},{f1_s},{auroc_s},{far_s},{mar_s}\n")
        print(f"全 epoch 测试结果已保存: {csv_path}")

    models = load_models(checkpoint_dir, K, backbone_type, device, epoch=None)
    if args.threshold_from_val:
        val_dataset = get_dataset(data_config, backbone_type, split='test', filter_classes=known_classes)
        val_loader = DataLoader(val_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
        uncertainty_threshold = compute_uncertainty_threshold_iqr(
            val_loader,
            models,
            device,
            K,
            fault_fusion=fault_fusion,
            fusion_tau=fusion_tau,
            ood_score_mode=ood_score_mode,
            ood_lambda=ood_lambda,
            class_decision=class_decision,
        )
        print(f"由已知类测试子集 IQR 得到不确定性阈值: {uncertainty_threshold:.4f}")
    else:
        uncertainty_threshold = args.uncertainty_threshold
        print(f"阈值 (u > 此值判 OOD): {uncertainty_threshold}")

    all_labels, all_preds, all_uncertainties, binary_preds_matrix, closed_set_true, closed_set_pred, p_yes_matrix, u_k_matrix, logits_pos_matrix, logits_neg_matrix = run_test_loop(
        models,
        test_loader,
        device,
        K,
        uncertainty_threshold,
        fault_fusion=fault_fusion,
        fusion_tau=fusion_tau,
        ood_score_mode=ood_score_mode,
        ood_lambda=ood_lambda,
        class_decision=class_decision,
    )

    known_mask = all_labels < K
    unknown_mask = ~known_mask
    id_unc = all_uncertainties[known_mask]
    ood_unc = all_uncertainties[unknown_mask] if has_unknown_samples else None

    mapped_known_labels = all_labels[known_mask]
    known_preds_mapped = all_preds[known_mask]
    if len(mapped_known_labels) == 0:
        print("错误: 没有有效的已知类样本")
        return

    accuracy = accuracy_score(mapped_known_labels, known_preds_mapped)
    print(f"已知类准确率 (Closed-set Accuracy): {accuracy * 100:.2f}%")

    # 二分类模型评估（按 NvF 逻辑）
    known_idx = np.where(known_mask)[0]
    binary_accuracies = []
    binary_conf_info = []

    # 模型 0: Normal vs All Faults
    true_0 = (all_labels[known_idx] == 0).astype(np.int32)
    pred_0 = binary_preds_matrix[known_idx, 0]
    acc_0 = accuracy_score(true_0, pred_0)
    binary_accuracies.append(acc_0)
    title_0 = "Model 0 (Normal vs All Faults)"
    binary_conf_info.append({'model_idx': 0, 'true': true_0, 'pred': pred_0, 'title': title_0})

    for k in range(1, K):
        mask_k = (all_labels[known_idx] == 0) | (all_labels[known_idx] == k)
        idx_k = known_idx[mask_k]
        if len(idx_k) > 0:
            true_k = (all_labels[idx_k] == k).astype(np.int32)
            pred_k = binary_preds_matrix[idx_k, k]
            acc_k = accuracy_score(true_k, pred_k)
            binary_accuracies.append(acc_k)
            title_k = f"Model {k} (Normal vs Fault {k} [{known_classes[k]}])"
            binary_conf_info.append({'model_idx': k, 'true': true_k, 'pred': pred_k, 'title': title_k})

    print(f"平均二分类准确率: {np.mean(binary_accuracies) * 100:.2f}%")

    if len(binary_conf_info) > 0:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
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
            t_bin = info['true']
            p_bin = info['pred']
            title_bin = info['title']
            cm_bin = confusion_matrix(t_bin, p_bin, labels=[0, 1])
            row_sum_bin = cm_bin.sum(axis=1, keepdims=True)
            cm_bin_pct = np.where(row_sum_bin > 0, cm_bin.astype(float) / row_sum_bin * 100, 0)
            annot_bin = np.array([[f'{cm_bin_pct[i, j]:.1f}%' for j in range(cm_bin_pct.shape[1])] for i in range(cm_bin_pct.shape[0])])
            sns.heatmap(cm_bin_pct, annot=annot_bin, fmt='', cmap='Blues',
                        xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'],
                        ax=ax, cbar=(idx == 0))
            ax.set_title(f'{title_bin}')
            ax.set_ylabel('True')
            ax.set_xlabel('Pred')
        for idx in range(len(binary_conf_info), n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            axes[r, c].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_binary_all_models_nvf.png'), dpi=150)
        plt.close()

    # 闭集混淆矩阵（仅已知类，忽略 OOD 判别）
    if len(closed_set_true) > 0:
        cm_closed = confusion_matrix(closed_set_true, closed_set_pred, labels=np.arange(K))
        row_sum_closed = cm_closed.sum(axis=1, keepdims=True)
        cm_closed_pct = np.where(row_sum_closed > 0, cm_closed.astype(float) / row_sum_closed * 100, 0)
        annot_closed = np.array(
            [[f'{cm_closed_pct[i, j]:.1f}%' for j in range(cm_closed_pct.shape[1])] for i in range(cm_closed_pct.shape[0])]
        )

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm_closed_pct,
            annot=annot_closed,
            fmt='',
            cmap='Blues',
            xticklabels=known_classes,
            yticklabels=known_classes,
        )
        plt.title('Confusion Matrix (Closed-set, %)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_closed.png'), dpi=150)
        plt.close()

    # 全集混淆矩阵（已知类 + Unknown，使用 EDL mean 阈值辅助 OOD 判别）
    label_names = [str(c) for c in known_classes] + ['Unknown']
    cm_labels = []
    cm_preds = []
    for true_label, pred_label in zip(all_labels, all_preds):
        true_label = int(true_label)
        pred_label = int(pred_label)

        cm_labels.append(true_label if true_label < K else K)
        cm_preds.append(pred_label if 0 <= pred_label < K else K)

    if len(cm_labels) > 0:
        cm_labels = np.array(cm_labels)
        cm_preds = np.array(cm_preds)
        cm = confusion_matrix(cm_labels, cm_preds, labels=np.arange(K + 1))
        row_sum = cm.sum(axis=1, keepdims=True)
        cm_pct = np.where(row_sum > 0, cm.astype(float) / row_sum * 100, 0)
        annot = np.array([[f'{cm_pct[i, j]:.1f}%' for j in range(cm_pct.shape[1])] for i in range(cm_pct.shape[0])])

        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues', xticklabels=label_names, yticklabels=label_names)
        plt.title('Confusion Matrix (method=edl_mean, OOD-Assisted, %)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
        plt.close()

    if has_unknown_samples:
        true_is_unknown = (~known_mask).astype(int)
        pred_is_unknown = (all_preds == -1).astype(int)
        f1 = f1_score(true_is_unknown, pred_is_unknown)
        print(f"F1-Score (检测未知类): {f1:.4f}")
        auroc = roc_auc_score(true_is_unknown, all_uncertainties)
        print(f"AUROC (区分已知/未知): {auroc:.4f}")
        tp = np.sum((true_is_unknown == 1) & (pred_is_unknown == 1))
        fp = np.sum((true_is_unknown == 0) & (pred_is_unknown == 1))
        tn = np.sum((true_is_unknown == 0) & (pred_is_unknown == 0))
        fn = np.sum((true_is_unknown == 1) & (pred_is_unknown == 0))
        far = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
        mar = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        print(f"FAR (虚警率): {far:.2f}%")
        print(f"MAR (漏警率): {mar:.2f}%")
        fpr, tpr, _ = roc_curve(true_is_unknown, all_uncertainties)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (EDL NvF)')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, 'roc_curve_nvf.png'), dpi=150)
        plt.close()

        # 不确定度分布图：ID vs OOD（NvF，使用 mean u_k），配色与 old 版本保持一致
        with plt.style.context('default'):
            plt.figure(figsize=(8, 6))
            # 与 old test.py 保持一致：使用 matplotlib 默认配色与参数
            plt.hist(id_unc, bins=30, alpha=0.6, label='ID', density=True)
            if ood_unc is not None and len(ood_unc) > 0:
                plt.hist(ood_unc, bins=30, alpha=0.6, label='OOD', density=True)
            plt.xlabel('OOD score / Uncertainty')
            plt.ylabel('Density')
            plt.xlim(0.0, 1.0)
            plt.xticks(np.arange(0, 1.05, 0.05))
            plt.title(f'Uncertainty Distribution (mean, AUROC={auroc:.4f})')
            plt.legend()
            plt.grid(True, linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'uncertainty_distribution_mean_nvf.png'), dpi=150)
            plt.close()

    # ====== NvF: 矩阵化可视化：故障类型 × 子模型 ======
    has_ood_row = np.any(all_labels >= K)

    # 1) 正类预测比例矩阵（基于每个子模型自身的二分类正类概率 p_yes）
    if p_yes_matrix.size > 0:
        hard_preds = (p_yes_matrix > 0.5).astype(np.float32)  # [N, K]
        pref_matrix, pref_rows = compute_class_model_stats(
            hard_preds,
            all_labels,
            K,
            has_ood_row,
            reduce='mean',
        )
        plot_class_model_heatmap(
            pref_matrix,
            pref_rows,
            K,
            title='Per-model Positive Prediction Ratio (NvF, %)',
            cbar_label='Ratio (%)',
            filename='per_model_class_preference_matrix_nvf.png',
            output_dir=output_dir,
            fmt='.1f',
            cmap='Blues',
            annotate=True,
            value_scale=100.0,
            vmin=0.0,
            vmax=100.0,
        )

    # 2) 不确定度矩阵（EDL mean u_k）
    if u_k_matrix.size > 0:
        uk_stats, uk_rows = compute_class_model_stats(
            u_k_matrix, all_labels, K, has_ood_row, reduce='mean'
        )
        unc_vals = uk_stats.reshape(-1)
        if unc_vals.size > 0:
            unc_vmin = float(np.percentile(unc_vals, 5))
            unc_vmax = float(np.percentile(unc_vals, 95))
            if unc_vmax <= unc_vmin:
                unc_vmin, unc_vmax = None, None
        else:
            unc_vmin, unc_vmax = None, None

        plot_class_model_heatmap(
            uk_stats,
            uk_rows,
            K,
            title='Class-Model Mean EDL Uncertainty u_k (NvF)',
            cbar_label='mean u_k',
            filename='per_model_uncertainty_uk_nvf.png',
            output_dir=output_dir,
            fmt='.3f',
            cmap='mako',
            annotate=True,
            vmin=unc_vmin,
            vmax=unc_vmax,
        )

    # 3) logits 正/负类矩阵
    if logits_pos_matrix.size > 0 and logits_neg_matrix.size > 0:
        logits_pos_stats, logits_rows = compute_class_model_stats(
            logits_pos_matrix, all_labels, K, has_ood_row, reduce='mean'
        )
        logits_neg_stats, _ = compute_class_model_stats(
            logits_neg_matrix, all_labels, K, has_ood_row, reduce='mean'
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

        plot_class_model_heatmap(
            logits_pos_stats,
            logits_rows,
            K,
            title='Class-Model Mean Positive Logits (NvF)',
            cbar_label='mean logits_pos',
            filename='per_model_logits_pos_nvf.png',
            output_dir=output_dir,
            fmt='.2f',
            cmap='rocket',
            annotate=True,
            vmin=logits_vmin,
            vmax=logits_vmax,
        )
        plot_class_model_heatmap(
            logits_neg_stats,
            logits_rows,
            K,
            title='Class-Model Mean Negative Logits (NvF)',
            cbar_label='mean logits_neg',
            filename='per_model_logits_neg_nvf.png',
            output_dir=output_dir,
            fmt='.2f',
            cmap='rocket',
            annotate=True,
            vmin=logits_vmin,
            vmax=logits_vmax,
        )

    results_path = os.path.join(output_dir, 'binary_test_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Closed-set Accuracy: {accuracy * 100:.2f}%\n")
        f.write(
            f"Strategy: class_decision={class_decision}, fault_fusion={fault_fusion}, fusion_tau={fusion_tau:.4f}, "
            f"ood_score_mode={ood_score_mode}, ood_lambda={ood_lambda:.4f}\n"
        )
        f.write(f"OOD Threshold: {uncertainty_threshold:.6f}\n")
        f.write(f"Average Binary Accuracy: {np.mean(binary_accuracies) * 100:.2f}%\n")
    print(f"结果摘要已保存: {results_path}")


if __name__ == '__main__':
    main()
