import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
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
    """根据配置动态加载数据集，filter_classes=None 时加载全量（已知+未知）。"""
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

def compute_uncertainty_threshold_iqr(val_loader, models, device, K):
    """
    对验证集计算多种方法的不确定度分布，并分别返回各自的 IQR 阈值。
    （all_rest 为 0/1 掩码，这里不依赖阈值，可选地给出一个固定值）
    返回 dict: {method: threshold}
    """
    for m in models:
        m.eval()
    scores = {'max_prob': [], 'edl_mean': [], 'edl_positive_only': [], 'edl_dynamic': [], 'all_rest': []}
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            p_yes_list = []
            u_k_list = []
            alpha_pos_list = []
            for k in range(K):
                logits_k = models[k](inputs)
                evidence_k = relu_evidence(logits_k)
                alpha_k = evidence_k + 1
                S_k = torch.sum(alpha_k, dim=1)
                probs_k = alpha_k / S_k.unsqueeze(1)
                p_yes_k = probs_k[:, 1]
                u_k_k = 2.0 / S_k
                alpha_pos_k = alpha_k[:, 1]

                p_yes_list.append(p_yes_k)
                u_k_list.append(u_k_k)
                alpha_pos_list.append(alpha_pos_k)
            p_yes_stack = torch.stack(p_yes_list, dim=1)
            max_p_yes = p_yes_stack.max(dim=1)[0]
            score_max_prob = (1.0 - max_p_yes).cpu().numpy()

            u_k_stack = torch.stack(u_k_list, dim=1)
            score_mean = u_k_stack.mean(dim=1).cpu().numpy()

            alpha_pos_all = torch.stack(alpha_pos_list, dim=1)
            S_pos = alpha_pos_all.sum(dim=1).clamp(min=1e-6)
            score_positive_only = (K / S_pos).cpu().numpy()

            winner = p_yes_stack.argmax(dim=1)
            winner_alpha_pos = alpha_pos_all[torch.arange(alpha_pos_all.size(0), device=alpha_pos_all.device), winner]
            score_dynamic = (2.0 / (winner_alpha_pos + 1.0)).cpu().numpy()

            all_rest_mask = (p_yes_stack <= 0.5).all(dim=1).cpu().numpy().astype(np.float32)

            scores['max_prob'].append(score_max_prob)
            scores['edl_mean'].append(score_mean)
            scores['edl_positive_only'].append(score_positive_only)
            scores['edl_dynamic'].append(score_dynamic)
            scores['all_rest'].append(all_rest_mask)

    thresholds = {}
    for k, chunks in scores.items():
        all_u = np.concatenate(chunks)
        if k == 'all_rest':
            # all_rest 只是 0/1 掩码，这里的阈值不会真正用到，取 0.5 即可
            thresholds[k] = 0.5
        else:
            q1, q3 = np.percentile(all_u, [25, 75])
            iqr = q3 - q1
            thresholds[k] = float(q3 + 1.5 * iqr)
    return thresholds


def discover_epochs(checkpoint_dir, K):
    """从 checkpoint_dir/epochs/{e}/ 发现所有已保存的 epoch（每目录需含 model_0.pth ... model_{K-1}.pth）。"""
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
    """
    加载 K 个模型。epoch=None 时加载最佳权重 (model_k.pth)；否则从 epochs/{epoch}/ 加载。
    返回 list of models。
    """
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


def run_test_loop(models, test_loader, device, K, uncertainty_threshold):
    """
    跑一遍测试循环，同时计算多种 OOD 分数：
    - edl_mean
    - edl_positive_only
    - edl_dynamic
    - max_prob

    返回：
    - all_labels: numpy, 所有样本真实标签
    - preds_dict: {method: numpy}，每种方法下的预测（含 OOD=-1）
    - unc_dict: {method: numpy}，每种方法下的 OOD 分数 / 不确定度
    - binary_preds_matrix: numpy, 形状 [N, K]，每个子模型的二分类硬预测
    - closed_set_true, closed_set_pred: numpy, 仅已知类样本在集成下的 closed-set 预测
    - p_yes_matrix: numpy, 形状 [N, K]，每个样本在每个子模型下的正类概率
    - u_k_matrix: numpy, 形状 [N, K]，每个样本在每个子模型下的 EDL 不确定度 u_k
    """
    all_labels = []
    methods = ['edl_mean', 'edl_positive_only', 'edl_dynamic', 'max_prob', 'all_rest']
    preds_dict = {m: [] for m in methods}
    unc_dict = {m: [] for m in methods}
    binary_preds_list = []
    closed_set_true = []
    closed_set_pred = []
    p_yes_rows = []
    u_k_rows = []
    logits_pos_rows = []
    logits_neg_rows = []
    alpha_pos_rows = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            p_yes_list = []
            u_k_list = []
            alpha_pos_list = []
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
                alpha_pos_k = alpha_k[:, 1]

                p_yes_list.append(p_yes_k)
                u_k_list.append(u_k_k)
                alpha_pos_list.append(alpha_pos_k)

                logits_pos_list.append(logits_k[:, 1])
                logits_neg_list.append(logits_k[:, 0])

            p_yes_stack = torch.stack(p_yes_list, dim=1)  # [B, K]
            final_probs = p_yes_stack

            # 记录每个样本在各模型下的正类概率和 u_k
            u_k_stack = torch.stack(u_k_list, dim=1)      # [B, K]
            alpha_pos_all = torch.stack(alpha_pos_list, dim=1)  # [B, K]
            logits_pos_stack = torch.stack(logits_pos_list, dim=1)  # [B, K]
            logits_neg_stack = torch.stack(logits_neg_list, dim=1)  # [B, K]

            p_yes_rows.append(p_yes_stack.cpu().numpy())
            u_k_rows.append(u_k_stack.cpu().numpy())
            alpha_pos_rows.append(alpha_pos_all.cpu().numpy())
            logits_pos_rows.append(logits_pos_stack.cpu().numpy())
            logits_neg_rows.append(logits_neg_stack.cpu().numpy())

            # 每个子模型各自的二分类预测（P_yes > 0.5 视为“正类”）
            binary_preds = []
            for k in range(K):
                binary_preds.append((p_yes_list[k] > 0.5).cpu().numpy().astype(np.int32))
            binary_preds_list.append(np.stack(binary_preds, axis=1))

            preds_local = final_probs.argmax(dim=1).cpu().numpy()
            for i in range(len(labels)):
                if int(labels[i]) < K:
                    closed_set_true.append(int(labels[i]))
                    closed_set_pred.append(int(preds_local[i]))

            # 多种 OOD 分数
            max_p_yes = p_yes_stack.max(dim=1)[0]
            score_max_prob = (1.0 - max_p_yes).cpu().numpy()

            u_k_stack = torch.stack(u_k_list, dim=1)          # [B, K]
            score_mean = u_k_stack.mean(dim=1).cpu().numpy()  # edl_mean

            S_pos = alpha_pos_all.sum(dim=1).clamp(min=1e-6)
            score_positive_only = (K / S_pos).cpu().numpy()

            winner = final_probs.argmax(dim=1)
            winner_alpha_pos = alpha_pos_all[torch.arange(alpha_pos_all.size(0), device=alpha_pos_all.device), winner]
            score_dynamic = (2.0 / (winner_alpha_pos + 1.0)).cpu().numpy()

            scores = {
                'edl_mean': score_mean,
                'edl_positive_only': score_positive_only,
                'edl_dynamic': score_dynamic,
                'max_prob': score_max_prob,
            }

            # 基于每种分数的 OOD 判定与预测
            batch_labels = labels.cpu().numpy().tolist()
            all_labels.extend(batch_labels)

            for method, s in scores.items():
                for i in range(len(labels)):
                    u_val = float(s[i])
                    unc_dict[method].append(u_val)
                    if u_val > uncertainty_threshold:
                        preds_dict[method].append(-1)
                    else:
                        pl = int(preds_local[i])
                        preds_dict[method].append(pl if pl < K else -1)

            # all_rest 方法：若所有子模型都判为负类(Rest) 则记为 OOD（1），否则为 0
            all_rest_mask = (p_yes_stack <= 0.5).all(dim=1).cpu().numpy().astype(np.float32)
            for i in range(len(labels)):
                u_val = float(all_rest_mask[i])
                unc_dict['all_rest'].append(u_val)
                if all_rest_mask[i] == 1.0:
                    preds_dict['all_rest'].append(-1)
                else:
                    pl = int(preds_local[i])
                    preds_dict['all_rest'].append(pl if pl < K else -1)

    all_labels = np.array(all_labels)
    for k in preds_dict:
        preds_dict[k] = np.array(preds_dict[k])
        unc_dict[k] = np.array(unc_dict[k])
    binary_preds_matrix = np.concatenate(binary_preds_list, axis=0)
    closed_set_true = np.array(closed_set_true)
    closed_set_pred = np.array(closed_set_pred)
    p_yes_matrix = np.concatenate(p_yes_rows, axis=0)
    u_k_matrix = np.concatenate(u_k_rows, axis=0)
    alpha_pos_matrix = np.concatenate(alpha_pos_rows, axis=0)
    logits_pos_matrix = np.concatenate(logits_pos_rows, axis=0)
    logits_neg_matrix = np.concatenate(logits_neg_rows, axis=0)
    return (
        all_labels,
        preds_dict,
        unc_dict,
        binary_preds_matrix,
        closed_set_true,
        closed_set_pred,
        p_yes_matrix,
        u_k_matrix,
        alpha_pos_matrix,
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


def plot_per_model_overlaid_histograms(
    metric_matrix,
    all_labels,
    class_list,
    class_colors,
    class_names,
    K,
    title,
    filename,
    x_label,
    output_dir,
    bins=30,
    x_range=None,
):
    """每个子模型一个子图，叠加所有类别分布（用于详细诊断图，可选输出）。"""
    n_cols_plot = min(3, K) if K > 0 else 1
    n_rows_plot = int(np.ceil(K / n_cols_plot)) if K > 0 else 1
    fig, axes = plt.subplots(
        n_rows_plot,
        n_cols_plot,
        figsize=(5 * n_cols_plot, 4 * n_rows_plot),
        sharey=True,
    )
    axes_flat = axes.flatten() if isinstance(axes, np.ndarray) else np.array([axes]).flatten()

    legend_patches = [
        Patch(facecolor=class_colors[i], edgecolor='none', label=class_names[i])
        for i in range(len(class_list))
    ]

    for k_idx in range(K):
        ax = axes_flat[k_idx]
        for c_idx, c in enumerate(class_list):
            mask_c = (all_labels == c) if c < K else (all_labels >= K)
            vals = metric_matrix[mask_c, k_idx]
            if vals.size == 0:
                continue
            ax.hist(
                vals,
                bins=bins,
                range=x_range,
                alpha=0.35,
                density=True,
                histtype='stepfilled',
                color=class_colors[c_idx],
            )
        ax.set_title(f'Model {k_idx}')
        ax.set_xlabel(x_label)
        if k_idx == 0:
            ax.set_ylabel('Density')
        ax.grid(True, linestyle='--', alpha=0.3)

    for j in range(K, len(axes_flat)):
        axes_flat[j].set_visible(False)

    if legend_patches:
        fig.legend(
            legend_patches,
            [p.get_label() for p in legend_patches],
            loc='upper right',
            fontsize=8,
            frameon=True,
        )
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0, 0.82, 0.92])
    fig.savefig(os.path.join(output_dir, filename), dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description='EDL 集成二分类 (OvR)：测试 K 个模型，EDL 不确定度，OOD 指标')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/bench_OvR_LaoDA.yaml',
        help='配置文件路径（相对项目根；默认: configs/bench_OvR_LaoDA.yaml）'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint 目录（含 model_0.pth..model_{K-1}.pth）；未指定则从 config 的 checkpoint_dir 读取')
    parser.add_argument('--test_all_epochs', '--all', action='store_true', dest='test_all_epochs',
                        help='若指定，则对每个已保存的 epoch 权重都跑一遍测试；默认仅用最佳权重 (model_k.pth)')
    parser.add_argument('--uncertainty_threshold', '--uncertainty', type=float, default=0.5, dest='uncertainty_threshold')
    parser.add_argument('--threshold_from_val', action='store_true',
                        help='若指定，则从验证集不确定性 IQR 自动计算阈值，忽略 --uncertainty_threshold')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录；未指定则为 checkpoint_dir/test')
    # 旧版 agg/mode 参数保留但不再实际使用，仅为兼容
    parser.add_argument('--agg', type=str, default='mean', choices=['dynamic', 'mean', 'positive_only'],
                        help='兼容参数（当前脚本一次性计算多种聚合方式）')
    parser.add_argument('--mode', type=str, default='edl',
                        choices=['edl', 'max_prob'],
                        help="兼容参数（当前脚本一次性计算 mean/positive_only/max_prob/dynamic 四种方法）")
    parser.add_argument('--cm_method',
                        type=str,
                        default='edl_positive_only',
                        choices=['edl_mean', 'edl_positive_only', 'edl_dynamic', 'max_prob', 'all_rest'],
                        help='用于绘制含 OOD 总体混淆矩阵(confusion_matrix.png)时所采用的集成判断方法')
    parser.add_argument('--plot_detailed_distributions', action='store_true',
                        help='若指定，则额外输出详细的 per-model 叠加分布图；默认仅输出矩阵热力图和关键评估图')
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config['data'] 
    model_config = config['model']
    train_config = config['train']
    
    ensemble_strategy = train_config.get('ensemble_strategy', 'One_vs_Rest')
    if ensemble_strategy != 'One_vs_Rest':
        print(f"警告：配置文件中的策略为 {ensemble_strategy}，但本脚本(test_ovr.py)专用于 One_vs_Rest。将按 One_vs_Rest 逻辑执行。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    print("本脚本将同时评估: edl_mean / edl_positive_only / edl_dynamic / max_prob 四种方法")

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
    num_known_classes = len(known_classes)
    K = num_known_classes

    backbone_type = model_config.get('type', 'ResNet18_2d_Light')

    # 读取测试集全量（先建 loader，用于定阈值和 run_test_loop）
    test_dataset = get_dataset(data_config, backbone_type, split='test', filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
    has_unknown_samples = any([v >= K for v in test_dataset.y])
    print(f"测试集大小: {len(test_dataset)}")

    # 默认使用最佳权重；若指定 --test_all_epochs 则先对每个 epoch 跑一遍测试并写 csv，再以最佳权重做后续绘图
    if args.test_all_epochs:
        epochs = discover_epochs(checkpoint_dir, K)
        if not epochs:
            print("未找到任何 epoch 权重（需 checkpoint_dir/epochs/*/ 下 model_k.pth），请先训练并开启 save_every_epoch。")
            return
        print(f"全测试模式：共 {len(epochs)} 个 epoch，将依次测试并汇总到 test_results_all_epochs.csv")
        # 先加载最佳模型用于定阈值（与默认行为一致）
        models = load_models(checkpoint_dir, K, backbone_type, device, epoch=None)
        if args.threshold_from_val:
            val_dataset = get_dataset(data_config, backbone_type, split='test', filter_classes=known_classes)
            val_loader = DataLoader(val_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
            thresholds = compute_uncertainty_threshold_iqr(val_loader, models, device, K)
        else:
            thresholds = {m: args.uncertainty_threshold for m in ['edl_mean', 'edl_positive_only', 'edl_dynamic', 'max_prob']}
        del models
        # 全 epoch 模式下为了简化，不再分别输出每种方法，仅保留兼容逻辑（可按需扩展）
        all_epoch_rows = []
        for e in epochs:
            models_e = load_models(checkpoint_dir, K, backbone_type, device, epoch=e)
            all_labels_e, preds_dict_e, unc_dict_e, binary_preds_matrix_e, closed_set_true_e, closed_set_pred_e, *_ = run_test_loop(
                models_e, test_loader, device, K, thresholds['edl_mean']
            )
            # 这里以 edl_mean 作为代表性方法统计 epoch 级别精度
            known_mask_e = all_labels_e < K
            mapped_known_e = all_labels_e[known_mask_e]
            known_preds_e = preds_dict_e['edl_mean'][known_mask_e]
            acc_e = accuracy_score(mapped_known_e, known_preds_e) * 100.0
            all_epoch_rows.append({
                'epoch': e, 'accuracy': acc_e, 'f1_score': None, 'auroc': None, 'far': None, 'mar': None
            })
            del models_e
        csv_path = os.path.join(output_dir, 'test_results_all_epochs.csv')
        with open(csv_path, 'w', encoding='utf-8') as f:
            f.write('epoch,accuracy,f1_score,auroc,far,mar\n')
            for r in all_epoch_rows:
                f1_s = '' if r['f1_score'] is None else f"{r['f1_score']:.4f}"
                auroc_s = '' if r['auroc'] is None else f"{r['auroc']:.4f}"
                far_s = '' if r['far'] is None else f"{r['far']:.2f}"
                mar_s = '' if r['mar'] is None else f"{r['mar']:.2f}"
                f.write(f"{r['epoch']},{r['accuracy']:.2f},{f1_s},{auroc_s},{far_s},{mar_s}\n")
        print(f"全 epoch 测试结果已保存: {csv_path}")

    # 使用最佳权重 (model_k.pth) 做本次测试与绘图
    models = load_models(checkpoint_dir, K, backbone_type, device, epoch=None)
    print(f"已加载 K={K} 个模型（最佳权重 model_k.pth），backbone={backbone_type}")

    if args.threshold_from_val:
        val_dataset = get_dataset(data_config, backbone_type, split='test', filter_classes=known_classes)
        val_loader = DataLoader(val_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
        thresholds = compute_uncertainty_threshold_iqr(val_loader, models, device, K)
        for m, th in thresholds.items():
            print(f"由已知类测试子集 IQR 得到 {m} 不确定性阈值: {th:.4f}")
    else:
        thresholds = {m: args.uncertainty_threshold for m in ['edl_mean', 'edl_positive_only', 'edl_dynamic', 'max_prob', 'all_rest']}
        print(f"所有方法统一使用手动指定阈值: {args.uncertainty_threshold}")

    all_labels, preds_dict, unc_dict, binary_preds_matrix, closed_set_true, closed_set_pred, p_yes_matrix, u_k_matrix, alpha_pos_matrix, logits_pos_matrix, logits_neg_matrix = run_test_loop(
        models, test_loader, device, K, thresholds['edl_mean']
    )

    # 已知类的掩码
    known_mask = all_labels < K
    unknown_mask = ~known_mask
    mapped_known_labels = all_labels[known_mask]
    
    if len(mapped_known_labels) == 0:
        print("错误: 没有有效的已知类样本")
        return

    # 以 edl_mean 方法作为 closed-set 准确率代表
    known_preds_mapped = preds_dict['edl_mean'][known_mask]
    accuracy = accuracy_score(mapped_known_labels, known_preds_mapped)

    print(f"\n--- 各模型单独（二分类，仅已知类）---")
    known_idx = np.where(known_mask)[0]
    binary_accuracies = []
    binary_conf_info = []

    for k in range(K):
        true_k = (all_labels[known_idx] == k).astype(np.int32)
        pred_k = binary_preds_matrix[known_idx, k]
        acc_k = accuracy_score(true_k, pred_k)
        binary_accuracies.append(acc_k)
        title_k = f"Model {k} (Class {k} [{known_classes[k]}] vs Rest)"
        binary_conf_info.append({
            'model_idx': k, 'true': true_k, 'pred': pred_k, 'title': title_k
        })
        print(f"  {title_k}: 二分类准确率 = {acc_k * 100:.2f}%")
            
    print(f"  平均二分类准确率: {np.mean(binary_accuracies) * 100:.2f}%")

    # 各模型二分类混淆矩阵（放在一张图中，子图形式）
    if len(binary_conf_info) > 0:
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        n_models = len(binary_conf_info)
        n_cols = min(3, n_models)  # 最多每行 3 个子图
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

        # 对于没有用到的子图（如果有），隐藏
        for idx in range(len(binary_conf_info), n_rows * n_cols):
            r = idx // n_cols
            c = idx % n_cols
            axes[r, c].axis('off')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_binary_all_models.png'), dpi=150)
        plt.close()

    print(f"\n--- 集成整体 ---")
    print(f"已知类准确率 (Closed-set Accuracy): {accuracy * 100:.2f}%")

    test_results = {
        'accuracy': accuracy * 100,
        'f1_score': None,
        'auroc': None,
        'far': None,
        'mar': None,
        'has_unknown': has_unknown_samples,
    }

    if has_unknown_samples:
        # 对多种方法分别计算开放集指标并绘图
        method_pretty = {
            'edl_mean': 'mean',
            'edl_positive_only': 'positive_only',
            'edl_dynamic': 'dynamic',
            'max_prob': 'max_prob',
            'all_rest': 'all_rest',
        }
        ood_metrics = {}
        for method, scores in unc_dict.items():
            true_is_unknown = (~known_mask).astype(int)
            pred_is_unknown = (preds_dict[method] == -1).astype(int)
            f1 = f1_score(true_is_unknown, pred_is_unknown)
            auroc = roc_auc_score(true_is_unknown, scores)
            tp = np.sum((true_is_unknown == 1) & (pred_is_unknown == 1))
            fp = np.sum((true_is_unknown == 0) & (pred_is_unknown == 1))
            tn = np.sum((true_is_unknown == 0) & (pred_is_unknown == 0))
            fn = np.sum((true_is_unknown == 1) & (pred_is_unknown == 0))
            far = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
            mar = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
            ood_metrics[method] = {'f1': f1, 'auroc': auroc, 'far': far, 'mar': mar}

            print(f"\n[{method_pretty[method]}] F1-Score (检测未知类): {f1:.4f}")
            print(f"[{method_pretty[method]}] AUROC (区分已知/未知): {auroc:.4f}")
            print(f"[{method_pretty[method]}] FAR (虚警率): {far:.2f}%")
            print(f"[{method_pretty[method]}] MAR (漏警率): {mar:.2f}%")

            # ROC 曲线
            fpr, tpr, _ = roc_curve(true_is_unknown, scores)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'ROC Curve ({method_pretty[method]})')
            plt.legend(loc='lower right')
            plt.savefig(os.path.join(output_dir, f'roc_curve_{method_pretty[method]}.png'), dpi=150)
            plt.close()

            # 不确定度分布图：ID vs OOD
            id_unc_m = scores[known_mask]
            ood_unc_m = scores[unknown_mask]
            with plt.style.context('default'):
                plt.figure(figsize=(8, 6))
                # 与 old test.py 保持一致：使用 matplotlib 默认配色与参数
                plt.hist(id_unc_m, bins=30, alpha=0.6, label='ID', density=True)
                plt.hist(ood_unc_m, bins=30, alpha=0.6, label='OOD', density=True)
                plt.xlabel('OOD score / Uncertainty')
                plt.ylabel('Density')
                plt.xlim(0.0, 1.0)
                plt.xticks(np.arange(0, 1.05, 0.05))
                plt.title(f'Uncertainty Distribution ({method_pretty[method]}, AUROC={auroc:.4f})')
                plt.legend()
                plt.grid(True, linestyle='--', alpha=0.5)
                plt.tight_layout()
                plt.savefig(os.path.join(output_dir, f'uncertainty_distribution_{method_pretty[method]}.png'), dpi=150)
                plt.close()
    else:
        print("测试集中无未知类，跳过开放集指标与 ROC。")

    # 闭集混淆矩阵（仅已知类，忽略 OOD 判别）
    if len(closed_set_true) > 0:
        cm_closed = confusion_matrix(closed_set_true, closed_set_pred, labels=np.arange(K))
        row_sum_closed = cm_closed.sum(axis=1, keepdims=True)
        cm_closed_pct = np.where(row_sum_closed > 0, cm_closed.astype(float) / row_sum_closed * 100, 0)
        annot_closed = np.array([[f'{cm_closed_pct[i, j]:.1f}%' for j in range(cm_closed_pct.shape[1])] for i in range(cm_closed_pct.shape[0])])

        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用于显示中文标签
        plt.rcParams['axes.unicode_minus'] = False
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_closed_pct, annot=annot_closed, fmt='', cmap='Blues',
                    xticklabels=known_classes, yticklabels=known_classes)
        plt.title('Confusion Matrix (Closed-set, %)')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'confusion_matrix_closed.png'), dpi=150)
        plt.close()

    # 混淆矩阵（已知类 + Unknown，全数据集，含 OOD），可指定使用哪种集成方法
    label_names = [str(c) for c in known_classes] + ['Unknown']
    cm_labels = []
    cm_preds = []

    # 使用命令行参数 --cm_method 指定的方法来构造含 OOD 的混淆矩阵
    cm_method = args.cm_method
    for true_label, pred_label in zip(all_labels, preds_dict[cm_method]):
        true_label, pred_label = int(true_label), int(pred_label)
        
        if true_label < K:
            cm_labels.append(true_label)
        else:
            cm_labels.append(K)
            
        if pred_label == -1 or pred_label >= K:
            cm_preds.append(K)
        else:
            cm_preds.append(pred_label)
            
    cm_labels = np.array(cm_labels)
    cm_preds = np.array(cm_preds)
    cm = confusion_matrix(cm_labels, cm_preds)
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sum > 0, cm.astype(float) / row_sum * 100, 0)
    annot = np.array([[f'{cm_pct[i,j]:.1f}%' for j in range(cm_pct.shape[1])] for i in range(cm_pct.shape[0])])
    
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 用于显示中文标签
    plt.rcParams['axes.unicode_minus'] = False
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title(f'Confusion Matrix (method={cm_method}, OOD-Assisted, %)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    # ====== 矩阵化可视化：故障类型 × 子模型 ======
    has_ood_row = np.any(all_labels >= K)
    row_labels = [f'Class {i}' for i in range(K)] + (['OOD'] if has_ood_row else [])

    # 1) 正类预测比例矩阵（已统一到通用热力图风格）
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
        title='Per-model Positive Prediction Ratio (%)',
        cbar_label='Ratio (%)',
        filename='per_model_class_preference_matrix.png',
        output_dir=output_dir,
        fmt='.1f',
        cmap='Blues',
        annotate=True,
        value_scale=100.0,
        vmin=0.0,
        vmax=100.0,
    )

    # 2) 各方法不确定度矩阵（类 × 模型）
    per_method_matrices = {
        'edl_mean': u_k_matrix,
        'edl_positive_only': K / np.maximum(alpha_pos_matrix, 1e-6),
        'edl_dynamic': 2.0 / (alpha_pos_matrix + 1.0),
        'max_prob': 1.0 - p_yes_matrix,
        'all_rest': (p_yes_matrix <= 0.5).astype(np.float32),
    }
    method_titles = {
        'edl_mean': 'Class-Model Mean Uncertainty (edl_mean)',
        'edl_positive_only': 'Class-Model Mean Uncertainty (positive_only)',
        'edl_dynamic': 'Class-Model Mean Uncertainty (dynamic)',
        'max_prob': 'Class-Model Mean Uncertainty (max_prob)',
        'all_rest': 'Class-Model Rest Prediction Ratio',
    }
    method_cbar = {
        'edl_mean': 'mean uncertainty',
        'edl_positive_only': 'mean uncertainty',
        'edl_dynamic': 'mean uncertainty',
        'max_prob': 'mean uncertainty',
        'all_rest': 'ratio',
    }

    # 同类指标尽量共用色条范围，便于横向比较
    unc_stack_for_range = np.concatenate(
        [
            per_method_matrices['edl_mean'].reshape(-1),
            per_method_matrices['edl_positive_only'].reshape(-1),
            per_method_matrices['edl_dynamic'].reshape(-1),
            per_method_matrices['max_prob'].reshape(-1),
        ],
        axis=0,
    )
    unc_vmin = float(np.percentile(unc_stack_for_range, 5))
    unc_vmax = float(np.percentile(unc_stack_for_range, 95))
    if unc_vmax <= unc_vmin:
        unc_vmin, unc_vmax = None, None

    for m_name, mat in per_method_matrices.items():
        stat_mat, stat_rows = compute_class_model_stats(mat, all_labels, K, has_ood_row, reduce='mean')
        if m_name == 'all_rest':
            vmin, vmax, value_scale, fmt = 0.0, 1.0, 100.0, '.1f'
            cbar_label = 'rest ratio (%)'
        else:
            vmin, vmax, value_scale, fmt = unc_vmin, unc_vmax, 1.0, '.3f'
            cbar_label = method_cbar[m_name]
        plot_class_model_heatmap(
            stat_mat,
            stat_rows,
            K,
            title=method_titles[m_name],
            cbar_label=cbar_label,
            filename=f'per_model_uncertainty_{m_name}.png',
            output_dir=output_dir,
            fmt=fmt,
            cmap='mako',
            annotate=True,
            value_scale=value_scale,
            vmin=vmin,
            vmax=vmax,
        )

    # 兼容旧文件名：额外输出 per_model_uncertainty_uk.png（等价于 edl_mean 的类-模型均值矩阵）
    uk_stats, uk_rows = compute_class_model_stats(u_k_matrix, all_labels, K, has_ood_row, reduce='mean')
    plot_class_model_heatmap(
        uk_stats,
        uk_rows,
        K,
        title='Class-Model Mean EDL Uncertainty u_k',
        cbar_label='mean u_k',
        filename='per_model_uncertainty_uk.png',
        output_dir=output_dir,
        fmt='.3f',
        cmap='mako',
        annotate=True,
        vmin=unc_vmin,
        vmax=unc_vmax,
    )

    # 3) logits 正/负类矩阵（类 × 模型）
    logits_pos_stats, logits_rows = compute_class_model_stats(
        logits_pos_matrix, all_labels, K, has_ood_row, reduce='mean'
    )
    logits_neg_stats, _ = compute_class_model_stats(
        logits_neg_matrix, all_labels, K, has_ood_row, reduce='mean'
    )
    logits_range = np.concatenate([logits_pos_stats.reshape(-1), logits_neg_stats.reshape(-1)], axis=0)
    logits_vmin = float(np.percentile(logits_range, 5))
    logits_vmax = float(np.percentile(logits_range, 95))
    if logits_vmax <= logits_vmin:
        logits_vmin, logits_vmax = None, None

    plot_class_model_heatmap(
        logits_pos_stats,
        logits_rows,
        K,
        title='Class-Model Mean Positive Logits',
        cbar_label='mean logits_pos',
        filename='per_model_logits_pos.png',
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
        title='Class-Model Mean Negative Logits',
        cbar_label='mean logits_neg',
        filename='per_model_logits_neg.png',
        output_dir=output_dir,
        fmt='.2f',
        cmap='rocket',
        annotate=True,
        vmin=logits_vmin,
        vmax=logits_vmax,
    )

    # 4) 可选：保留详细叠加分布图（默认关闭）
    class_list = list(range(K)) + ([K] if has_unknown_samples else [])
    class_names = [f'Class {i}' for i in range(K)] + (['OOD'] if has_unknown_samples else [])
    cmap = plt.get_cmap('tab10')
    class_colors = [cmap(i % 10) for i in range(len(class_list))]

    if args.plot_detailed_distributions:
        if u_k_matrix.size > 0:
            uk_min = float(np.percentile(u_k_matrix, 1))
            uk_max = float(np.percentile(u_k_matrix, 99))
        else:
            uk_min, uk_max = 0.0, 1.0

        plot_per_model_overlaid_histograms(
            u_k_matrix,
            all_labels,
            class_list,
            class_colors,
            class_names,
            K,
            title='Per-model EDL Uncertainty u_k (All Classes Overlapped)',
            filename='per_model_uncertainty_uk_detailed.png',
            x_label='u_k',
            output_dir=output_dir,
            bins=30,
            x_range=(uk_min, uk_max),
        )
        plot_per_model_overlaid_histograms(
            p_yes_matrix,
            all_labels,
            class_list,
            class_colors,
            class_names,
            K,
            title='Per-model Positive Probability P_yes (All Classes Overlapped)',
            filename='per_model_prob_pyes_detailed.png',
            x_label='P_yes',
            output_dir=output_dir,
            bins=30,
            x_range=(0.0, 1.0),
        )
        plot_per_model_overlaid_histograms(
            logits_pos_matrix,
            all_labels,
            class_list,
            class_colors,
            class_names,
            K,
            title='Per-model Positive Logits (All Classes Overlapped)',
            filename='per_model_logits_pos_detailed.png',
            x_label='logits_pos',
            output_dir=output_dir,
            bins=30,
            x_range=None,
        )
        plot_per_model_overlaid_histograms(
            logits_neg_matrix,
            all_labels,
            class_list,
            class_colors,
            class_names,
            K,
            title='Per-model Negative Logits (All Classes Overlapped)',
            filename='per_model_logits_neg_detailed.png',
            x_label='logits_neg',
            output_dir=output_dir,
            bins=30,
            x_range=None,
        )

    results_path = os.path.join(output_dir, 'binary_test_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Closed-set Accuracy (Ensemble, edl_mean): {test_results['accuracy']:.2f}%\n")
        f.write("\n--- 各模型单独（二分类，仅已知类）---\n")
        for k in range(K):
            f.write(f"Model {k} (Class {k} [{known_classes[k]}] vs Rest): {binary_accuracies[k] * 100:.2f}%\n")
        f.write(f"Average Binary Accuracy: {np.mean(binary_accuracies) * 100:.2f}%\n")
        if has_unknown_samples:
            f.write("\n--- OOD Metrics (per method) ---\n")
            for method, m in ood_metrics.items():
                name = method_pretty[method]
                f.write(f"[{name}] F1: {m['f1']:.4f}, AUROC: {m['auroc']:.4f}, FAR: {m['far']:.2f}%, MAR: {m['mar']:.2f}%\n")
            
    print(f"\n测试完成。结果已保存至 {output_dir}")

if __name__ == '__main__':
    main()