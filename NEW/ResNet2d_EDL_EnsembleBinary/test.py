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
import seaborn as sns

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from common.utils.helpers import load_config
from common.utils.data_loader import NpyPackDataset
from common.edl_losses import relu_evidence
from models import get_model

def get_dataset(data_config, split='test', filter_classes=None):
    """根据配置动态加载数据集，filter_classes=None 时加载全量（已知+未知）。"""
    data_dir = data_config.get('data_dir')
    if data_dir is None:
        raise ValueError("配置文件中必须指定 data.data_dir")
        
    openset_config = data_config.get('openset', {})
    known_classes = openset_config.get('known_classes')
    unknown_classes = openset_config.get('unknown_classes', [])
    
    return NpyPackDataset(
        data_dir=data_dir,
        split=split,
        filter_classes=filter_classes,
        known_classes=known_classes,
        unknown_classes=unknown_classes
    )

def compute_uncertainty_threshold_iqr(val_loader, models, device, K):
    for m in models:
        m.eval()
    uncertainties_list = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            u_batch = []
            for k in range(K):
                logits_k = models[k](inputs)
                evidence_k = relu_evidence(logits_k)
                alpha_k = evidence_k + 1
                S_k = torch.sum(alpha_k, dim=1)
                u_k = (2.0 / S_k).cpu().numpy()
                u_batch.append(u_k)
            u_mean = np.mean(np.stack(u_batch, axis=0), axis=0)
            uncertainties_list.append(u_mean)
    all_u = np.concatenate(uncertainties_list)
    q1, q3 = np.percentile(all_u, [25, 75])
    iqr = q3 - q1
    return float(q3 + 1.5 * iqr)

def main():
    parser = argparse.ArgumentParser(description='EDL 集成二分类：测试 K 个模型，EDL 不确定度，OOD 指标')
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='配置文件路径（默认: config.yaml）'
    )
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint 目录（含 model_0.pth..model_{K-1}.pth）；未指定则从 config 的 checkpoint_dir 读取')
    parser.add_argument('--uncertainty_threshold', '--uncertainty', type=float, default=0.5, dest='uncertainty_threshold')
    parser.add_argument('--threshold_from_val', action='store_true',
                        help='若指定，则从验证集不确定性 IQR 自动计算阈值，忽略 --uncertainty_threshold')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录；未指定则为 checkpoint_dir/test')
    parser.add_argument('--agg', type=str, default='dynamic', choices=['dynamic', 'mean'],
                        help='不确定度聚合方式：dynamic=按预测类别动态选择，mean=所有模型均值')
    parser.add_argument('--mode', type=str, default='edl', choices=['edl', 'max_prob'],
                        help="不确定性: 'edl' (u_k=2/S_k 聚合) 或 'max_prob' (1 - max P_yes)")
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config['data'] 
    model_config = config['model']
    train_config = config['train']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

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
    models = []
    for k in range(K):
        path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
        if not os.path.exists(path_k):
            raise FileNotFoundError(f"未找到 {path_k}，请先运行 train.py")
        model_k = get_model(backbone_type, num_classes=2).to(device)
        ckpt = torch.load(path_k, map_location=device, weights_only=True)
        state = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
        model_k.load_state_dict(state, strict=True)
        model_k.eval()
        models.append(model_k)
    print(f"已加载 K={K} 个模型，backbone={backbone_type}")

    # 读取测试集全量
    test_dataset = get_dataset(data_config, split='test', filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
    
    # 类别映射和数字标签有关
    # 全局映射保存在 test_dataset.global_label_map
    # OOD 测试时，我们将不在 known_classes 范围的数值视作未知。
    # 由于我们在 data_loader 中保证已知类的 id 在 0..K-1，所以 >= K 即为 OOD
    has_unknown_samples = any([v >= K for v in test_dataset.y])
    print(f"测试集大小: {len(test_dataset)}")

    if args.threshold_from_val:
        # 使用已知类的测试数据作为 val，用于定阈值
        val_dataset = get_dataset(data_config, split='test', filter_classes=known_classes)
        val_loader = DataLoader(val_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
        uncertainty_threshold = compute_uncertainty_threshold_iqr(val_loader, models, device, K)
        print(f"由已知类测试子集 IQR 得到不确定性阈值: {uncertainty_threshold:.4f}")
    else:
        uncertainty_threshold = args.uncertainty_threshold
        print(f"不确定性阈值 (u > 此值判 OOD): {uncertainty_threshold}")

    all_labels = []
    all_preds = []
    all_uncertainties = []
    binary_preds_list = []
    closed_set_true = []
    closed_set_pred = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            p_yes_list = []
            u_k_list = []
            for k in range(K):
                logits_k = models[k](inputs)
                evidence_k = relu_evidence(logits_k)
                alpha_k = evidence_k + 1
                S_k = torch.sum(alpha_k, dim=1)
                probs_k = alpha_k / S_k.unsqueeze(1)
                p_yes_list.append(probs_k[:, 1])
                u_k_list.append(2.0 / S_k)
                    
            # 概率重构
            p_normal = p_yes_list[0] # 模型0的正类概率是正常概率
            p_fault_total = 1.0 - p_normal
            
            final_probs_list = [p_normal]
            for k in range(1, K):
                # 模型k的输出是在故障前提下，属于故障k的概率
                # 最终概率 = 总故障概率 * 模型k的概率
                p_k_final = p_fault_total * p_yes_list[k]
                final_probs_list.append(p_k_final)
                
            final_probs = torch.stack(final_probs_list, dim=1)
            
            # 记录二分类预测结果用于单独评估
            binary_preds = []
            binary_preds.append((p_yes_list[0] > 0.5).cpu().numpy().astype(np.int32))
            for k in range(1, K):
                binary_preds.append((p_yes_list[k] > 0.5).cpu().numpy().astype(np.int32))
            binary_preds_list.append(np.stack(binary_preds, axis=1))
            
            preds_local = final_probs.argmax(dim=1).cpu().numpy()

            # 闭集预测（仅已知类，忽略 OOD 判别）
            for i in range(len(labels)):
                if int(labels[i]) < K:
                    closed_set_true.append(int(labels[i]))
                    closed_set_pred.append(int(preds_local[i]))
            
            u_k_stack = torch.stack(u_k_list, dim=1)
            if args.agg == 'mean':
                uncertainty = u_k_stack.mean(dim=1).cpu().numpy()
            elif args.agg == 'dynamic':
                # 动态选择：如果预测为正常(0)，取 u_0
                # 如果预测为故障k(k>0)，取 u_k
                winner = final_probs.argmax(dim=1)
                uncertainty = u_k_stack[torch.arange(u_k_stack.size(0), device=u_k_stack.device), winner].cpu().numpy()
            else:
                uncertainty = u_k_stack.mean(dim=1).cpu().numpy() # fallback

            for i in range(len(labels)):
                u_val = float(uncertainty[i])
                all_uncertainties.append(u_val)
                if u_val > uncertainty_threshold:
                    all_preds.append(-1)
                else:
                    pl = int(preds_local[i])
                    if pl < K:
                        all_preds.append(pl)
                    else:
                        all_preds.append(-1)
            all_labels.extend(labels.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_uncertainties = np.array(all_uncertainties)
    binary_preds_matrix = np.concatenate(binary_preds_list, axis=0)  # [N, K]
    closed_set_true = np.array(closed_set_true)
    closed_set_pred = np.array(closed_set_pred)

    # 已知类的掩码
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

    print(f"\n--- 各模型单独（二分类，仅已知类）---")
    known_idx = np.where(known_mask)[0]
    binary_accuracies = []
    binary_conf_info = []
    
    # 模型 0 的评估 (正常 vs 所有故障)
    true_0 = (all_labels[known_idx] == 0).astype(np.int32)
    pred_0 = binary_preds_matrix[known_idx, 0]
    acc_0 = accuracy_score(true_0, pred_0)
    binary_accuracies.append(acc_0)
    title_0 = "Model 0 (Normal vs All Faults)"
    binary_conf_info.append({
        'model_idx': 0,
        'true': true_0,
        'pred': pred_0,
        'title': title_0
    })
    print(f"  {title_0}: 二分类准确率 = {acc_0 * 100:.2f}%")
    
    # 模型 k 的评估 (正常 vs 故障 k)
    for k in range(1, K):
        # 仅选取类别为 0 或 k 的样本进行评估
        mask_k = (all_labels[known_idx] == 0) | (all_labels[known_idx] == k)
        idx_k = known_idx[mask_k]
        
        if len(idx_k) > 0:
            true_k = (all_labels[idx_k] == k).astype(np.int32)
            pred_k = binary_preds_matrix[idx_k, k]
            acc_k = accuracy_score(true_k, pred_k)
            binary_accuracies.append(acc_k)
            title_k = f"Model {k} (Normal vs Fault {k} [{known_classes[k]}])"
            binary_conf_info.append({
                'model_idx': k,
                'true': true_k,
                'pred': pred_k,
                'title': title_k
            })
            print(f"  {title_k}: 二分类准确率 = {acc_k * 100:.2f}%")
        else:
            binary_accuracies.append(0.0)
            print(f"  Model {k}: 无评估样本")
            
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
        true_is_unknown = (~known_mask).astype(int)
        pred_is_unknown = (all_preds == -1).astype(int)
        f1 = f1_score(true_is_unknown, pred_is_unknown)
        test_results['f1_score'] = f1
        print(f"F1-Score (检测未知类): {f1:.4f}")
        auroc = roc_auc_score(true_is_unknown, all_uncertainties)
        test_results['auroc'] = auroc
        print(f"AUROC (区分已知/未知): {auroc:.4f}")
        
        tp = np.sum((true_is_unknown == 1) & (pred_is_unknown == 1))
        fp = np.sum((true_is_unknown == 0) & (pred_is_unknown == 1))
        tn = np.sum((true_is_unknown == 0) & (pred_is_unknown == 0))
        fn = np.sum((true_is_unknown == 1) & (pred_is_unknown == 0))
        far = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
        mar = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        test_results['far'] = far
        test_results['mar'] = mar
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
        plt.title('ROC Curve (EDL Ensemble Binary)')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
        plt.close()
        
        # 不确定度分布图：ID vs OOD
        plt.figure(figsize=(8, 6))
        plt.hist(id_unc, bins=30, alpha=0.6, label='ID', density=True)
        plt.hist(ood_unc, bins=30, alpha=0.6, label='OOD', density=True)
        plt.xlabel('Uncertainty')
        plt.ylabel('Density')
        plt.xlim(0.0, 1.0)
        plt.title(f'Uncertainty Distribution (AUROC={auroc:.4f})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'uncertainty_distribution.png'), dpi=150)
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

    # 混淆矩阵（已知类 + Unknown，全数据集，含 OOD）
    label_names = [str(c) for c in known_classes] + ['Unknown']
    cm_labels = []
    cm_preds = []
    
    for true_label, pred_label in zip(all_labels, all_preds):
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
    plt.title('Confusion Matrix (EDL Ensemble Binary, OOD-Assisted, %)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()

    results_path = os.path.join(output_dir, 'binary_test_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Uncertainty Threshold: {uncertainty_threshold}\n")
        f.write(f"Closed-set Accuracy (Ensemble): {test_results['accuracy']:.2f}%\n")
        f.write("\n--- 各模型单独（二分类，仅已知类）---\n")
        f.write(f"Model 0 (Normal vs All Faults): {binary_accuracies[0] * 100:.2f}%\n")
        for k in range(1, K):
            f.write(f"Model {k} (Normal vs Fault {k} [{known_classes[k]}]): {binary_accuracies[k] * 100:.2f}%\n")
        f.write(f"Average Binary Accuracy: {np.mean(binary_accuracies) * 100:.2f}%\n")
        if test_results['has_unknown']:
            f.write(f"\nF1 (OOD): {test_results['f1_score']:.4f}\n")
            f.write(f"AUROC: {test_results['auroc']:.4f}\n")
            f.write(f"FAR: {test_results['far']:.2f}%\n")
            f.write(f"MAR: {test_results['mar']:.2f}%\n")
            
    print(f"\n测试完成。结果已保存至 {output_dir}")

if __name__ == '__main__':
    main()
