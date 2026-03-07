"""
EDL + K 个独立二分类小模型的测试与 OOD 评估脚本。

加载 model_0.pth .. model_{K-1}.pth，对每个样本得到 K 路「Yes」概率与 EDL 不确定度 u_k=2/S_k；
预测为 argmax p_yes，不确定度采用 mean=(1/K)*sum(u_k) 或 winner=u_{argmax}；
按不确定度阈值判 OOD，输出闭集准确率、F1、AUROC、FAR、MAR、ROC 曲线、混淆矩阵并写入 experiment_info.txt。
与 test_edl_binary.py 区别：本脚本加载 K 个独立小模型，不确定度为 EDL 定义 2/S；后者为单模型多二分类头。
"""
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录到 sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config
from common.utils.data_loader import RawSignalDataset, NpyIndexDataset, LabeledImageDataset
from common.edl_losses import relu_evidence
from models import get_model


def get_dataset(data_config, split='test', filter_classes=None):
    """根据配置动态加载数据集，与 test_edl_binary 一致；filter_classes=None 时加载全量（已知+未知）。"""
    data_type = data_config.get('type')
    if data_type is None:
        raise ValueError("配置文件中必须指定 data.type")
    if data_type == 'raw_signal':
        base_dir = data_config.get('raw_signal_output_dir')
        split_dir = os.path.join(base_dir, split)
        return RawSignalDataset(split_dir=split_dir, filter_classes=filter_classes)
    elif data_type == 'wpt':
        base_dir = data_config.get('wpt_output_dir')
        split_dir = os.path.join(base_dir, split)
        return NpyIndexDataset(split_dir=split_dir, filter_classes=filter_classes)
    elif data_type in ('image', 'stft'):
        base_dir = data_config.get('img_output_dir') or data_config.get('stft_output_dir')
        if not base_dir:
            raise ValueError("image/stft 数据需在配置中指定 data.img_output_dir")
        path = os.path.join(base_dir, split)
        transform = transforms.Compose([transforms.ToTensor()])
        return LabeledImageDataset(path=path, transform=transform, filter_classes=filter_classes)
    else:
        raise ValueError(f"test_edl_ensemble_binary 支持 'raw_signal'、'wpt'、'image'/'stft'，当前: {data_type}")


def compute_uncertainty_threshold_iqr(val_loader, models, device, K):
    """从验证集（仅已知类）不确定性分布用 IQR 规则计算阈值：Upper bound = Q3 + 1.5*IQR。
    每个样本的不确定度为 K 个模型 EDL 不确定度 u_k=2/S_k 的均值。"""
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
    parser.add_argument('--config', type=str, required=True, help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='checkpoint 目录（含 model_0.pth..model_{K-1}.pth）；未指定则从 config 的 checkpoint_dir 读取')
    parser.add_argument('--uncertainty_threshold', '--uncertainty', type=float, default=0.5, dest='uncertainty_threshold',
                        help='不确定性阈值，超过则判为 OOD（默认 0.5）；可简写为 --uncertainty')
    parser.add_argument('--threshold_from_val', action='store_true',
                        help='若指定，则从验证集不确定性 IQR 自动计算阈值，忽略 --uncertainty_threshold')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录；未指定则为 checkpoint_dir/test')
    parser.add_argument('--uncertainty_agg', type=str, default='mean', choices=['mean', 'winner'],
                        help='不确定度聚合方式：mean=(1/K)*sum(u_k)，winner=取赢家头 u_{argmax p_yes}')
    parser.add_argument('--uncertainty_mode', type=str, default='edl', choices=['edl', 'max_prob'],
                        help="不确定性: 'edl' (u_k=2/S_k 聚合) 或 'max_prob' (1 - max P_yes)")
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 确定 checkpoint 目录与输出目录
    if args.checkpoint:
        checkpoint_dir = os.path.abspath(args.checkpoint)
    else:
        checkpoint_dir = train_config['checkpoint_dir']
    if not os.path.isdir(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    output_dir = args.output_dir or os.path.join(checkpoint_dir, 'test')
    os.makedirs(output_dir, exist_ok=True)

    K = model_config.get('num_classes')
    if K is None or K < 1:
        raise ValueError("config model.num_classes 必须为 K")
    # 每个小模型为 ResNet18_2d(num_classes=2)，若配置为 Binary 则强制使用 ResNet18_2d
    backbone_type = model_config.get('type', 'ResNet18_2d')
    if 'Binary' in str(backbone_type):
        backbone_type = 'ResNet18_2d'

    models = []
    for k in range(K):
        path_k = os.path.join(checkpoint_dir, f'model_{k}.pth')
        if not os.path.exists(path_k):
            raise FileNotFoundError(f"未找到 {path_k}，请先运行 train_edl_ensemble_binary.py")
        model_k = get_model(backbone_type, num_classes=2).to(device)
        ckpt = torch.load(path_k, map_location=device, weights_only=True)
        state = ckpt.get('state_dict') or ckpt.get('model_state_dict') or ckpt
        model_k.load_state_dict(state, strict=True)
        model_k.eval()
        models.append(model_k)
    print(f"已加载 K={K} 个模型，backbone={backbone_type}")

    # 3. 测试数据：全量测试集（已知+未知），并获取训练时标签映射
    known_classes = data_config['openset']['known_classes']
    # 训练时数据集用 sorted(known_classes) 做 0..K-1 重映射，预测下标 pl 对应第 pl 个「排序后的已知类」
    sorted_known_classes = sorted(known_classes)
    unknown_classes = data_config['openset'].get('unknown_classes', [])
    num_known_classes = len(known_classes)

    temp_dataset = get_dataset(data_config, split='test', filter_classes=known_classes)
    train_label_map = getattr(temp_dataset, 'label_map', None) or (temp_dataset.label_map if hasattr(temp_dataset, 'label_map') else None)
    test_dataset = get_dataset(data_config, split='test', filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)
    print(f"测试集大小: {len(test_dataset)}")
    if train_label_map:
        print(f"训练时标签映射: {train_label_map}")

    # 4. OOD 阈值：u > 阈值判 OOD；阈值可手动指定或由验证集 IQR 计算
    if args.threshold_from_val:
        val_dataset = get_dataset(data_config, split='val', filter_classes=known_classes)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
        uncertainty_threshold = compute_uncertainty_threshold_iqr(val_loader, models, device, K)
        print(f"由验证集 IQR 得到不确定性阈值: {uncertainty_threshold:.4f}")
    else:
        uncertainty_threshold = args.uncertainty_threshold
        print(f"不确定性阈值 (u > 此值判 OOD): {uncertainty_threshold}")

    # 5. 前向：全量测试集，K 路 evidence -> p_yes 与 u_k=2/S_k，再按阈值判 OOD；并收集每模型二分类预测（供逐模型评估）
    all_labels = []
    all_preds = []
    all_uncertainties = []
    binary_preds_list = []  # 每 batch 的 [B, K]，模型 k 预测「是否第 k 类」1/0

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
                if args.uncertainty_mode == 'edl':
                    u_k_list.append(2.0 / S_k)
            p_yes = torch.stack(p_yes_list, dim=1)
            binary_preds_list.append((p_yes > 0.5).cpu().numpy().astype(np.int32))
            if args.uncertainty_mode == 'max_prob':
                uncertainty = (1.0 - torch.max(p_yes, dim=1)[0]).cpu().numpy()
            else:
                u_k_stack = torch.stack(u_k_list, dim=1)
                if args.uncertainty_agg == 'mean':
                    uncertainty = u_k_stack.mean(dim=1).cpu().numpy()
                else:
                    winner = p_yes.argmax(dim=1)
                    uncertainty = u_k_stack[torch.arange(u_k_stack.size(0), device=u_k_stack.device), winner].cpu().numpy()
            preds_local = p_yes.argmax(dim=1).cpu().numpy()

            # 按不确定性阈值判 OOD：u > 阈值 → 预测为 -1，否则映射回类别
            for i in range(len(labels)):
                u_val = float(uncertainty[i])
                all_uncertainties.append(u_val)
                if u_val > uncertainty_threshold:
                    all_preds.append(-1)
                else:
                    pl = int(preds_local[i])
                    if pl < len(sorted_known_classes):
                        all_preds.append(sorted_known_classes[pl])
                    else:
                        all_preds.append(-1)
            all_labels.extend(labels.cpu().numpy().tolist())

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_uncertainties = np.array(all_uncertainties)
    binary_preds_matrix = np.concatenate(binary_preds_list, axis=0)  # [N, K]
    has_unknown_samples = len(unknown_classes) > 0 and np.any(np.isin(all_labels, unknown_classes))

    # 6. 闭集准确率（仅已知类样本，预测 -1 视为错误）
    label_map = train_label_map
    if label_map is None:
        sorted_known = sorted(known_classes)
        label_map = {orig: idx for idx, orig in enumerate(sorted_known)}

    known_mask = np.isin(all_labels, known_classes)
    mapped_known_labels = np.array([label_map.get(int(l), -1) for l in all_labels[known_mask]])
    known_preds_orig = all_preds[known_mask]
    known_preds_mapped = np.array([label_map.get(int(p), -2) if p != -1 else -1 for p in known_preds_orig])
    valid_mask = mapped_known_labels >= 0
    mapped_known_labels = mapped_known_labels[valid_mask]
    known_preds_mapped = known_preds_mapped[valid_mask]
    if len(mapped_known_labels) == 0:
        print("错误: 没有有效的已知类样本")
        return

    accuracy = accuracy_score(mapped_known_labels, known_preds_mapped)

    # 6.1 各模型单独评估（二分类准确率，仅已知类子集，与论文「每个 DBL 单独测」一致）
    print(f"\n--- 各模型单独（二分类，仅已知类）---")
    known_idx = np.where(known_mask)[0]
    binary_accuracies = []
    for k in range(K):
        true_k = (all_labels[known_idx] == sorted_known_classes[k]).astype(np.int32)
        pred_k = binary_preds_matrix[known_idx, k]
        acc_k = accuracy_score(true_k, pred_k)
        binary_accuracies.append(acc_k)
        print(f"  Model {k} (Class {sorted_known_classes[k]}): 二分类准确率 = {acc_k * 100:.2f}%")
    print(f"  平均二分类准确率: {np.mean(binary_accuracies) * 100:.2f}%")

    # 6.2 集成整体
    print(f"\n--- 集成整体 ---")
    print(f"已知类准确率 (Closed-set Accuracy): {accuracy * 100:.2f}%")

    # 7. OOD 检测指标：F1、AUROC、FAR、MAR
    test_results = {
        'accuracy': accuracy * 100,
        'f1_score': None,
        'auroc': None,
        'far': None,
        'mar': None,
        'has_unknown': has_unknown_samples,
    }

    if has_unknown_samples:
        unknown_mask = np.isin(all_labels, unknown_classes)
        pred_is_unknown = (all_preds == -1).astype(int)
        true_is_unknown = unknown_mask.astype(int)
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
        print(f"ROC 曲线已保存: {os.path.join(output_dir, 'roc_curve.png')}")
    else:
        print("测试集中无未知类，跳过开放集指标与 ROC。")

    # 8. 混淆矩阵（已知类 + Unknown）
    label_names = [f'Class_{c}' for c in known_classes] + ['Unknown']
    cm_labels = []
    cm_preds = []
    for true_label, pred_label in zip(all_labels, all_preds):
        true_label, pred_label = int(true_label), int(pred_label)
        if true_label in known_classes:
            cm_labels.append(known_classes.index(true_label))
        else:
            cm_labels.append(num_known_classes)
        if pred_label == -1:
            cm_preds.append(num_known_classes)
        elif pred_label in known_classes:
            cm_preds.append(known_classes.index(pred_label))
        else:
            cm_preds.append(num_known_classes)
    cm_labels = np.array(cm_labels)
    cm_preds = np.array(cm_preds)
    cm = confusion_matrix(cm_labels, cm_preds)
    # 按行归一化为百分比（每行真实类别下预测分布）
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sum > 0, cm.astype(float) / row_sum * 100, 0)
    annot = np.array([[f'{cm_pct[i,j]:.1f}%' for j in range(cm_pct.shape[1])] for i in range(cm_pct.shape[0])])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix (EDL Ensemble Binary, OOD-Assisted, %)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"混淆矩阵已保存: {os.path.join(output_dir, 'confusion_matrix.png')}")

    # 7.1 各模型单独的二分类混淆矩阵（仅已知类），与论文「每模型都测」一致
    ncols = min(3, K)
    nrows = (K + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
    if K == 1:
        axes = np.array([[axes]])
    else:
        axes = axes.flatten() if K > 1 else [axes]
    for k in range(K):
        ax = axes[k]
        true_k = (all_labels[known_idx] == sorted_known_classes[k]).astype(np.int32)
        pred_k = binary_preds_matrix[known_idx, k]
        cm_k = confusion_matrix(true_k, pred_k, labels=[0, 1])
        row_sum_k = cm_k.sum(axis=1, keepdims=True)
        cm_k_pct = np.where(row_sum_k > 0, cm_k.astype(float) / row_sum_k * 100, 0)
        annot_k = np.array([[f'{cm_k_pct[i,j]:.1f}%' for j in range(2)] for i in range(2)])
        tick_labels = [f'Not {sorted_known_classes[k]}', f'Class {sorted_known_classes[k]}']
        sns.heatmap(cm_k_pct, annot=annot_k, fmt='', cmap='Blues', ax=ax, xticklabels=tick_labels, yticklabels=tick_labels)
        ax.set_title(f'Model {k} (Class {sorted_known_classes[k]})')
        ax.set_ylabel('True')
        ax.set_xlabel('Pred')
    for idx in range(K, len(axes)):
        axes[idx].set_visible(False)
    plt.suptitle('Per-Model Binary Confusion Matrices (Known Classes Only)', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix_per_model.png'), dpi=150)
    plt.close()
    print(f"各模型二分类混淆矩阵已保存: {os.path.join(output_dir, 'confusion_matrix_per_model.png')}")

    # 8.1 写入各模型 + 集成结果到 binary_test_results.txt（与论文「每模型 + 整体」一致）
    results_path = os.path.join(output_dir, 'binary_test_results.txt')
    with open(results_path, 'w', encoding='utf-8') as f:
        f.write(f"Uncertainty Threshold: {uncertainty_threshold}\n")
        f.write(f"Closed-set Accuracy (Ensemble): {test_results['accuracy']:.2f}%\n")
        f.write("\n--- 各模型单独（二分类，仅已知类）---\n")
        for k in range(K):
            f.write(f"Model {k} (Class {sorted_known_classes[k]}): {binary_accuracies[k] * 100:.2f}%\n")
        f.write(f"Average Binary Accuracy: {np.mean(binary_accuracies) * 100:.2f}%\n")
        if test_results['has_unknown']:
            f.write(f"\nF1 (OOD): {test_results['f1_score']:.4f}\n")
            f.write(f"AUROC: {test_results['auroc']:.4f}\n")
            f.write(f"FAR: {test_results['far']:.2f}%\n")
            f.write(f"MAR: {test_results['mar']:.2f}%\n")
    print(f"各模型与集成结果已保存: {results_path}")

    # 9. 保存不确定性 CSV，供 analyze_uncertainty.py --load_csv 绘制 ID/OOD 分布图
    import pandas as pd
    results_df = pd.DataFrame({
        'true_label': all_labels,
        'predicted_label': all_preds,
        'uncertainty': all_uncertainties,
    })
    csv_path = os.path.join(output_dir, 'uncertainty_analysis.csv')
    results_df.to_csv(csv_path, index=False)
    print(f"不确定性结果已保存: {csv_path}（可用 analyze_uncertainty.py --load_csv 做 KDE 图）")

    # 10. 将评估结果追加到 experiment_info.txt
    experiment_info_path = os.path.join(checkpoint_dir, 'experiment_info.txt')
    if os.path.exists(experiment_info_path):
        with open(experiment_info_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        if '【测试评估结果】' in existing_content:
            lines = existing_content.split('\n')
            new_lines = []
            skip = False
            for line in lines:
                if '【测试评估结果】' in line:
                    skip = True
                elif skip and line.startswith('=') and len(line) >= 80:
                    skip = False
                    new_lines.append(line)
                elif not skip:
                    new_lines.append(line)
            existing_content = '\n'.join(new_lines)
        with open(experiment_info_path, 'w', encoding='utf-8') as f:
            f.write(existing_content.rstrip() + '\n\n')
            f.write("【测试评估结果】\n")
            f.write("-" * 80 + "\n")
            f.write(f"测试集大小: {len(test_dataset):,}\n")
            f.write(f"不确定性阈值 (u > 判 OOD): {uncertainty_threshold}\n")
            f.write(f"已知类准确率 (Closed-set Accuracy): {test_results['accuracy']:.2f}%\n")
            f.write("各模型二分类准确率 (仅已知类): ")
            f.write(", ".join([f"Model{k}={binary_accuracies[k]*100:.1f}%" for k in range(K)]) + "\n")
            f.write(f"平均二分类准确率: {np.mean(binary_accuracies)*100:.2f}%\n")
            if test_results['has_unknown']:
                if test_results['f1_score'] is not None:
                    f.write(f"F1-Score (检测未知类): {test_results['f1_score']:.4f}\n")
                if test_results['auroc'] is not None:
                    f.write(f"AUROC (区分已知/未知): {test_results['auroc']:.4f}\n")
                if test_results.get('far') is not None:
                    f.write(f"FAR (虚警率): {test_results['far']:.2f}%\n")
                if test_results.get('mar') is not None:
                    f.write(f"MAR (漏警率): {test_results['mar']:.2f}%\n")
            else:
                f.write("测试集中未发现未知类样本，未计算开放集指标。\n")
            f.write("\n")
            f.write("=" * 80 + "\n")
        print(f"评估结果已追加至: {experiment_info_path}")
    else:
        print(f"警告: 未找到 {experiment_info_path}，跳过写入。")


if __name__ == '__main__':
    main()
