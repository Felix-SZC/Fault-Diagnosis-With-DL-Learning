"""
OOD 检测辅助的可信故障诊断评估脚本（多二分类头 EDL）。
参照论文 "Out-of-distribution detection-assisted trustworthy machinery fault diagnosis"
与 test_edl.py：在已知+未知混合测试集上按不确定性阈值判 OOD，输出闭集准确率、
F1/AUROC/FAR/MAR、ROC 曲线、混淆矩阵并写入 experiment_info.txt。
"""
import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config
from common.utils.data_loader import RawSignalDataset, NpyIndexDataset
from common.edl_losses import relu_evidence
from models import get_model

def get_dataset(data_config, split='test', filter_classes=None):
    """根据配置动态加载数据集 (test_edl 版本)"""
    data_type = data_config.get('type')
    if data_type is None:
        raise ValueError("配置文件中必须指定 data.type")

    if data_type == 'raw_signal':
        base_dir = data_config.get('raw_signal_output_dir')
        if base_dir is None:
            raise ValueError("配置文件中缺少 'raw_signal_output_dir'")
        split_dir = os.path.join(base_dir, split)
        return RawSignalDataset(split_dir=split_dir, filter_classes=filter_classes)
    elif data_type == 'wpt':
        base_dir = data_config.get('wpt_output_dir')
        if base_dir is None:
            raise ValueError("配置文件中缺少 'wpt_output_dir'")
        split_dir = os.path.join(base_dir, split)
        return NpyIndexDataset(split_dir=split_dir, filter_classes=filter_classes)
    else:
        raise ValueError(f"test_edl_binary.py 目前仅支持 'raw_signal' 和 'wpt' 数据类型，当前为: {data_type}")


def compute_uncertainty_threshold_iqr(val_loader, model, device, num_classes, use_main_head):
    """从验证集（仅已知类）不确定性分布用 IQR 规则计算阈值：Upper bound = Q3 + 1.5*IQR"""
    model.eval()
    uncertainties_list = []
    with torch.no_grad():
        for inputs, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], list):
                main_logits, binary_logits_list = outputs[0], outputs[1]
            else:
                main_logits, binary_logits_list = None, outputs

            if main_logits is not None:
                evidence = relu_evidence(main_logits)
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1)
                u = (num_classes / S).cpu().numpy()
            else:
                yes_probs = []
                for b_logits in binary_logits_list:
                    evidence = relu_evidence(b_logits)
                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    yes_probs.append(probs[:, 1])
                yes_probs = torch.stack(yes_probs, dim=1)
                max_yes = torch.max(yes_probs, dim=1)[0]
                u = (1.0 - max_yes).cpu().numpy()
            uncertainties_list.append(u)
    all_u = np.concatenate(uncertainties_list)
    q1, q3 = np.percentile(all_u, [25, 75])
    iqr = q3 - q1
    threshold = float(q3 + 1.5 * iqr)
    return threshold


def main():
    parser = argparse.ArgumentParser(description='EDL Binary OOD-Assisted Evaluation Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='模型权重文件路径（.pth）。未指定则从 config 的 checkpoint_dir 读取')
    parser.add_argument('--uncertainty_threshold', type=float, default=0.5,
                        help='不确定性阈值，超过则判为 OOD (默认 0.5)')
    parser.add_argument('--threshold_from_val', action='store_true',
                        help='若指定，则从验证集不确定性 IQR 自动计算阈值，忽略 --uncertainty_threshold')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='结果输出目录。未指定则 checkpoint_dir/test')
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 模型路径与输出目录
    if args.checkpoint:
        model_path = args.checkpoint
        checkpoint_dir = os.path.dirname(os.path.abspath(model_path))
    else:
        checkpoint_dir = train_config['checkpoint_dir']
        model_path = os.path.join(checkpoint_dir, 'best_model.pth')

    if not os.path.exists(model_path):
        print(f"错误: 在 {model_path} 未找到模型文件")
        return

    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(checkpoint_dir, 'test')
    os.makedirs(output_dir, exist_ok=True)

    # 3. 加载模型
    known_classes = data_config['openset']['known_classes']
    unknown_classes = data_config['openset'].get('unknown_classes', [])
    num_classes = model_config.get('num_classes')
    use_main_head = model_config.get('use_main_head', True)

    model = get_model(
        model_config.get('type'),
        num_classes=num_classes,
        use_main_head=use_main_head
    ).to(device)

    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        state_dict = checkpoint.get('state_dict') or checkpoint.get('model_state_dict') or checkpoint
    else:
        state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"从 {model_path} 加载模型成功。")

    # 4. 测试数据：全量测试集（已知+未知），并获取训练时标签映射
    temp_dataset = get_dataset(data_config, split='test', filter_classes=known_classes)
    train_label_map = temp_dataset.label_map if hasattr(temp_dataset, 'label_map') and temp_dataset.label_map else None
    test_dataset = get_dataset(data_config, split='test', filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)
    print(f"测试集大小（已知+未知）: {len(test_dataset)}")
    if train_label_map is not None:
        print(f"训练时的标签映射: {train_label_map}")

    # 5. 可选：从验证集 IQR 计算不确定性阈值
    if args.threshold_from_val:
        val_dataset = get_dataset(data_config, split='val', filter_classes=known_classes)
        val_loader = DataLoader(val_dataset, batch_size=train_config['batch_size'], shuffle=False)
        uncertainty_threshold = compute_uncertainty_threshold_iqr(
            val_loader, model, device, num_classes, use_main_head
        )
        print(f"由验证集 IQR 得到不确定性阈值: {uncertainty_threshold:.4f}")
    else:
        uncertainty_threshold = args.uncertainty_threshold
        print(f"不确定性阈值: {uncertainty_threshold}")

    # 6. 前向：全量测试集，计算不确定性及类别预测，再按阈值判 OOD
    all_labels = []
    all_preds = []
    all_uncertainties = []
    binary_preds_matrix = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple) and len(outputs) == 2 and isinstance(outputs[1], list):
                main_logits, binary_logits_list = outputs[0], outputs[1]
            else:
                main_logits, binary_logits_list = None, outputs

            if main_logits is not None:
                evidence = relu_evidence(main_logits)
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                uncertainties = (num_classes / S).squeeze(1)
                preds_local = torch.argmax(probs, dim=1)
            else:
                yes_probs = []
                for b_logits in binary_logits_list:
                    evidence = relu_evidence(b_logits)
                    alpha = evidence + 1
                    S = torch.sum(alpha, dim=1, keepdim=True)
                    probs = alpha / S
                    yes_probs.append(probs[:, 1])
                yes_probs = torch.stack(yes_probs, dim=1)
                uncertainties = 1.0 - torch.max(yes_probs, dim=1)[0]
                preds_local = torch.argmax(yes_probs, dim=1)

            # 按阈值判 OOD，预测为 -1 或映射回原始类别 ID
            preds_local_np = preds_local.cpu().numpy()
            uncertainties_np = uncertainties.cpu().numpy()
            for i in range(len(labels)):
                u_val = float(uncertainties_np[i])
                all_uncertainties.append(u_val)
                if u_val > uncertainty_threshold:
                    all_preds.append(-1)
                else:
                    pl = int(preds_local_np[i])
                    if pl < len(known_classes):
                        all_preds.append(known_classes[pl])
                    else:
                        all_preds.append(-1)

            all_labels.extend(labels.numpy().tolist())

            # 二分类头硬预测（用于后续已知类子集上的逐头准确率）
            batch_binary = []
            for b_logits in binary_logits_list:
                evidence = relu_evidence(b_logits)
                alpha = evidence + 1
                S = torch.sum(alpha, dim=1, keepdim=True)
                probs = alpha / S
                _, b_pred = torch.max(probs, 1)
                batch_binary.append(b_pred.cpu().numpy())
            binary_preds_matrix.append(np.stack(batch_binary, axis=1))

    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    all_uncertainties = np.array(all_uncertainties)
    binary_preds_matrix = np.concatenate(binary_preds_matrix, axis=0)
    num_known_classes = len(known_classes)
    has_unknown_samples = len(unknown_classes) > 0 and np.any(np.isin(all_labels, unknown_classes))

    # 7. 闭集准确率（仅已知类样本，预测 -1 视为错误）
    if train_label_map is not None:
        label_map = train_label_map
    else:
        sorted_known = sorted(known_classes)
        label_map = {orig: idx for idx, orig in enumerate(sorted_known)}

    known_mask = np.isin(all_labels, known_classes)
    mapped_known_labels = np.array([label_map.get(int(l), -1) for l in all_labels[known_mask]])
    known_preds_orig = all_preds[known_mask]
    known_preds_mapped = []
    for p in known_preds_orig:
        if p == -1:
            known_preds_mapped.append(-1)
        else:
            known_preds_mapped.append(label_map.get(int(p), -2))
    known_preds_mapped = np.array(known_preds_mapped)
    valid_mask = mapped_known_labels >= 0
    mapped_known_labels = mapped_known_labels[valid_mask]
    known_preds_mapped = known_preds_mapped[valid_mask]
    if len(mapped_known_labels) == 0:
        print("错误: 没有有效的已知类样本用于评估")
        return

    accuracy = accuracy_score(mapped_known_labels, known_preds_mapped)
    print(f"\n--- 评估结果 ---")
    print(f"已知类准确率 (Closed-set Accuracy): {accuracy * 100:.2f}%")

    test_results = {
        'accuracy': accuracy * 100,
        'f1_score': None,
        'auroc': None,
        'far': None,
        'mar': None,
        'has_unknown': has_unknown_samples,
    }

    # 8. OOD 检测指标：F1、AUROC、FAR、MAR
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
        print(f"FAR (虚警率, ID 判为 OOD): {far:.2f}%")
        print(f"MAR (漏警率, OOD 判为 ID): {mar:.2f}%")

        # ROC 曲线
        fpr, tpr, _ = roc_curve(true_is_unknown, all_uncertainties)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve for Unknown Detection (EDL Binary)')
        plt.legend(loc='lower right')
        plt.savefig(os.path.join(output_dir, 'roc_curve.png'), dpi=150)
        plt.close()
        print(f"ROC 曲线已保存至: {os.path.join(output_dir, 'roc_curve.png')}")
    else:
        print("测试集中未发现未知类样本，跳过开放集指标与 ROC。")

    # 9. 混淆矩阵（已知类 + Unknown）
    label_names = [f'Class_{c}' for c in known_classes] + ['Unknown']
    cm_labels = []
    cm_preds = []
    for true_label, pred_label in zip(all_labels, all_preds):
        true_label = int(true_label)
        pred_label = int(pred_label)
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
    row_sum = cm.sum(axis=1, keepdims=True)
    cm_pct = np.where(row_sum > 0, cm.astype(float) / row_sum * 100, 0)
    annot = np.array([[f'{cm_pct[i,j]:.1f}%' for j in range(cm_pct.shape[1])] for i in range(cm_pct.shape[0])])
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_pct, annot=annot, fmt='', cmap='Blues',
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix (EDL Binary, OOD-Assisted, %)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'confusion_matrix.png'), dpi=150)
    plt.close()
    print(f"混淆矩阵已保存至: {os.path.join(output_dir, 'confusion_matrix.png')}")

    # 10. 二分类头逐头准确率 + 各头 2×2 混淆矩阵（仅已知类子集）
    known_idx = np.where(known_mask)[0]
    sorted_known_classes = sorted(known_classes)
    binary_accuracies = []
    if len(known_idx) > 0:
        labels_known = all_labels[known_idx]
        mapped = np.array([label_map.get(int(l), -1) for l in labels_known])
        valid = mapped >= 0
        if np.any(valid):
            mapped = mapped[valid]
            idx_ok = known_idx[valid]
            binary_sub = binary_preds_matrix[idx_ok]
            print(f"\n[二分类头] 性能（仅已知类子集）:")
            for i, c_orig in enumerate(known_classes):
                binary_targets = (mapped == i).astype(int)
                head_preds = binary_sub[:, i]
                b_acc = accuracy_score(binary_targets, head_preds)
                binary_accuracies.append(b_acc)
                print(f"  Head {i} (Class {c_orig}): Acc = {b_acc*100:.2f}%")
            avg_binary = np.mean(binary_accuracies)
            print(f"平均二分类准确率: {avg_binary*100:.2f}%")
            results_path = os.path.join(output_dir, 'binary_test_results.txt')
            with open(results_path, 'w', encoding='utf-8') as f:
                f.write(f"Closed-set Accuracy: {test_results['accuracy']:.2f}%\n")
                f.write(f"Uncertainty Threshold: {uncertainty_threshold}\n")
                f.write(f"Average Binary Accuracy (known only): {avg_binary*100:.2f}%\n")
                for i, c_orig in enumerate(known_classes):
                    f.write(f"Head {i} (Class {c_orig}): {binary_accuracies[i]*100:.2f}%\n")
            print(f"结果已保存至: {results_path}")

            # 各二分类头 2×2 混淆矩阵（行归一化百分比），与 Ensemble 版一致
            K = num_known_classes
            ncols = min(3, K)
            nrows = (K + ncols - 1) // ncols
            fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 4 * nrows))
            if K == 1:
                axes = np.array([[axes]])
            else:
                axes = axes.flatten() if K > 1 else [axes]
            for k in range(K):
                ax = axes[k]
                true_k = (mapped == k).astype(np.int32)
                pred_k = binary_sub[:, k]
                cm_k = confusion_matrix(true_k, pred_k, labels=[0, 1])
                row_sum_k = cm_k.sum(axis=1, keepdims=True)
                cm_k_pct = np.where(row_sum_k > 0, cm_k.astype(float) / row_sum_k * 100, 0)
                annot_k = np.array([[f'{cm_k_pct[i,j]:.1f}%' for j in range(2)] for i in range(2)])
                tick_labels = [f'Not {sorted_known_classes[k]}', f'Class {sorted_known_classes[k]}']
                sns.heatmap(cm_k_pct, annot=annot_k, fmt='', cmap='Blues', ax=ax, xticklabels=tick_labels, yticklabels=tick_labels)
                ax.set_title(f'Head {k} (Class {sorted_known_classes[k]})')
                ax.set_ylabel('True')
                ax.set_xlabel('Pred')
            for idx in range(K, len(axes)):
                axes[idx].set_visible(False)
            plt.suptitle('Per-Head Binary Confusion Matrices (Known Classes Only, %)', y=1.02)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'confusion_matrix_per_model.png'), dpi=150)
            plt.close()
            print(f"各二分类头混淆矩阵已保存: {os.path.join(output_dir, 'confusion_matrix_per_model.png')}")

    # 11. 写入 experiment_info.txt
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
            f.write(f"不确定性阈值: {uncertainty_threshold}\n")
            f.write(f"已知类准确率 (Closed-set Accuracy): {test_results['accuracy']:.2f}%\n")
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
        print(f"\n评估结果已追加至: {experiment_info_path}")
    else:
        print(f"警告: 未找到 {experiment_info_path}，跳过结果写入。")


if __name__ == '__main__':
    main()
