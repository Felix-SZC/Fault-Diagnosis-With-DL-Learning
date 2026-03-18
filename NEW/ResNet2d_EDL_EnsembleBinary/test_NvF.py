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


def run_test_loop(models, test_loader, device, K, uncertainty_threshold):
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

            final_probs_list = []
            p_normal = p_yes_list[0]
            p_fault_total = 1.0 - p_normal
            final_probs_list.append(p_normal)
            for k in range(1, K):
                p_k_final = p_fault_total * p_yes_list[k]
                final_probs_list.append(p_k_final)
            final_probs = torch.stack(final_probs_list, dim=1)

            binary_preds = []
            binary_preds.append((p_yes_list[0] > 0.5).cpu().numpy().astype(np.int32))
            for k in range(1, K):
                binary_preds.append((p_yes_list[k] > 0.5).cpu().numpy().astype(np.int32))
            binary_preds_list.append(np.stack(binary_preds, axis=1))

            preds_local = final_probs.argmax(dim=1).cpu().numpy()
            for i in range(len(labels)):
                if int(labels[i]) < K:
                    closed_set_true.append(int(labels[i]))
                    closed_set_pred.append(int(preds_local[i]))

            u_k_stack = torch.stack(u_k_list, dim=1)
            uncertainty = u_k_stack.mean(dim=1).cpu().numpy()

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
    return all_labels, all_preds, all_uncertainties, binary_preds_matrix, closed_set_true, closed_set_pred


def main():
    parser = argparse.ArgumentParser(description='EDL 集成二分类 (NvF)：仅使用 EDL mean 不确定度的测试脚本')
    parser.add_argument('--config', type=str, default='config.yaml')
    parser.add_argument('--checkpoint', type=str, default=None)
    parser.add_argument('--test_all_epochs', '--all', action='store_true', dest='test_all_epochs')
    parser.add_argument('--uncertainty_threshold', '--uncertainty', type=float, default=0.5, dest='uncertainty_threshold')
    parser.add_argument('--threshold_from_val', action='store_true')
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']

    ensemble_strategy = train_config.get('ensemble_strategy', 'Normal_vs_Fault_i')
    if ensemble_strategy != 'Normal_vs_Fault_i':
        print(f\"警告：配置中的 ensemble_strategy={ensemble_strategy}，但 test_NvF.py 仅支持 Normal_vs_Fault_i，将按该策略解释结果。")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f\"使用设备: {device}")

    if args.checkpoint:
        checkpoint_dir = os.path.abspath(args.checkpoint)
    else:
        checkpoint_dir = train_config['checkpoint_dir']
    if not os.path.isdir(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir)
    output_dir = args.output_dir or os.path.join(checkpoint_dir, 'test_nvf')
    os.makedirs(output_dir, exist_ok=True)

    known_classes = data_config['openset']['known_classes']
    unknown_classes = data_config['openset'].get('unknown_classes', [])
    K = len(known_classes)

    backbone_type = model_config.get('type', 'ResNet18_2d_Light')

    test_dataset = get_dataset(data_config, split='test', filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
    has_unknown_samples = any([v >= K for v in test_dataset.y])
    print(f\"测试集大小: {len(test_dataset)}")

    if args.test_all_epochs:
        epochs = discover_epochs(checkpoint_dir, K)
        if not epochs:
            print("未找到任何 epoch 权重（需 checkpoint_dir/epochs/*/ 下 model_k.pth），请先训练并开启 save_every_epoch。")
            return
        print(f\"全测试模式：共 {len(epochs)} 个 epoch，将依次测试并汇总到 test_results_all_epochs_nvf.csv")
        models = load_models(checkpoint_dir, K, backbone_type, device, epoch=None)
        if args.threshold_from_val:
            val_dataset = get_dataset(data_config, split='test', filter_classes=known_classes)
            val_loader = DataLoader(val_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
            uncertainty_threshold = compute_uncertainty_threshold_iqr(val_loader, models, device, K)
        else:
            uncertainty_threshold = args.uncertainty_threshold
        del models
        all_epoch_rows = []
        for e in epochs:
            models_e = load_models(checkpoint_dir, K, backbone_type, device, epoch=e)
            all_labels_e, all_preds_e, all_uncertainties_e, binary_preds_matrix_e, closed_set_true_e, closed_set_pred_e = run_test_loop(
                models_e, test_loader, device, K, uncertainty_threshold
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
        print(f\"全 epoch 测试结果已保存: {csv_path}")

    models = load_models(checkpoint_dir, K, backbone_type, device, epoch=None)
    if args.threshold_from_val:
        val_dataset = get_dataset(data_config, split='test', filter_classes=known_classes)
        val_loader = DataLoader(val_dataset, batch_size=train_config.get('batch_size', 32), shuffle=False)
        uncertainty_threshold = compute_uncertainty_threshold_iqr(val_loader, models, device, K)
        print(f\"由已知类测试子集 IQR 得到不确定性阈值: {uncertainty_threshold:.4f}")
    else:
        uncertainty_threshold = args.uncertainty_threshold
        print(f\"阈值 (u > 此值判 OOD): {uncertainty_threshold}")

    all_labels, all_preds, all_uncertainties, binary_preds_matrix, closed_set_true, closed_set_pred = run_test_loop(
        models, test_loader, device, K, uncertainty_threshold
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
    print(f\"已知类准确率 (Closed-set Accuracy): {accuracy * 100:.2f}%")

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

    print(f\"平均二分类准确率: {np.mean(binary_accuracies) * 100:.2f}%")

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

    if has_unknown_samples:
        true_is_unknown = (~known_mask).astype(int)
        pred_is_unknown = (all_preds == -1).astype(int)
        f1 = f1_score(true_is_unknown, pred_is_unknown)
        print(f\"F1-Score (检测未知类): {f1:.4f}")
        auroc = roc_auc_score(true_is_unknown, all_uncertainties)
        print(f\"AUROC (区分已知/未知): {auroc:.4f}")
        tp = np.sum((true_is_unknown == 1) & (pred_is_unknown == 1))
        fp = np.sum((true_is_unknown == 0) & (pred_is_unknown == 1))
        tn = np.sum((true_is_unknown == 0) & (pred_is_unknown == 0))
        fn = np.sum((true_is_unknown == 1) & (pred_is_unknown == 0))
        far = (fp / (fp + tn) * 100) if (fp + tn) > 0 else 0.0
        mar = (fn / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
        print(f\"FAR (虚警率): {far:.2f}%")
        print(f\"MAR (漏警率): {mar:.2f}%")
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


if __name__ == '__main__':
    main()

