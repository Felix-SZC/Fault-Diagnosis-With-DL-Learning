import argparse
import os
import sys
import torch
from torch.utils.data import DataLoader
import pickle
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# 添加项目根目录
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from common.utils.helpers import load_config
from common.utils.data_loader import RawSignalDataset
from models import get_model
from common.openmax import compute_openmax_prob

def main():
    parser = argparse.ArgumentParser(description='OpenMax Evaluation Script')
    parser.add_argument('--config', type=str, required=True, help='Path to the experiment config file')
    args = parser.parse_args()

    # 1. 加载配置
    config = load_config(args.config)
    data_config = config['data']
    model_config = config['model']
    train_config = config['train']
    
    checkpoint_dir = train_config['checkpoint_dir']
    openmax_dir = os.path.join(checkpoint_dir, 'openmax_files')
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    # 2. 加载模型
    model = get_model(model_config.get('type'), num_classes=model_config.get('num_classes')).to(device)
    model.load_state_dict(torch.load(os.path.join(checkpoint_dir, 'best_model.pth'), weights_only=True))
    model.eval()
    print("模型加载成功。")

    # 3. 加载 OpenMax 文件
    mavs = torch.load(os.path.join(openmax_dir, 'mavs.pth'), weights_only=True).to(device)
    with open(os.path.join(openmax_dir, 'weibull_models.pkl'), 'rb') as f:
        weibull_models = pickle.load(f)
    print("MAVs 和 Weibull 模型加载成功。")

    # 4. 准备测试数据（不过滤任何类，保持原始标签）
    base_dir = data_config.get('raw_signal_output_dir')
    split_dir = os.path.join(base_dir, 'test')
    test_dataset = RawSignalDataset(split_dir=split_dir, filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)
    print(f"测试集大小: {len(test_dataset)}")

    # 5. 执行评估
    all_labels = []
    all_preds = []
    all_openmax_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits, features = model(inputs, return_features=True)
            
            # OpenMax 计算需要在 CPU 上进行
            openmax_probs = compute_openmax_prob(logits.cpu(), features.cpu(), mavs.cpu(), weibull_models)
            
            preds = torch.argmax(openmax_probs, dim=1)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_openmax_probs.append(openmax_probs.numpy())

    all_openmax_probs = np.concatenate(all_openmax_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 6. 计算指标
    known_classes = data_config['openset']['known_classes']
    unknown_classes = data_config['openset']['unknown_classes']
    num_known_classes = len(known_classes)

    # 分离已知和未知样本
    known_mask = np.isin(all_labels, known_classes)
    unknown_mask = np.isin(all_labels, unknown_classes)

    # 指标1: 已知类的分类准确率 (Closed-set Accuracy)
    # 需要将原始标签映射到训练时的标签
    label_map = {orig_label: new_label for new_label, orig_label in enumerate(known_classes)}
    mapped_known_labels = np.array([label_map[l] for l in all_labels[known_mask]])
    known_preds = all_preds[known_mask]
    
    accuracy = accuracy_score(mapped_known_labels, known_preds)
    print(f"\n--- 评估结果 ---")
    print(f"已知类准确率 (Accuracy): {accuracy * 100:.2f}%")

    # 指标2: F1-Score 用于检测未知类
    # 将预测为 "未知" (类别索引为 num_known_classes) 的作为正例
    pred_is_unknown = (all_preds == num_known_classes).astype(int)
    true_is_unknown = unknown_mask.astype(int)
    
    f1 = f1_score(true_is_unknown, pred_is_unknown)
    print(f"F1-Score (检测未知类): {f1:.4f}")

    # 指标3: AUROC 用于区分已知/未知
    # 使用 "未知" 类的概率作为分数
    unknown_prob_scores = all_openmax_probs[:, num_known_classes]
    auroc = roc_auc_score(true_is_unknown, unknown_prob_scores)
    print(f"AUROC (区分已知/未知): {auroc:.4f}")
    
    # 绘制 ROC 曲线
    fpr, tpr, _ = roc_curve(true_is_unknown, unknown_prob_scores)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {auroc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve for Unknown Detection')
    plt.legend(loc="lower right")
    
    # 保存ROC曲线图
    roc_curve_path = os.path.join(checkpoint_dir, 'test', 'roc_curve.png')
    os.makedirs(os.path.dirname(roc_curve_path), exist_ok=True)
    plt.savefig(roc_curve_path)
    print(f"ROC 曲线已保存至: {roc_curve_path}")
    plt.close()
    
    # 绘制混淆矩阵
    # 准备标签：已知类使用原始标签，未知类统一标记为 "Unknown"
    # 预测：已知类使用预测的类别索引，未知类标记为 num_known_classes
    cm_labels = []
    cm_preds = []
    
    # 为混淆矩阵创建标签映射
    # 已知类：使用原始标签名称
    # 未知类：统一标记为 "Unknown"
    label_names = []
    for orig_label in known_classes:
        label_names.append(f'Class_{orig_label}')
    label_names.append('Unknown')
    
    # 构建混淆矩阵的标签和预测
    for true_label, pred_label in zip(all_labels, all_preds):
        if true_label in known_classes:
            # 已知类的真实标签：找到在 known_classes 中的索引
            true_idx = known_classes.index(true_label)
            cm_labels.append(true_idx)
        else:
            # 未知类的真实标签：标记为 "Unknown" (索引为 num_known_classes)
            cm_labels.append(num_known_classes)
        
        # 预测标签直接使用（已经是 0 到 num_known_classes 的索引）
        cm_preds.append(pred_label)
    
    cm_labels = np.array(cm_labels)
    cm_preds = np.array(cm_preds)
    
    # 计算混淆矩阵
    cm = confusion_matrix(cm_labels, cm_preds)
    
    # 绘制混淆矩阵
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=label_names, yticklabels=label_names)
    plt.title('Confusion Matrix (OpenMax)')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    # 保存混淆矩阵
    cm_path = os.path.join(checkpoint_dir, 'test', 'confusion_matrix.png')
    plt.savefig(cm_path, dpi=150)
    print(f"混淆矩阵已保存至: {cm_path}")
    plt.close()

if __name__ == '__main__':
    main()
