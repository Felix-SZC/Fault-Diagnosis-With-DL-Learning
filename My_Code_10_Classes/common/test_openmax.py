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

    # 4. 准备测试数据
    base_dir = data_config.get('raw_signal_output_dir')
    split_dir = os.path.join(base_dir, 'test')
    known_classes = data_config['openset']['known_classes']
    
    # 创建一个临时数据集以获取训练时的标签映射
    # 这个映射与训练时使用的映射完全一致（基于 sorted(unique_labels)）
    temp_dataset = RawSignalDataset(split_dir=split_dir, filter_classes=known_classes)
    train_label_map = temp_dataset.label_map if hasattr(temp_dataset, 'label_map') and temp_dataset.label_map else None
    
    # 使用全部数据（包括已知和未知类）进行测试，保持原始标签
    test_dataset = RawSignalDataset(split_dir=split_dir, filter_classes=None)
    test_loader = DataLoader(test_dataset, batch_size=train_config['batch_size'], shuffle=False)
    print(f"测试集大小: {len(test_dataset)}")
    
    if train_label_map:
        print(f"训练时的标签映射: {train_label_map}")

    # 5. 执行评估
    all_labels = []
    all_preds = []
    all_openmax_probs = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(device)
            logits, features = model(inputs, return_features=True)
            
            # OpenMax 计算需要在 CPU 上进行，并且是针对单个样本的
            logits_np = logits.cpu().numpy()
            features_np = features.cpu().numpy()
            mavs_np = mavs.cpu().numpy()
            
            batch_probs = []
            for i in range(logits_np.shape[0]): # 遍历批次中的每个样本
                # 关键修复：只取已知类对应的 Logits
                # 假设 mavs 是按照 known_classes 的顺序存储的（由 train_openmax.py 保证）
                # 且模型的 logits 输出索引对应原始类别 ID
                # known_classes: [1, 2, ..., 9]
                # mavs[0] 对应 class 1, mavs[1] 对应 class 2 ...
                
                current_logits = logits_np[i]
                
                # 模型输出已经是针对已知类的 (0..num_classes-1)，对应 known_classes 中的顺序
                known_logits = current_logits
                
                # Strict Implementation:
                # 第二个参数 input_vector 必须是 logits (与 MAV 所在空间一致)
                prob = compute_openmax_prob(
                    known_logits, known_logits, mavs_np, weibull_models
                )
                
                # prob 的长度是 len(known_classes) + 1
                # 我们需要将其映射回原始的类别空间 + 未知类
                # 这里为了简化评估，我们保持 OpenMax 的输出格式 (K+1)，
                # 但需要注意，现在的索引 0 对应 known_classes[0]，索引 1 对应 known_classes[1]...
                # 最后一个索引对应 Unknown。
                # 下面的 preds 计算逻辑需要适配这一点。
                batch_probs.append(prob)
            
            openmax_probs = torch.from_numpy(np.array(batch_probs))
            
            # 预测类别索引 (0 到 K)
            # 0 到 K-1 对应 known_classes 中的类别
            # K 对应 Unknown
            preds_local_idx = torch.argmax(openmax_probs, dim=1)
            
            # 将局部索引映射回原始类别索引
            # 0..K-1 -> known_classes[idx]
            # K -> -1 (表示 Unknown)
            preds = []
            for p in preds_local_idx:
                if p < len(known_classes):
                    preds.append(known_classes[p])
                else:
                    preds.append(-1) # 使用 -1 明确标记为 Unknown
            
            preds = torch.tensor(preds)
            
            all_labels.extend(labels.numpy())
            all_preds.extend(preds.numpy())
            all_openmax_probs.append(openmax_probs.numpy())

    all_openmax_probs = np.concatenate(all_openmax_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)
    
    # 6. 计算指标
    unknown_classes = data_config['openset'].get('unknown_classes', [])
    num_known_classes = len(known_classes)
    
    # 检查测试集中是否有未知类
    has_unknown_samples = len(unknown_classes) > 0 and np.any(np.isin(all_labels, unknown_classes))

    # 分离已知和未知样本（使用原始标签）
    known_mask = np.isin(all_labels, known_classes)
    
    # 指标1: 已知类的分类准确率 (Closed-set Accuracy)
    # 需要将原始标签映射到训练时的标签
    # 训练时 RawSignalDataset 使用 sorted(unique_labels) 创建映射
    if train_label_map is not None:
        # 使用训练时的实际映射（这是最准确的方式）
        label_map = train_label_map
        print(f"使用训练时的标签映射: {label_map}")
    else:
        # 回退方案：使用 sorted(known_classes) 创建映射（与训练时 RawSignalDataset 的逻辑一致）
        sorted_known = sorted(known_classes)
        label_map = {orig_label: new_label for new_label, orig_label in enumerate(sorted_known)}
        print(f"使用回退标签映射: {label_map}")
    
    # 将已知类的原始标签映射到训练时的标签
    mapped_known_labels = np.array([label_map.get(l, -1) for l in all_labels[known_mask]])
    
    # 获取已知类样本的预测值
    known_preds_orig = all_preds[known_mask] # 这里的预测值是原始 ID (1..9) 或 -1 (Unknown)
    
    # 如果预测为 -1 (Unknown)，在闭集准确率计算中应视为错误
    # 如果预测为 1..9，需要将其映射回 0..8 (训练标签) 才能和 mapped_known_labels 比较
    # 注意：闭集准确率通常只看已知类是否分对，如果分到未知类也算错
    
    known_preds_mapped = []
    for p in known_preds_orig:
        if p == -1:
            known_preds_mapped.append(-1) # 错误
        else:
            # 查找预测的原始 ID 在 label_map 中的对应值
            if p in label_map:
                known_preds_mapped.append(label_map[p])
            else:
                known_preds_mapped.append(-2) # 预测为了其他已知类？或者异常
                
    known_preds_mapped = np.array(known_preds_mapped)
    
    # 过滤掉映射失败的样本（理论上不应该发生）
    valid_mask = mapped_known_labels >= 0
    if not np.all(valid_mask):
        print(f"警告: {np.sum(~valid_mask)} 个已知类样本无法映射到训练标签")
    mapped_known_labels = mapped_known_labels[valid_mask]
    known_preds_mapped = known_preds_mapped[valid_mask]
    
    if len(mapped_known_labels) == 0:
        print("错误: 没有有效的已知类样本用于评估")
        return
    
    accuracy = accuracy_score(mapped_known_labels, known_preds_mapped)
    print(f"\n--- 评估结果 ---")
    print(f"已知类准确率 (Accuracy): {accuracy * 100:.2f}%")

    # 存储评估结果，用于写入文件
    test_results = {
        'accuracy': accuracy * 100,
        'f1_score': None,
        'auroc': None,
        'has_unknown': has_unknown_samples
    }

    # 只有在存在未知类时才计算开放集指标
    if has_unknown_samples:
        unknown_mask = np.isin(all_labels, unknown_classes)
        # 指标2: F1-Score 用于检测未知类
        # 将预测为 "未知" (类别 ID 为 -1) 的作为正例
        pred_is_unknown = (all_preds == -1).astype(int)
        true_is_unknown = unknown_mask.astype(int)
        
        f1 = f1_score(true_is_unknown, pred_is_unknown)
        print(f"F1-Score (检测未知类): {f1:.4f}")
        test_results['f1_score'] = f1

        # 指标3: AUROC 用于区分已知/未知
        # 使用 "未知" 类的概率作为分数 (openmax_probs 的最后一列)
        unknown_prob_scores = all_openmax_probs[:, num_known_classes]
        auroc = roc_auc_score(true_is_unknown, unknown_prob_scores)
        print(f"AUROC (区分已知/未知): {auroc:.4f}")
        test_results['auroc'] = auroc
        
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
    else:
        print("测试集中未发现未知类样本，跳过开放集指标评估。")

    # 绘制混淆矩阵
    # 准备标签：已知类使用原始标签索引，未知类统一标记为 num_known_classes
    # 预测：需要将 all_preds (包含原始 ID 和 -1) 映射到混淆矩阵的索引
    # CM 索引：0..K-1 对应 known_classes, K 对应 Unknown
    
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
        
        # 处理预测标签
        if pred_label == -1:
            # 预测为未知
            cm_preds.append(num_known_classes)
        elif pred_label in known_classes:
            # 预测为已知类，找到其在 known_classes 中的索引
            pred_idx = known_classes.index(pred_label)
            cm_preds.append(pred_idx)
        else:
            # 异常情况 (例如预测出了不在 known_classes 中的类，理论上不应发生)
            cm_preds.append(num_known_classes) # 归为 Unknown 或其他
    
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

    # 7. 将评估结果追加到 experiment_info.txt
    experiment_info_path = os.path.join(checkpoint_dir, 'experiment_info.txt')
    if os.path.exists(experiment_info_path):
        # 读取现有内容
        with open(experiment_info_path, 'r', encoding='utf-8') as f:
            existing_content = f.read()
        
        # 检查是否已经有评估结果部分，如果有则删除旧的部分
        if '【测试评估结果】' in existing_content:
            # 找到评估结果部分的位置并删除
            lines = existing_content.split('\n')
            new_lines = []
            skip_section = False
            for line in lines:
                if '【测试评估结果】' in line:
                    skip_section = True
                elif skip_section and line.startswith('=') and len(line) >= 80:
                    # 遇到下一个分隔符，停止跳过
                    skip_section = False
                    new_lines.append(line)
                elif not skip_section:
                    new_lines.append(line)
            existing_content = '\n'.join(new_lines)
        
        # 追加评估结果
        with open(experiment_info_path, 'w', encoding='utf-8') as f:
            f.write(existing_content.rstrip() + '\n\n')
            f.write("【测试评估结果】\n")
            f.write("-" * 80 + "\n")
            f.write(f"测试集大小: {len(test_dataset):,}\n")
            f.write(f"已知类准确率 (Closed-set Accuracy): {test_results['accuracy']:.2f}%\n")
            
            if test_results['has_unknown']:
                if test_results['f1_score'] is not None:
                    f.write(f"F1-Score (检测未知类): {test_results['f1_score']:.4f}\n")
                if test_results['auroc'] is not None:
                    f.write(f"AUROC (区分已知/未知): {test_results['auroc']:.4f}\n")
            else:
                f.write("测试集中未发现未知类样本，未计算开放集指标。\n")
            
            f.write("\n")
            f.write("=" * 80 + "\n")
        
        print(f"\n评估结果已追加至: {experiment_info_path}")
    else:
        print(f"警告: 未找到实验信息文件 {experiment_info_path}，跳过结果写入。")

if __name__ == '__main__':
    main()
