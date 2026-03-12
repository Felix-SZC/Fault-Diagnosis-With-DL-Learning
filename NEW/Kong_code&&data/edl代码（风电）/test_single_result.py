# -*- coding: utf-8 -*-
"""
Created on Fri Jul  4 16:52:20 2025

@author: admin
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Jul  1 19:57:51 2025

@author: kongz
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
from biuld_model import DeepModel
import torch.nn.functional as F
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score
import pandas as pd

#%%评估函数
#不确定度
def relu_evidence(y):
    return F.relu(y)

def calculate_uncertainty(out, num_classes):
    ev = relu_evidence(out)
    sum_alpha = ev.sum(dim=1) + num_classes
    uncertainty = num_classes / sum_alpha
    return uncertainty.detach().cpu().numpy()


#%%单个权重测试
def test_single_result(
    result_dir: str,
    w1: float,
    w2: float,
    epoch: int,
    test_loader: torch.utils.data.DataLoader,
    test_loader_ood: torch.utils.data.DataLoader,
    num_classes: int,
    device: torch.device,
    id_class_indices: list = []
):
    
    # 加载模型
    #result_dir = f"67results_w2=1/w1_{w1:.2f}_w2_{w2:.2f}"
    model = DeepModel(num_classes).to(device)
    model.load_state_dict(torch.load(f"{result_dir}/model_{epoch}.pth"))
    model.to(device)
    model.eval()
    
    # 验证计算准确率
    model.eval()
    test_correct = 0
    total = 0
    for data, target in test_loader:
        out0, out1, out2 = model(data)
        preds = out2.argmax(dim=1)
        test_correct += (preds == target).sum().item()
        total += target.size(0)
        
        preds_np = out2.detach().cpu().numpy()
        #print(preds_np.shape)
        
    test_acc = test_correct / total
        
    
    # 计算三层的不确定性（ID和OOD）
    uncertainties = {
        'id': {'layer0': [], 'layer1': [], 'layer2': []},
        'ood': {'layer0': [], 'layer1': [], 'layer2': []}
    }
    
    # ID数据
    with torch.no_grad():
        for data, _ in test_loader:
            out0, out1, out2 = model(data)
            uncertainties['id']['layer0'].extend(calculate_uncertainty(out0, num_classes))
            uncertainties['id']['layer1'].extend(calculate_uncertainty(out1, num_classes))
            uncertainties['id']['layer2'].extend(calculate_uncertainty(out2, num_classes))
    
    #print(np.array(uncertainties['id']['layer0']).shape)
    uncertainties_result_0 = np.array(uncertainties['id']['layer0']).reshape(-1, 1)
    uncertainties_result_1 = np.array(uncertainties['id']['layer1']).reshape(-1, 1)
    uncertainties_result_2 = np.array(uncertainties['id']['layer2']).reshape(-1, 1)
    
    prediction = np.concatenate((preds_np, uncertainties_result_0,uncertainties_result_1,uncertainties_result_2), axis=1)
    
    column_names = id_class_indices + ['第一层不确定性']+['第二层不确定性']+['第三层不确定性']
    
    prediction_df = pd.DataFrame(prediction,columns=column_names)
    prediction_df.to_csv(f'{result_dir}/prediction_result.csv', index=True, encoding='utf-8-sig')
    
    # uncertainties['id']['layer0']
    # uncertainties['id']['layer0']
    # uncertainties['id']['layer0']
    
    
    # OOD数据
    with torch.no_grad():
        for data, _ in test_loader_ood:
            out0, out1, out2 = model(data)
            uncertainties['ood']['layer0'].extend(calculate_uncertainty(out0, num_classes))
            uncertainties['ood']['layer1'].extend(calculate_uncertainty(out1, num_classes))
            uncertainties['ood']['layer2'].extend(calculate_uncertainty(out2, num_classes))
            preds_ood_np = out2.detach().cpu().numpy()

    uncertainties_result_0_ood = np.array(uncertainties['ood']['layer0']).reshape(-1, 1)
    uncertainties_result_1_ood = np.array(uncertainties['ood']['layer1']).reshape(-1, 1)
    uncertainties_result_2_ood = np.array(uncertainties['ood']['layer2']).reshape(-1, 1)
    
    prediction_ood = np.concatenate((preds_ood_np, uncertainties_result_0_ood,uncertainties_result_1_ood,uncertainties_result_2_ood), axis=1)
    
    column_names = id_class_indices + ['第一层不确定性']+['第二层不确定性']+['第三层不确定性']
    
    prediction_ood_df = pd.DataFrame(prediction_ood,columns=column_names)
    prediction_ood_df.to_csv(f'{result_dir}/prediction_ood_result.csv', index=True, encoding='utf-8-sig')


    
    # 计算AUROC（使用第二层）
    # 构建真实的标签：ID 样本为 0，OOD 样本为 1
    y_true = np.concatenate([np.zeros(len(uncertainties['id']['layer2'])), 
                            np.ones(len(uncertainties['ood']['layer2']))])
    # 构建模型的预测分数：ID 样本的不确定性分数和 OOD 样本的不确定性分数
    y_score = np.concatenate([uncertainties['id']['layer2'], 
                             uncertainties['ood']['layer2']])
    auroc = roc_auc_score(y_true, y_score)
    
    '''
# 统一bin范围
    all_uncertainties = (
        uncertainties['id']['layer0'] + uncertainties['ood']['layer0'] +
        uncertainties['id']['layer1'] + uncertainties['ood']['layer1'] +
        uncertainties['id']['layer2'] + uncertainties['ood']['layer2']
    ) 
    min_unc = min(all_uncertainties)
    max_unc = max(all_uncertainties)
    bins = np.linspace(min_unc, max_unc, 51)  # 50个等宽bin（51个边界点）
    '''

    '''
# 第一层
    plt.figure(figsize=(20, 12))
    plt.subplot(3, 1, 1)
    plt.hist(uncertainties['id']['layer0'], bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(uncertainties['ood']['layer0'], bins=bins, alpha=0.5, label='OOD', density=True)
    # 设置X轴和Y轴的范围
    plt.xlim(0, 1)  # X轴范围从0到1
    plt.ylim(0, 50) # Y轴范围从0到40
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title(f'Layer 0 Uncertainty Distribution (w1={w1}, w2={w2})')
    plt.legend()
    plt.grid(True)

# 第二层
    plt.subplot(3, 1, 2)
    plt.hist(uncertainties['id']['layer1'], bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(uncertainties['ood']['layer1'], bins=bins, alpha=0.5, label='OOD', density=True)
    # 设置X轴和Y轴的范围
    plt.xlim(0, 1)  # X轴范围从0到1
    plt.ylim(0, 50) # Y轴范围从0到40
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title(f'Layer 1 Uncertainty Distribution')
    plt.legend()
    plt.grid(True)

# 第三层
    plt.subplot(3, 1, 3)
    plt.hist(uncertainties['id']['layer2'], bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(uncertainties['ood']['layer2'], bins=bins, alpha=0.5, label='OOD', density=True)
    # 设置X轴和Y轴的范围
    plt.xlim(0, 1)  # X轴范围从0到1
    plt.ylim(0, 50) # Y轴范围从0到40
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title(f'Layer 2 Uncertainty Distribution, test_acc={test_acc:.4f},(AUROC={auroc:.4f})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/three_layer_uncertainty/{epoch}.png", dpi=300)
    plt.close()

# 绘制原始不确定性直方图（仅第二层，改为密度图）
    plt.figure(figsize=(10, 6))
    plt.hist(uncertainties['id']['layer2'], bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(uncertainties['ood']['layer2'], bins=bins, alpha=0.5, label='OOD', density=True)
    
    # 设置X轴和Y轴的范围
    plt.xlim(0, 1)  # X轴范围从0到1
    plt.ylim(0, 50) # Y轴范围从0到40
    
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title(f'Uncertainty Distribution (w1={w1}, w2={w2}, test_acc={test_acc:.4f},AUROC={auroc:.4f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{result_dir}/uncertainty_distribution/{epoch}.png", dpi=300)
    plt.close()
    
    # 计算混淆矩阵
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for data, target in test_loader:
            _, _, out2 = model(data)
            preds = out2.argmax(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(target.cpu().numpy())
    
    cm = confusion_matrix(all_targets, all_preds)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=[f'{i}' for i in id_class_indices], 
                yticklabels=[f'{i}' for i in id_class_indices])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (w1={w1}, w2={w2})')
    plt.savefig(f"{result_dir}/confusion_matrix/{epoch}.png", dpi=300)
    plt.close()

    
    # 保存指标
    with open(f"{result_dir}/metrics/{epoch}.txt", 'w') as f:
        f.write(f"Validation Accuracy: {test_acc:.4f}\n")
        f.write(f"Epoch: {epoch}\n")
        f.write(f"w1: {w1:.2f}\n")  # 确保保存两位小数
        f.write(f"w2: {w2:.2f}\n")  # 确保保存两位小数
        f.write(f"ID Uncertainty Mean (Layer0): {np.mean(uncertainties['id']['layer0']):.4f}\n")
        f.write(f"OOD Uncertainty Mean (Layer0): {np.mean(uncertainties['ood']['layer0']):.4f}\n")
        f.write(f"ID Uncertainty Mean (Layer1): {np.mean(uncertainties['id']['layer1']):.4f}\n")
        f.write(f"OOD Uncertainty Mean (Layer1): {np.mean(uncertainties['ood']['layer1']):.4f}\n")
        f.write(f"ID Uncertainty Mean (Layer2): {np.mean(uncertainties['id']['layer2']):.4f}\n")
        f.write(f"OOD Uncertainty Mean (Layer2): {np.mean(uncertainties['ood']['layer2']):.4f}\n")
        f.write(f"AUROC: {auroc:.4f}\n")
    '''
    
    return 0
