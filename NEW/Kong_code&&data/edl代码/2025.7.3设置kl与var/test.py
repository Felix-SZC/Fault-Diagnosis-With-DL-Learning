# -*- coding: utf-8 -*-
"""
Created on Thu Jul  3 14:45:21 2025

@author: admin
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_auc_score
import os
import pandas as pd
from tqdm import tqdm
from merge_load import load_data  # 确保存在
from biuld_model import SimpleMambaBlock,DeepModel
from test_single import test_single

#%%随机数种子
def set_random_seed(seed=42):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    #torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.benchmark = False
    
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#%% 参数

batch_size = 256
annealing_step = 500 #退火系数
epochs = 500

data_num = 1 #数据包文件夹




set_random_seed(42)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")


#%% 加载数据
# (train_data, train_target), (test_data, test_target) = load_data(2)
# train_target = np.argmax(train_target, axis=-1)
# test_target = np.argmax(test_target, axis=-1)

train_data = np.load(f'../../数据打包/{data_num}/X_train.npy')
train_target = np.load(f'../../数据打包/{data_num}/Y_train.npy')
test_data = np.load(f'../../数据打包/{data_num}/X_test.npy')
test_target = np.load(f'../../数据打包/{data_num}/Y_test.npy')

# 1. 定义 OOD 类别（举例）
class_labels = np.array([
    '1正常', '2轴承内圈', '3轴承外圈', '4轴承滚子',
    '5轴承组合', '6齿根裂纹', '7齿轮断齿', '8齿轮点蚀', '9齿轮缺齿'
])

ood_labels = ['7齿轮断齿','9齿轮缺齿']

num_classes = len(class_labels) - len(ood_labels)

id_class_indices = list(np.setdiff1d(class_labels, ood_labels))

# 2. 对训练集做分割
mask_train_ood = np.isin(train_target, ood_labels)   # True 表示该样本是 OOD
train_ood_idx  = np.where(mask_train_ood)[0]         # OOD 样本索引
train_id_idx   = np.where(~mask_train_ood)[0]        # ID 样本索引

train_id_data   = train_data[train_id_idx]
train_id_target = train_target[train_id_idx]
train_ood_data   = train_data[train_ood_idx]
train_ood_target = train_target[train_ood_idx]

# 3. 对测试集做分割（同理）
mask_test_ood = np.isin(test_target, ood_labels)
test_ood_idx  = np.where(mask_test_ood)[0]
test_id_idx   = np.where(~mask_test_ood)[0]

test_id_data   = test_data[test_id_idx]
test_id_target = test_target[test_id_idx]
test_ood_data   = test_data[test_ood_idx]
test_ood_target = test_target[test_ood_idx]

# 4. （可选）输出一下各部分大小
print(f"Train ID:   {len(train_id_idx)} 样本")
print(f"Train OOD:  {len(train_ood_idx)} 样本")
print(f"Test ID:    {len(test_id_idx)} 样本")
print(f"Test OOD:   {len(test_ood_idx)} 样本")


#转换成数字标签
all_id_labels = np.unique(
    np.concatenate([train_id_target, test_id_target])
)

label2idx = {label: idx for idx, label in enumerate(all_id_labels)}
# 比如 {'1正常':0, '2轴承内圈':1, ...}

# 训练集
train_labels_numeric = np.array(
    [label2idx[label] for label in train_id_target],
    dtype=np.int64
)

# 测试集
test_labels_numeric = np.array(
    [label2idx[label] for label in test_id_target],
    dtype=np.int64
)



#%%转换成tensor
train_data_id = torch.tensor(train_id_data, dtype=torch.float32)
train_target_id = torch.tensor(train_labels_numeric, dtype=torch.long)
test_data_id = torch.tensor(test_id_data, dtype=torch.float32)
test_target_id = torch.tensor(test_labels_numeric, dtype=torch.long)
test_data_ood = torch.tensor(test_ood_data, dtype=torch.float32)
test_target_ood = torch.tensor(np.zeros(shape=test_ood_idx.shape), dtype=torch.long)

# 移动到设备
train_data_id = train_data_id.to(device)
train_target_id = train_target_id.to(device)
test_data_id = test_data_id.to(device)
test_target_id = test_target_id.to(device)
test_data_ood = test_data_ood.to(device)
test_target_ood = test_target_ood.to(device)

train_dataset_id = TensorDataset(train_data_id, train_target_id)
train_loader = DataLoader(train_dataset_id, batch_size=batch_size, shuffle=True)
test_dataset_id = TensorDataset(test_data_id, test_target_id)
test_loader = DataLoader(test_dataset_id, batch_size=test_data_id.shape[0], shuffle=False)
test_dataset_ood = TensorDataset(test_data_ood, test_target_ood)
test_loader_ood = DataLoader(test_dataset_ood, batch_size=test_data_ood.shape[0], shuffle=False)




#%%测试

# 固定 w2=1
w2 = 1
# w1 从 1 到 10，步长为 0.05
#w1_values = np.arange(0.1, 10.1, 0.1)  # 包含10.0

# w1_values = [0.0001,0.0002,0.0005,0.0008,
#                  0.001,0.002,0.005,0.008,
#                  0.01,0.02,0.05,0.08,
#                  0.1,0.2,0.5,0.8,
#                  1,1.2,1.5,1.8,
#                  2,2.5,
#                  3,3.5,
#                  4,4.5,
#                  5,5.5,
#                  6,6.5,
#                  7,7.5,
#                  8,8.5,
#                  9,9.5,
#                  10]

w1_values = [1]
# 创建所有组合
weight_combinations = [(w1, w2) for w1 in w1_values]

# 计算组合数量
num_combinations = len(weight_combinations)
print(f"w1值数量: {len(w1_values)} (范围: {w1_values[0]} - {w1_values[-1]})")
print(f"w2固定为: {w2}")
print(f"总组合数: {num_combinations}")

results = []

print(f"开始测试 {num_combinations} 种权重组合...")

for i, (w1, w2) in enumerate(weight_combinations):
    
    print(f"进度: {i+1}/{num_combinations} ({(i+1)/num_combinations*100:.1f}%) - 测试 w1={w1}, w2={w2}")


    result_dir = f"results_w3=1/w1_{w1}_w2_{w2}"
    
    # 创建文件夹，exist_ok=True 表示已存在时不报错
    os.makedirs(result_dir+'/three_layer_uncertainty', exist_ok=True)
    os.makedirs(result_dir+'/uncertainty_distribution', exist_ok=True)
    os.makedirs(result_dir+'/confusion_matrix', exist_ok=True)
    os.makedirs(result_dir+'/metrics', exist_ok=True)


    for epoch in range(epochs):

        
        test_acc, auroc, id_unc_mean, ood_unc_mean = test_single(
            result_dir=result_dir,
            w1=w1,
            w2=w2,
            epoch=epoch,
            test_loader=test_loader,
            test_loader_ood=test_loader_ood,
            num_classes=num_classes,
            device=device,
            id_class_indices=id_class_indices
        )
        results.append({
            'epoch':epoch,
            'w1': w1,
            'w2': w2,
            'test_acc': test_acc,
            'auroc': auroc,
            'id_unc_mean': id_unc_mean,
            'ood_unc_mean': ood_unc_mean
        })

    # 每完成1个组合保存一次中间结果
    if (i+1) % 1 == 0:
        temp_df = pd.DataFrame(results)
        temp_df.to_csv(f"{result_dir}/intermediate_resultskaiming1.csv", index=False)
        print(f"已保存中间结果: {len(results)} 条记录")

# 保存所有结果
results_df = pd.DataFrame(results)
results_df.to_csv(f"{result_dir}/all_results.csv", index=False)
print("所有权重组合测试完成！")

# ... 其余可视化代码保持不变 ...

# ... 其余可视化代码保持不变 ...

# # 绘制权重组合效果热力图
# acc_matrix = results_df.pivot(index='w1', columns='w2', values='best_accuracy')
# auroc_matrix = results_df.pivot(index='w1', columns='w2', values='auroc')

# # 画 w1 vs accuracy/auroc 折线图
# plt.figure(figsize=(10, 6))
# plt.plot(results_df['w1'], results_df['best_accuracy'], marker='o', label='Accuracy')
# plt.xlabel('w1')
# plt.ylabel('Accuracy')
# plt.title('Accuracy vs w1 (w2=1)')
# plt.grid(True)
# plt.savefig(f"{result_dir}/accuracy_vs_w1.png")
# plt.close()

# plt.figure(figsize=(10, 6))
# plt.plot(results_df['w1'], results_df['auroc'], marker='o', label='AUROC')
# plt.xlabel('w1')
# plt.ylabel('AUROC')
# plt.title('AUROC vs w1 (w2=1)')
# plt.grid(True)
# plt.savefig(f"{result_dir}/auroc_vs_w1.png")
# plt.close()
    
