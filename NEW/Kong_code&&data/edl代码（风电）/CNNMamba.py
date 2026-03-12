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

train_data = np.load(f'../../数据打包（风电）/{data_num}/X_train.npy')
train_target = np.load(f'../../数据打包（风电）/{data_num}/Y_train.npy')
test_data = np.load(f'../../数据打包（风电）/{data_num}/X_test.npy')
test_target = np.load(f'../../数据打包（风电）/{data_num}/Y_test.npy')

# 1. 定义 OOD 类别（举例）
class_labels = np.array([
    '1正常', '2二级小齿轮点蚀', '3二级后轴承保持架', '4二级后轴承滚动体',
    '5二级后轴承磨损', '6行星轴承滚珠磨损'
])

ood_labels = ['3二级后轴承保持架']

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

#%% Evidential Loss 相关函数
def relu_evidence(y):
    return F.relu(y)

def kl_divergence(alpha, num_classes, device=None):
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_alpha)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    second_term = (
        (alpha - ones)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))
        .sum(dim=1, keepdim=True)
    )
    return first_term + second_term

def loglikelihood_loss(y, alpha, device=None, var_only=False):
    y = y.to(device)
    alpha = alpha.clamp(min=1e-4).to(device)
    S = torch.sum(alpha, dim=1, keepdim=True)
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    loglikelihood_var = torch.sum(alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True)
    return loglikelihood_var if var_only else loglikelihood_err + loglikelihood_var

def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None, var_only=False, kl=True):
    loglikelihood = loglikelihood_loss(y, alpha, device=device, var_only=var_only)
    annealing_coef = torch.min(
        torch.tensor(1.0),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32)
    ).to(device)
    kl_alpha = ((alpha - 1) * (1 - y) + 1).clamp(min=1e-4)
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    return loglikelihood + kl_div if kl else loglikelihood



#%%评估函数
#不确定度
def calculate_uncertainty(out, num_classes):
    ev = relu_evidence(out)
    sum_alpha = ev.sum(dim=1) + num_classes
    uncertainty = num_classes / sum_alpha
    return uncertainty.detach().cpu().numpy()

#%%训练模型
def train_and_evaluate(w1, w2, result_dir):
    """训练模型并评估性能"""
    
    #result_dir = "my_results"  # 你想创建的文件夹名称
    # 获取当前脚本文件所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))     
    # 拼出完整路径
    full_path = os.path.join(base_dir, result_dir)        
    # 创建文件夹，exist_ok=True 表示已存在时不报错
    os.makedirs(full_path, exist_ok=True)
    print(f"已在 {base_dir} 下创建/确认存在文件夹: {result_dir}")
    
    # 初始化模型
    model = DeepModel(num_classes).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    
    best_acc = 0.0
    best_epoch = 0
    train_acc_history = []
    val_acc_history = []
    
    loss_history = []
    loss_1_history = []
    loss_2_history = []
    loss_3_history = []

    # 脚本所在目录
    base_dir = os.path.dirname(os.path.abspath(__file__))    
    # 拼出一级文件夹路径
    level1 = os.path.join(base_dir, result_dir)      
    # 依次创建（exist_ok=True 表示已存在则忽略）
    os.makedirs(level1, exist_ok=True)        
    print(f"已创建或确认存在：{level1}")
    
    # 训练循环
    for epoch in range(epochs):
        
        model.train()
        correct, total = 0, 0
        epoch_loss = 0  # 初始化当前 epoch 的总损失
        epoch_loss_1 = 0
        epoch_loss_2 = 0
        epoch_loss_3 = 0
        
        for data, target in train_loader:
            optimizer.zero_grad()
            y_onehot = F.one_hot(target, num_classes).float().to(device)
            out0, out1, out2 = model(data)
            alpha0 = relu_evidence(out0) + 1
            alpha1 = relu_evidence(out1) + 1
            alpha2 = relu_evidence(out2) + 1

            diff_01 = relu_evidence(torch.log(alpha1) - torch.log(alpha0)) + 1
            diff_12 = relu_evidence(torch.log(alpha2) - torch.log(alpha1)) + 1

            #loss_0 = mse_loss(y_onehot, alpha0, epoch, num_classes, annealing_step, device, kl=False).mean()
            loss_1 = mse_loss(y_onehot, diff_01, epoch, num_classes, annealing_step, device, var_only=False, kl=False).mean()
            loss_2 = mse_loss(y_onehot, diff_12, epoch, num_classes, annealing_step, device, var_only=False, kl=False).mean()
            loss_3 = mse_loss(y_onehot, alpha2, epoch, num_classes, annealing_step, device, var_only=False,kl=True).mean()

            #loss = 0 * loss_0 + w1 * loss_1 + w2 * loss_2 + 0 * loss_3
            loss = w1 * loss_1 + w2 * loss_2 + loss_3
            #loss = loss_3
            
            epoch_loss += loss.item()
            epoch_loss_1 += loss_1.item()
            epoch_loss_2 += loss_2.item()
            epoch_loss_3 += loss_3.item()
            
            loss.backward()
            optimizer.step()

            preds = alpha2.argmax(dim=1)
            correct += (preds == target).sum().item()
            total += target.size(0)
        
        train_acc = correct / total
        train_acc_history.append(train_acc)
        
        # 计算当前 epoch 的平均损失，并将其添加到历史记录中
        epoch_loss /= len(train_loader)
        loss_history.append(epoch_loss)
        
        epoch_loss_1 /= len(train_loader)
        loss_1_history.append(epoch_loss_1)
        
        epoch_loss_2 /= len(train_loader)
        loss_2_history.append(epoch_loss_2)
        
        epoch_loss_3 /= len(train_loader)
        loss_3_history.append(epoch_loss_3)
        
        # 验证
        model.eval()
        test_correct = 0
        for data, target in test_loader:
            out0, out1, out2 = model(data)
            preds = out2.argmax(dim=1)
            test_correct += (preds == target).sum().item()
            
        test_acc = test_correct / len(test_dataset_id)
        val_acc_history.append(test_acc)
        
        #每次都保存模型
        #torch.save(model.state_dict(), f"{result_dir}/{epoch}/model.pth")
        torch.save(model.state_dict(), f"{result_dir}/model_{epoch}.pth")
        
        # 保存最佳模型
        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            torch.save(model.state_dict(), f"{result_dir}/best_model.pth")
            

    #%% 加载最佳模型
    model.load_state_dict(torch.load(f"{result_dir}/best_model.pth"))
    model.eval()
    
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
    
    # OOD数据
    with torch.no_grad():
        for data, _ in test_loader_ood:
            out0, out1, out2 = model(data)
            uncertainties['ood']['layer0'].extend(calculate_uncertainty(out0, num_classes))
            uncertainties['ood']['layer1'].extend(calculate_uncertainty(out1, num_classes))
            uncertainties['ood']['layer2'].extend(calculate_uncertainty(out2, num_classes))
    
    # 计算AUROC（使用第二层）
    y_true = np.concatenate([np.zeros(len(uncertainties['id']['layer2'])), 
                            np.ones(len(uncertainties['ood']['layer2']))])
    y_score = np.concatenate([uncertainties['id']['layer2'], 
                             uncertainties['ood']['layer2']])
    auroc = roc_auc_score(y_true, y_score)
    
# 统一bin范围
    all_uncertainties = (
        uncertainties['id']['layer0'] + uncertainties['ood']['layer0'] +
        uncertainties['id']['layer1'] + uncertainties['ood']['layer1'] +
        uncertainties['id']['layer2'] + uncertainties['ood']['layer2']
    ) 
    min_unc = min(all_uncertainties)
    max_unc = max(all_uncertainties)
    bins = np.linspace(min_unc, max_unc, 51)  # 50个等宽bin（51个边界点）

# 第一层
    plt.subplot(3, 1, 1)
    plt.hist(uncertainties['id']['layer0'], bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(uncertainties['ood']['layer0'], bins=bins, alpha=0.5, label='OOD', density=True)
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title(f'Layer 0 Uncertainty Distribution (w1={w1}, w2={w2})')
    plt.legend()
    plt.grid(True)

# 第二层
    plt.subplot(3, 1, 2)
    plt.hist(uncertainties['id']['layer1'], bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(uncertainties['ood']['layer1'], bins=bins, alpha=0.5, label='OOD', density=True)
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title(f'Layer 1 Uncertainty Distribution')
    plt.legend()
    plt.grid(True)

# 第三层
    plt.subplot(3, 1, 3)
    plt.hist(uncertainties['id']['layer2'], bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(uncertainties['ood']['layer2'], bins=bins, alpha=0.5, label='OOD', density=True)
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title(f'Layer 2 Uncertainty Distribution (AUROC={auroc:.4f})')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.savefig(f"{result_dir}/three_layer_uncertainty.png")
    plt.close()

# 绘制原始不确定性直方图（仅第二层，改为密度图）
    plt.figure(figsize=(10, 6))
    plt.hist(uncertainties['id']['layer2'], bins=bins, alpha=0.5, label='ID', density=True)
    plt.hist(uncertainties['ood']['layer2'], bins=bins, alpha=0.5, label='OOD', density=True)
    plt.xlabel('Uncertainty')
    plt.ylabel('Density')
    plt.title(f'Uncertainty Distribution (w1={w1}, w2={w2}, AUROC={auroc:.4f})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{result_dir}/uncertainty_distribution.png")
    plt.close()

    # 绘制训练和验证准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, epochs+1), train_acc_history, label='Train Accuracy')
    plt.plot(range(1, epochs+1), val_acc_history, label='Validation Accuracy')
    plt.axvline(x=best_epoch, color='r', linestyle='--', label=f'Best Epoch: {best_epoch}')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title(f'Training History (w1={w1}, w2={w2})')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{result_dir}/training_history.png")
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
                xticklabels=[f'Class {i}' for i in id_class_indices], 
                yticklabels=[f'Class {i}' for i in id_class_indices])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title(f'Confusion Matrix (w1={w1}, w2={w2})')
    plt.savefig(f"{result_dir}/confusion_matrix.png")
    plt.close()
    
    # 保存训练历史
    history_df = pd.DataFrame({
        'epoch': range(1, epochs+1),
        'train_acc': train_acc_history,
        'val_acc': val_acc_history,
        'loss': loss_history,
        'loss_1': loss_1_history,
        'loss_2': loss_2_history,
        'loss_3': loss_3_history        
    })
    history_df.to_csv(f"{result_dir}/training_history.csv", index=False)
    
    # 保存指标
    with open(f"{result_dir}/metrics.txt", 'w') as f:
        f.write(f"Best Validation Accuracy: {best_acc:.4f}\n")
        f.write(f"Best Epoch: {best_epoch}\n")
        f.write(f"w1: {w1:.2f}\n")  # 确保保存两位小数
        f.write(f"w2: {w2:.2f}\n")  # 确保保存两位小数
        f.write(f"ID Uncertainty Mean (Layer0): {np.mean(uncertainties['id']['layer0']):.4f}\n")
        f.write(f"OOD Uncertainty Mean (Layer0): {np.mean(uncertainties['ood']['layer0']):.4f}\n")
        f.write(f"ID Uncertainty Mean (Layer1): {np.mean(uncertainties['id']['layer1']):.4f}\n")
        f.write(f"OOD Uncertainty Mean (Layer1): {np.mean(uncertainties['ood']['layer1']):.4f}\n")
        f.write(f"ID Uncertainty Mean (Layer2): {np.mean(uncertainties['id']['layer2']):.4f}\n")
        f.write(f"OOD Uncertainty Mean (Layer2): {np.mean(uncertainties['ood']['layer2']):.4f}\n")
        f.write(f"AUROC: {auroc:.4f}\n")
    
    return best_acc, auroc, np.mean(uncertainties['id']['layer2']), np.mean(uncertainties['ood']['layer2'])





#%%主函数




#%%一组参数整体数据训练与测试

# 固定 w2=1
w2 = 0.1
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

w1_values = [0.1]
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

    # 检查是否已测试过此组合
    result_dir = f"results_w3=1_ood{ood_labels}/w1_{w1}_w2_{w2}"
    if os.path.exists(f"{result_dir}/metrics.txt"):
        print(f"权重组合 w1={w1}, w2={w2} 已测试，跳过...")
        continue

    best_acc, auroc, id_unc_mean, ood_unc_mean = train_and_evaluate(w1, w2,result_dir)
    results.append({
        'w1': w1,
        'w2': w2,
        'best_accuracy': best_acc,
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
print("所有权重组合训练完成！")

# ... 其余可视化代码保持不变 ...

# ... 其余可视化代码保持不变 ...

# 绘制权重组合效果热力图
acc_matrix = results_df.pivot(index='w1', columns='w2', values='best_accuracy')
auroc_matrix = results_df.pivot(index='w1', columns='w2', values='auroc')

# 画 w1 vs accuracy/auroc 折线图
plt.figure(figsize=(10, 6))
plt.plot(results_df['w1'], results_df['best_accuracy'], marker='o', label='Accuracy')
plt.xlabel('w1')
plt.ylabel('Accuracy')
plt.title('Accuracy vs w1 (w2=1)')
plt.grid(True)
plt.savefig(f"{result_dir}/accuracy_vs_w1.png")
plt.close()

plt.figure(figsize=(10, 6))
plt.plot(results_df['w1'], results_df['auroc'], marker='o', label='AUROC')
plt.xlabel('w1')
plt.ylabel('AUROC')
plt.title('AUROC vs w1 (w2=1)')
plt.grid(True)
plt.savefig(f"{result_dir}/auroc_vs_w1.png")
plt.close()
    

