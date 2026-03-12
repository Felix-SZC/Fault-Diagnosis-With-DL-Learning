import numpy as np
from scipy.io import loadmat
import os

# ———————— 设置随机数种子并初始化生成器 ————————
seed = 1
rng = np.random.default_rng(seed)

# ———————— 读取 .mat 文件（如有需要，请修改路径） ————————
mat = loadmat('coefAll.mat')
data = mat['coefAll']    # 数据形状应为 (10800, 64, 64)

# ———————— 参数设置 ————————
n_classes = 6            # 类别总数
samples_per_class = 1200 # 每个类别的样本数
test_per_class = 200     # 每个类别测试集样本数

# ———————— 划分索引列表初始化 ————————
train_indices = []
test_indices = []

# ———————— 按类别随机抽样生成训练/测试索引 ————————
for cls in range(n_classes):
    start = cls * samples_per_class
    end = start + samples_per_class
    class_idx = np.arange(start, end)
    
    # 打乱并分割
    perm = rng.permutation(samples_per_class)
    test_idx = class_idx[perm[:test_per_class]]
    train_idx = class_idx[perm[test_per_class:]]
    
    test_indices.append(test_idx)
    train_indices.append(train_idx)

# ———————— 合并所有类别的索引 ————————
test_indices = np.concatenate(test_indices)
train_indices = np.concatenate(train_indices)

# ———————— 根据索引提取训练/测试集 ————————
X_train = data[train_indices]
X_test  = data[test_indices]

# ———————— 定义文字标签列表并生成对应标签向量 ————————
class_labels = np.array([
    '1正常', '2二级小齿轮点蚀', '3二级后轴承保持架', '4二级后轴承滚动体',
    '5二级后轴承磨损', '6行星轴承滚珠磨损'
])
y_full = np.repeat(class_labels, samples_per_class)
y_train = y_full[train_indices]
y_test  = y_full[test_indices]

# ———————— 创建以随机种子为名的输出文件夹 ————————
output_dir = str(seed)
os.makedirs(output_dir, exist_ok=True)

# ———————— 保存数据和文字标签为 .npy 文件 ————————
np.save(os.path.join(output_dir, 'X_train.npy'), X_train)
np.save(os.path.join(output_dir, 'X_test.npy'),  X_test)
np.save(os.path.join(output_dir, 'y_train.npy'), y_train)
np.save(os.path.join(output_dir, 'y_test.npy'),  y_test)

print(f"已将训练集和测试集及文字标签保存到文件夹 '{output_dir}'，使用随机种子：{seed}")

#[64,64]第一个64是行数，表示频带特征，第二个64是列数，表示时间