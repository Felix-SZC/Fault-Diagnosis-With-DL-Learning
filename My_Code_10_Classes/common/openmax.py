import numpy as np
import torch
import scipy.stats
import scipy.spatial

def compute_mavs(model, dataloader, num_classes, device, source='features'):
    """
    计算每个类别的平均激活向量 (Mean Activation Vector, MAV)。
    
    参数:
    - model: 训练好的模型。
    - dataloader: 包含训练集数据的 DataLoader。
    - num_classes: 已知类别的数量。
    - device: 'cuda' or 'cpu'。
    - source: 'features' 或 'logits'，指定计算 MAV 的数据源。
    
    返回:
    - mavs: (num_classes, data_dim) 的张量，包含每个类的 MAV。
    - all_data: 包含所有样本源数据的张量 (features 或 logits)。
    - all_labels: 包含所有样本标签的列表。
    """
    model.eval()
    all_data = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            if source == 'logits':
                logits, _ = model(inputs, return_features=True)
                all_data.append(logits.cpu())
            else: # 默认为 'features'
                _, features = model(inputs, return_features=True)
                all_data.append(features.cpu())

            all_labels.append(labels.cpu())

    all_data = torch.cat(all_data)
    all_labels = torch.cat(all_labels)
    
    data_dim = all_data.shape[1]
    mavs = torch.zeros(num_classes, data_dim)
    
    for i in range(num_classes):
        class_data = all_data[all_labels == i]
        if len(class_data) > 0:
            mavs[i] = class_data.mean(dim=0)
            
    return mavs, all_data, all_labels


def fit_weibull(mavs, all_data, all_labels, tail_size=10):
    """
    为每个类别拟合 Weibull 分布模型。
    
    参数:
    - mavs: (num_classes, data_dim) 的 MAV 张量
    - all_data: (N, data_dim) 的所有样本源数据张量 (features 或 logits)
    - all_labels: (N,) 的所有样本标签
    - tail_size: 建议设为每类样本数的 10%-30%。
    """
    num_classes = mavs.shape[0]
    weibull_models = []

    for i in range(num_classes):
        class_data = all_data[all_labels == i]
        # 如果某一类的样本太少，动态调整 tail_size
        current_tail_size = min(tail_size, len(class_data))
        
        mav = mavs[i].unsqueeze(0)
        
        # 计算特征与对应 MAV 之间的距离
        # 使用欧氏距离
        distances = torch.norm(class_data - mav, p=2, dim=1)
        
        # 对距离进行排序并选择尾部
        sorted_distances = torch.sort(distances, descending=True).values
        tail = sorted_distances[:current_tail_size]
        
        # 拟合 Weibull 分布
        params = scipy.stats.weibull_min.fit(tail.numpy(), floc=0)
        weibull_models.append(params)
        
    return weibull_models


# 修改：使用 Deep Features 进行 OpenMax 计算
def compute_openmax_prob(features, logits, mavs, weibull_models, alpha=1):
    """
    参数:
        features: (data_dim,) 用于计算距离的数据 (features 或 logits)。
        logits: (num_classes,) 输入的原始 Logits（用于计算概率）。
        mavs: (num_classes, data_dim) MAV 矩阵。
        weibull_models: Weibull 分布参数列表。
        alpha (int): Top-K 校准参数。
    """
    num_classes = logits.shape[0]
    distances = np.zeros(num_classes)
    
    # 0. 找到 Logits 最大的 Top-K 个类别的索引
    # argsort 是从小到大，取最后 alpha 个即为最大的 alpha 个
    # [::-1] 是为了反转，让最大的在前面（虽然顺序不影响计算，但方便调试）
    ranked_list = logits.argsort().ravel()
    top_k_indices = ranked_list[-alpha:][::-1]
    
    # 1. & 2. 仅对 Top-K 类别计算距离和校准
    # 其他类别的 alphap 默认为 1.0 (即不进行衰减/惩罚)
    alphap = np.ones(num_classes) 
    
    for i in top_k_indices:
        # 只计算 Top-K 的距离
        distances[i] = compute_distance(features, mavs[i])
        
        weibull_params = weibull_models[i]
        shape, loc, scale = weibull_params
        
        # 计算校准系数 w_score
        w_score = 1 - scipy.stats.weibull_min.cdf(distances[i], shape, loc=loc, scale=scale)
        alphap[i] = w_score

    # 处理可能的 NaN
    alphap[np.isnan(alphap)] = 1.0

    # 计算修正后的激活值
    # 只有 Top-K 的 logits 会被 alphap 衰减，其他保持不变
    rev_activations = alphap * logits
    
    # 3. 计算 "未知" 类别的激活值
    # 仅当 Top-K 类别且 Logit 为正时，收集其被抑制的能量
    corrections = logits - rev_activations
    unknown_activation = np.sum(corrections[logits > 0])
    
    # 4. 计算 OpenMax 概率
    total_denominator = np.sum(np.exp(rev_activations)) + np.exp(unknown_activation)
    
    if total_denominator == 0 or np.isinf(total_denominator):
        return np.exp(logits) / np.sum(np.exp(logits))

    openmax_prob = np.concatenate([np.exp(rev_activations) / total_denominator, 
                                   [np.exp(unknown_activation) / total_denominator]])
                                   
    return openmax_prob

def compute_distance(feature, mav):
    """
    计算特征与 MAV 之间的欧氏距离 (Euclidean Distance)。
    注意：这必须与 fit_weibull 中使用的距离度量一致。
    
    参数:
    - feature: 特征向量 (feature_dim,)。
    - mav: MAV 向量 (feature_dim,)。
    
    返回:
    - distance: 欧氏距离。
    """
    return scipy.spatial.distance.euclidean(feature, mav)
