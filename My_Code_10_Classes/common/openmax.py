import numpy as np
import torch
import scipy.stats
import scipy.spatial

def compute_mavs(model, dataloader, num_classes, device):
    """
    计算每个类别的平均激活向量 (Mean Activation Vector, MAV)。
    注意：Classic OpenMax 基于 Logits (Activations) 层计算，而非中间层特征。
    
    参数:
    - model: 训练好的模型。
    - dataloader: 包含训练集数据的 DataLoader。
    - num_classes: 已知类别的数量。
    - device: 'cuda' or 'cpu'。
    
    返回:
    - mavs: (num_classes, num_classes) 的张量，包含每个类的 MAV (在 Logits 空间)。
    - all_logits: 包含所有样本 Logits 的列表。
    - all_labels: 包含所有样本标签的列表。
    """
    model.eval()
    all_logits = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # Classic OpenMax 使用 Logits 作为特征空间
            logits, _ = model(inputs, return_features=True)
            all_logits.append(logits.cpu())
            all_labels.append(labels.cpu())

    all_logits = torch.cat(all_logits)
    all_labels = torch.cat(all_labels)
    
    # Logits 维度通常等于 num_classes
    feature_dim = all_logits.shape[1]
    mavs = torch.zeros(num_classes, feature_dim)
    
    for i in range(num_classes):
        class_logits = all_logits[all_labels == i]
        if len(class_logits) > 0:
            mavs[i] = class_logits.mean(dim=0)
            
    return mavs, all_logits, all_labels


def fit_weibull(mavs, all_logits, all_labels, tail_size=10):
    """
    为每个类别拟合 Weibull 分布模型。
    tail_size: 建议设为每类样本数的 10%-30%。
    """
    num_classes = mavs.shape[0]
    weibull_models = []

    for i in range(num_classes):
        class_logits = all_logits[all_labels == i]
        # 如果某一类的样本太少，动态调整 tail_size
        current_tail_size = tail_size
        
        mav = mavs[i].unsqueeze(0)
        
        # 计算特征与对应 MAV 之间的距离 (OpenMax 经典实现通常使用欧氏距离或余弦距离)
        # 对于 Logits (Activations)，欧氏距离能保留 Logit 的幅度信息(置信度)，通常优于余弦距离
        # 这里我们修改为计算 batch 的距离
        
        # 手动计算欧氏距离以支持 batch
        # class_logits: (N, D), mav: (1, D)
        # distances: (N,)
        distances = torch.norm(class_logits - mav, p=2, dim=1)
        
        # 对距离进行排序并选择尾部
        sorted_distances = torch.sort(distances, descending=True).values
        tail = sorted_distances[:current_tail_size]
        
        # 拟合 Weibull 分布
        params = scipy.stats.weibull_min.fit(tail.numpy(), floc=0)
        weibull_models.append(params)
        
    return weibull_models


# 修改 2: 引入 alpha 参数实现 Top-K 校准
def compute_openmax_prob(logits, features, mavs, weibull_models, alpha=1):
    """
    参数:
        logits: (num_classes,) 输入的 Logits
        features: 这里的 features 应该也是 Logits (与 MAV 空间一致)
        alpha (int): Top-K 校准参数
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
