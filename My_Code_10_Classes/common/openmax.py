import numpy as np
import torch
import scipy.stats
import scipy.spatial

def compute_mavs(model, dataloader, num_classes, device):
    """
    计算每个类别的平均激活向量 (Mean Activation Vector, MAV)。
    
    参数:
    - model: 训练好的模型。
    - dataloader: 包含训练集数据的 DataLoader。
    - num_classes: 已知类别的数量。
    - device: 'cuda' or 'cpu'。
    
    返回:
    - mavs: (num_classes, feature_dim) 的张量，包含每个类的 MAV。
    - all_features: 包含所有样本特征的列表。
    - all_labels: 包含所有样本标签的列表。
    """
    model.eval()
    all_features = []
    all_labels = []

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            # 假设模型返回 (logits, features)
            _, features = model(inputs, return_features=True)
            all_features.append(features.cpu())
            all_labels.append(labels.cpu())

    all_features = torch.cat(all_features)
    all_labels = torch.cat(all_labels)
    
    feature_dim = all_features.shape[1]
    mavs = torch.zeros(num_classes, feature_dim)
    
    for i in range(num_classes):
        class_features = all_features[all_labels == i]
        if len(class_features) > 0:
            mavs[i] = class_features.mean(dim=0)
            
    return mavs, all_features, all_labels


# 修改 1: 增大默认 tail_size 到 40
def fit_weibull(mavs, all_features, all_labels, tail_size=40):
    """
    为每个类别拟合 Weibull 分布模型。
    tail_size: 建议设为每类样本数的 10%-30%。你每类约140个，40是个不错的选择。
    """
    num_classes = mavs.shape[0]
    weibull_models = []

    for i in range(num_classes):
        class_features = all_features[all_labels == i]
        # 如果某一类的样本太少，动态调整 tail_size
        current_tail_size = min(tail_size, len(class_features))
        
        mav = mavs[i].unsqueeze(0)
        
        # 计算特征与对应 MAV 之间的余弦距离
        distances = 1 - torch.nn.functional.cosine_similarity(class_features, mav)
        
        # 对距离进行排序并选择尾部
        sorted_distances = torch.sort(distances, descending=True).values
        tail = sorted_distances[:current_tail_size]
        
        # 拟合 Weibull 分布
        params = scipy.stats.weibull_min.fit(tail.numpy(), floc=0)
        weibull_models.append(params)
        
    return weibull_models


# 修改 2: 引入 alpha 参数实现 Top-K 校准
def compute_openmax_prob(logits, features, mavs, weibull_models, alpha=5):
    """
    参数:
        alpha (int): Top-K 校准参数，建议设为总类别数的一半左右 (如 3-5)。
                     只对得分最高的 alpha 个类别进行距离检查。
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
    计算特征与 MAV 之间的余弦距离 (Cosine Distance)。
    注意：这必须与 fit_weibull 中使用的距离度量一致。
    
    参数:
    - feature: 特征向量 (feature_dim,)。
    - mav: MAV 向量 (feature_dim,)。
    
    返回:
    - distance: 余弦距离 (1 - cosine_similarity)。
    """
    return scipy.spatial.distance.cosine(feature, mav)
