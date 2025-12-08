import numpy as np
import torch
import scipy.stats

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


def fit_weibull(mavs, all_features, all_labels, tail_size=20):
    """
    为每个类别拟合 Weibull 分布模型。
    
    参数:
    - mavs: (num_classes, feature_dim) 的 MAV 张量。
    - all_features: 所有训练样本的特征。
    - all_labels: 所有训练样本的标签。
    - tail_size: 用于拟合 Weibull 分布的尾部大小。
    
    返回:
    - weibull_models: 包含每个类别 Weibull 模型参数的列表。
    """
    num_classes = mavs.shape[0]
    weibull_models = []

    for i in range(num_classes):
        class_features = all_features[all_labels == i]
        mav = mavs[i].unsqueeze(0)
        
        # 计算特征与对应 MAV 之间的余弦距离
        distances = 1 - torch.nn.functional.cosine_similarity(class_features, mav)
        
        # 对距离进行排序并选择尾部
        sorted_distances = torch.sort(distances, descending=True).values
        tail = sorted_distances[:tail_size]
        
        # 拟合 Weibull 分布
        params = scipy.stats.weibull_min.fit(tail.numpy(), floc=0)
        weibull_models.append(params)
        
    return weibull_models


def compute_openmax_prob(scores, features, mavs, weibull_models):
    """
    计算给定样本的 OpenMax 概率。
    
    参数:
    - scores: 模型的原始 logits (batch_size, num_classes)。
    - features: 模型的特征向量 (batch_size, feature_dim)。
    - mavs: MAV 张量。
    - weibull_models: Weibull 模型列表。
    
    返回:
    - openmax_probs: (batch_size, num_classes + 1) 的 OpenMax 概率。
    """
    num_classes = mavs.shape[0]
    
    # 计算特征与所有 MAV 的欧氏距离
    distances = torch.cdist(features, mavs, p=2)
    
    # 使用 Weibull CDF 评估距离
    w_scores = torch.zeros_like(scores)
    for i in range(num_classes):
        shape, loc, scale = weibull_models[i]
        # scipy.stats.weibull_min.cdf 返回 numpy 数组，需要转换为 torch 张量
        w_scores[:, i] = torch.from_numpy(
            scipy.stats.weibull_min.cdf(distances[:, i].numpy(), shape, loc, scale)
        ).float()
    
    # 重新校准激活向量
    revised_scores = scores * (1 - w_scores)
    
    # 计算 "未知" 类别的分数
    unknown_scores = torch.sum(scores * w_scores, dim=1)
    
    # 组合成 OpenMax 激活向量
    openmax_scores = torch.cat([revised_scores, unknown_scores.unsqueeze(1)], dim=1)
    
    # 应用 Softmax 得到最终概率
    openmax_probs = torch.nn.functional.softmax(openmax_scores, dim=1)
    
    return openmax_probs
