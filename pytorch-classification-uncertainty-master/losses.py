"""
Evidential Deep Learning (EDL) 损失函数模块
实现了论文中的三种损失函数和KL散度正则化项

核心思想：
- 不直接输出概率，而是输出证据（evidence）
- 通过 alpha = evidence + 1 构造 Dirichlet 分布
- 使用KL散度作为正则项，防止模型过度自信
"""
import torch
import torch.nn.functional as F
from helpers import get_device


def relu_evidence(y):
    """
    使用ReLU将网络输出转换为证据（evidence）
    
    Args:
        y: 网络输出的logits，形状为 [batch_size, num_classes]
    
    Returns:
        torch.Tensor: 证据值，形状为 [batch_size, num_classes]，所有值 >= 0
    
    注意：
        ReLU确保证据非负，这是Dirichlet分布的要求
    """
    return F.relu(y)


def exp_evidence(y):
    """
    使用指数函数将网络输出转换为证据（未在代码中使用，但提供了另一种选择）
    
    Args:
        y: 网络输出的logits
    
    Returns:
        torch.Tensor: 证据值，通过exp函数转换，并限制在合理范围内
    """
    return torch.exp(torch.clamp(y, -10, 10))


def softplus_evidence(y):
    """
    使用Softplus函数将网络输出转换为证据（未在代码中使用）
    
    Args:
        y: 网络输出的logits
    
    Returns:
        torch.Tensor: 证据值，通过softplus函数转换
    """
    return F.softplus(y)


def kl_divergence(alpha, num_classes, device=None):
    """
    计算Dirichlet分布之间的KL散度
    
    计算 KL(Dir(alpha) || Dir(1,1,...,1))，即从预测的Dirichlet分布
    到均匀先验Dirichlet分布的KL散度。这作为正则项，防止模型过度自信。
    
    Args:
        alpha: Dirichlet分布的参数，形状为 [batch_size, num_classes]
               alpha = evidence + 1，每个元素 >= 1
        num_classes: 类别总数
        device: 计算设备（CPU或GPU）
    
    Returns:
        torch.Tensor: KL散度值，形状为 [batch_size, 1]
    
    数学公式：
        KL = log(Γ(S) / ∏Γ(α_k)) - log(Γ(K) / ∏Γ(1))
            + Σ(α_k - 1)[ψ(α_k) - ψ(S)]
        其中：
        - S = Σα_k（浓度参数总和）
        - Γ是Gamma函数
        - ψ是digamma函数（Gamma函数的对数导数）
        - K是类别数
    
    作用：
        - 当alpha偏离均匀先验[1,1,...,1]时，KL值增大
        - 通过梯度反向传播，推动alpha向1靠近
        - 对错误预测，降低非真实类别的证据，提高不确定性
    """
    if not device:
        device = get_device()
    
    # 创建均匀先验 Dirichlet(1, 1, ..., 1)
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    
    # 计算alpha的总和（浓度参数）
    sum_alpha = torch.sum(alpha, dim=1, keepdim=True)
    
    # 第一项：涉及Gamma函数的对数项
    first_term = (
        torch.lgamma(sum_alpha)                                    # log Γ(S)
        - torch.lgamma(alpha).sum(dim=1, keepdim=True)             # -Σ log Γ(α_k)
        + torch.lgamma(ones).sum(dim=1, keepdim=True)             # +Σ log Γ(1)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))              # -log Γ(K)
    )
    
    # 第二项：涉及digamma函数的期望项
    second_term = (
        (alpha - ones)                                              # (α_k - 1)
        .mul(torch.digamma(alpha) - torch.digamma(sum_alpha))      # × [ψ(α_k) - ψ(S)]
        .sum(dim=1, keepdim=True)
    )
    
    kl = first_term + second_term
    return kl


def loglikelihood_loss(y, alpha, device=None):
    """
    计算期望均方误差（Expected Mean Square Error）
    对应论文中的 Eq.5
    
    Args:
        y: one-hot编码的真实标签，形状为 [batch_size, num_classes]
        alpha: Dirichlet分布参数，形状为 [batch_size, num_classes]
        device: 计算设备
    
    Returns:
        torch.Tensor: 损失值，形状为 [batch_size, 1]
    
    数学公式：
        E[||y - p||²] = ||y - E[p]||² + Var[p]
        其中：
        - E[p] = α / S（Dirichlet分布的期望）
        - Var[p] = α(S-α) / (S²(S+1))（Dirichlet分布的方差）
        - S = Σα_k
    """
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    
    # 计算浓度参数总和
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # 第一项：期望误差（偏差项）
    # ||y - E[p]||²，其中 E[p] = α/S
    loglikelihood_err = torch.sum((y - (alpha / S)) ** 2, dim=1, keepdim=True)
    
    # 第二项：方差项（不确定性项）
    # Var[p] = α(S-α) / (S²(S+1))
    loglikelihood_var = torch.sum(
        alpha * (S - alpha) / (S * S * (S + 1)), dim=1, keepdim=True
    )
    
    # 总损失 = 偏差 + 方差
    loglikelihood = loglikelihood_err + loglikelihood_var
    return loglikelihood


def mse_loss(y, alpha, epoch_num, num_classes, annealing_step, device=None):
    """
    MSE型损失函数（包含KL正则化）
    对应论文中的 Eq.5 + KL正则项
    
    Args:
        y: one-hot编码的真实标签
        alpha: Dirichlet分布参数
        epoch_num: 当前训练轮数
        num_classes: 类别总数
        annealing_step: KL退火步数（通常为10）
        device: 计算设备
    
    Returns:
        torch.Tensor: 总损失 = 数据拟合项 + KL正则项
    
    退火机制：
        - 训练初期：KL权重较小，主要学习拟合数据
        - 训练后期：KL权重增大，防止过度自信
        - annealing_coef = min(1.0, epoch_num / annealing_step)
    """
    if not device:
        device = get_device()
    y = y.to(device)
    alpha = alpha.to(device)
    
    # 数据拟合项：期望均方误差
    loglikelihood = loglikelihood_loss(y, alpha, device=device)
    
    # 计算退火系数：从0线性增长到1
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    
    # 关键操作：只对非真实类别进行KL正则化
    # 对于真实类别：kl_alpha = 1（不惩罚）
    # 对于非真实类别：kl_alpha = alpha（惩罚过度证据）
    kl_alpha = (alpha - 1) * (1 - y) + 1
    
    # KL正则项
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    
    # 总损失 = 数据拟合项 + KL正则项
    return loglikelihood + kl_div


def edl_loss(func, y, alpha, epoch_num, num_classes, annealing_step, device=None):
    """
    通用的EDL损失函数
    可以用于实现Eq.3（log）和Eq.4（digamma）
    
    Args:
        func: 函数对象，可以是torch.log或torch.digamma
        y: one-hot编码的真实标签
        alpha: Dirichlet分布参数
        epoch_num: 当前训练轮数
        num_classes: 类别总数
        annealing_step: KL退火步数
        device: 计算设备
    
    Returns:
        torch.Tensor: 总损失 = 数据拟合项 + KL正则项
    
    当func=torch.log时，对应Eq.3（负对数期望似然）
    当func=torch.digamma时，对应Eq.4（期望交叉熵）
    """
    y = y.to(device)
    alpha = alpha.to(device)
    
    # 计算浓度参数总和
    S = torch.sum(alpha, dim=1, keepdim=True)
    
    # 数据拟合项：Σ y_k * [func(S) - func(α_k)]
    # 这是期望损失的核心项
    A = torch.sum(y * (func(S) - func(alpha)), dim=1, keepdim=True)
    
    # 计算退火系数
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )
    
    # 只对非真实类别进行KL正则化
    kl_alpha = (alpha - 1) * (1 - y) + 1
    
    # KL正则项
    kl_div = annealing_coef * kl_divergence(kl_alpha, num_classes, device=device)
    
    # 总损失 = 数据拟合项 + KL正则项
    return A + kl_div


def edl_mse_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    """
    EDL MSE损失函数（完整版本）
    对应论文中的 Eq.5
    
    Args:
        output: 网络原始输出（logits），形状为 [batch_size, num_classes]
        target: one-hot编码的真实标签，形状为 [batch_size, num_classes]
        epoch_num: 当前训练轮数
        num_classes: 类别总数
        annealing_step: KL退火步数（通常为10）
        device: 计算设备
    
    Returns:
        torch.Tensor: 平均损失值（标量）
    
    流程：
        1. output -> evidence (ReLU)
        2. evidence -> alpha (alpha = evidence + 1)
        3. 计算MSE损失（包含KL正则）
    """
    if not device:
        device = get_device()
    
    # 将网络输出转换为证据
    evidence = relu_evidence(output)
    
    # 计算Dirichlet参数：alpha = evidence + 1
    # 确保每个alpha >= 1（Dirichlet分布的要求）
    alpha = evidence + 1
    
    # 计算损失并取平均
    loss = torch.mean(
        mse_loss(target, alpha, epoch_num, num_classes, annealing_step, device=device)
    )
    return loss


def edl_log_loss(output, target, epoch_num, num_classes, annealing_step, device=None):
    """
    EDL Log损失函数（完整版本）
    对应论文中的 Eq.3（负对数期望似然）
    
    Args:
        output: 网络原始输出（logits）
        target: one-hot编码的真实标签
        epoch_num: 当前训练轮数
        num_classes: 类别总数
        annealing_step: KL退火步数
        device: 计算设备
    
    Returns:
        torch.Tensor: 平均损失值（标量）
    
    数学公式：
        -log E[p(y|x)] = -Σ y_k * log(α_k / S)
    """
    if not device:
        device = get_device()
    
    # 将网络输出转换为证据和alpha
    evidence = relu_evidence(output)
    alpha = evidence + 1
    
    # 使用log函数计算损失
    loss = torch.mean(
        edl_loss(
            torch.log, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss


def edl_digamma_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    """
    EDL Digamma损失函数（完整版本）
    对应论文中的 Eq.4（期望交叉熵）
    
    Args:
        output: 网络原始输出（logits）
        target: one-hot编码的真实标签
        epoch_num: 当前训练轮数
        num_classes: 类别总数
        annealing_step: KL退火步数
        device: 计算设备
    
    Returns:
        torch.Tensor: 平均损失值（标量）
    
    数学公式：
        E[-log p(y|x)] = -Σ y_k * [ψ(S) - ψ(α_k)]
        其中ψ是digamma函数
    """
    if not device:
        device = get_device()
    
    # 将网络输出转换为证据和alpha
    evidence = relu_evidence(output)
    alpha = evidence + 1
    
    # 使用digamma函数计算损失
    loss = torch.mean(
        edl_loss(
            torch.digamma, target, alpha, epoch_num, num_classes, annealing_step, device
        )
    )
    return loss
