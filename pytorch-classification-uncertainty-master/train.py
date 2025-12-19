"""
训练模块
实现模型的训练和验证循环，支持标准分类和不确定性建模两种模式
"""
import torch
import torch.nn as nn
import copy
import time
from helpers import get_device, one_hot_embedding
from losses import relu_evidence


def train_model(
    model,
    dataloaders,
    num_classes,
    criterion,
    optimizer,
    scheduler=None,
    num_epochs=25,
    device=None,
    uncertainty=False,
):
    """
    训练模型的主函数
    
    Args:
        model: 要训练的模型（LeNet）
        dataloaders: 数据加载器字典，包含'train'和'val'两个键
        num_classes: 类别总数（MNIST为10）
        criterion: 损失函数
                   - 普通模式：nn.CrossEntropyLoss()
                   - 不确定性模式：edl_mse_loss / edl_digamma_loss / edl_log_loss
        optimizer: 优化器（Adam）
        scheduler: 学习率调度器（可选）
        num_epochs: 训练轮数
        device: 计算设备（CPU或GPU）
        uncertainty: 是否使用不确定性建模模式
    
    Returns:
        model: 训练好的模型（加载了最佳验证准确率的权重）
        metrics: 元组 (losses, accuracy)，包含训练过程中的损失和准确率记录
    """
    # 记录训练开始时间
    since = time.time()
    
    # 如果没有指定设备，自动检测
    if not device:
        device = get_device()
    
    # 保存最佳模型权重（基于验证集准确率）
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    # 用于记录训练过程的字典
    losses = {"loss": [], "phase": [], "epoch": []}      # 损失记录
    accuracy = {"accuracy": [], "phase": [], "epoch": []} # 准确率记录
    evidences = {"evidence": [], "type": [], "epoch": []} # 证据记录（未使用）
    
    # 训练循环：每个epoch
    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs - 1))
        print("-" * 10)
        
        # 每个epoch包含训练和验证两个阶段
        for phase in ["train", "val"]:
            # 设置模型模式
            if phase == "train":
                print("Training...")
                model.train()  # 训练模式：启用dropout、batch normalization更新等
            else:
                print("Validating...")
                model.eval()   # 验证模式：禁用dropout、固定batch normalization等
            
            # 用于累积统计的变量
            running_loss = 0.0      # 累积损失
            running_corrects = 0.0   # 累积正确预测数
            correct = 0              # 未使用
            
            # 遍历数据批次
            for i, (inputs, labels) in enumerate(dataloaders[phase]):
                # 将数据移到指定设备（CPU或GPU）
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                # 清零梯度（重要：每次迭代前必须清零）
                optimizer.zero_grad()
                
                # 前向传播
                # 只在训练阶段启用梯度计算（节省内存和计算）
                with torch.set_grad_enabled(phase == "train"):
                    
                    if uncertainty:
                        # ========== 不确定性模式 ==========
                        # 1. 将标签转换为one-hot编码
                        y = one_hot_embedding(labels, num_classes)
                        y = y.to(device)
                        
                        # 2. 前向传播：获取网络输出（logits）
                        outputs = model(inputs)
                        
                        # 3. 获取预测类别（最大logit对应的类别）
                        _, preds = torch.max(outputs, 1)
                        
                        # 4. 计算损失
                        # 注意：EDL损失函数需要额外的参数（epoch, num_classes等）
                        loss = criterion(
                            outputs, y.float(), epoch, num_classes, 10, device
                        )
                        
                        # 5. 计算准确率
                        match = torch.reshape(torch.eq(preds, labels).float(), (-1, 1))
                        acc = torch.mean(match)
                        
                        # 6. 计算证据和不确定性（用于分析和可视化）
                        evidence = relu_evidence(outputs)  # 证据 = ReLU(outputs)
                        alpha = evidence + 1               # Dirichlet参数
                        u = num_classes / torch.sum(alpha, dim=1, keepdim=True)  # 不确定性
                        
                        # 7. 统计证据信息（用于分析）
                        total_evidence = torch.sum(evidence, 1, keepdim=True)  # 总证据
                        mean_evidence = torch.mean(total_evidence)            # 平均证据
                        
                        # 正确预测样本的平均证据
                        mean_evidence_succ = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * match
                        ) / torch.sum(match + 1e-20)
                        
                        # 错误预测样本的平均证据
                        mean_evidence_fail = torch.sum(
                            torch.sum(evidence, 1, keepdim=True) * (1 - match)
                        ) / (torch.sum(torch.abs(1 - match)) + 1e-20)
                        
                    else:
                        # ========== 普通模式 ==========
                        # 1. 前向传播
                        outputs = model(inputs)
                        
                        # 2. 获取预测类别
                        _, preds = torch.max(outputs, 1)
                        
                        # 3. 计算损失（标准交叉熵）
                        loss = criterion(outputs, labels)
                    
                    # 反向传播和参数更新（仅在训练阶段）
                    if phase == "train":
                        loss.backward()      # 反向传播：计算梯度
                        optimizer.step()     # 更新参数
                
                # 累积统计信息
                running_loss += loss.item() * inputs.size(0)  # 损失 × 批次大小
                running_corrects += torch.sum(preds == labels.data)  # 正确预测数
            
            # 更新学习率（如果提供了调度器）
            if scheduler is not None:
                if phase == "train":
                    scheduler.step()
            
            # 计算epoch的平均损失和准确率
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            # 记录统计信息
            losses["loss"].append(epoch_loss)
            losses["phase"].append(phase)
            losses["epoch"].append(epoch)
            accuracy["accuracy"].append(epoch_acc.item())
            accuracy["epoch"].append(epoch)
            accuracy["phase"].append(phase)
            
            # 打印当前epoch的结果
            print(
                "{} loss: {:.4f} acc: {:.4f}".format(
                    phase.capitalize(), epoch_loss, epoch_acc
                )
            )
            
            # 保存最佳模型（基于验证集准确率）
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
        
        print()  # 空行分隔不同epoch
    
    # 计算总训练时间
    time_elapsed = time.time() - since
    print(
        "Training complete in {:.0f}m {:.0f}s".format(
            time_elapsed // 60, time_elapsed % 60
        )
    )
    print("Best val Acc: {:4f}".format(best_acc))
    
    # 加载最佳模型权重
    model.load_state_dict(best_model_wts)
    metrics = (losses, accuracy)
    
    return model, metrics
