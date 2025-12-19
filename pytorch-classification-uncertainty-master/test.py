"""
测试模块
实现单张图片测试和旋转实验，用于评估模型性能和不确定性估计
"""
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.transforms as transforms
import os
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from losses import relu_evidence
from helpers import rotate_img, one_hot_embedding, get_device


def test_single_image(model, img_path, uncertainty=False, device=None):
    """
    测试单张图片，显示预测结果和概率分布
    
    Args:
        model: 训练好的模型
        img_path: 图片路径
        uncertainty: 是否使用不确定性模式
        device: 计算设备
    
    功能：
        - 加载图片并预处理
        - 进行预测
        - 可视化预测结果和概率分布
        - 保存结果图片到 ./results/ 目录
    """
    # 加载图片并转换为灰度图
    img = Image.open(img_path).convert("L")
    
    if not device:
        device = get_device()
    
    num_classes = 10
    
    # 图片预处理：调整大小并转换为张量
    trans = transforms.Compose([transforms.Resize((28, 28)), transforms.ToTensor()])
    img_tensor = trans(img)
    img_tensor.unsqueeze_(0)  # 添加batch维度：[1, 28, 28] -> [1, 1, 28, 28]
    img_variable = Variable(img_tensor)
    img_variable = img_variable.to(device)
    
    if uncertainty:
        # ========== 不确定性模式 ==========
        # 1. 前向传播
        output = model(img_variable)
        
        # 2. 计算证据和Dirichlet参数
        evidence = relu_evidence(output)
        alpha = evidence + 1
        
        # 3. 计算不确定性：u = K / S，其中S是alpha的总和
        # 不确定性越高，说明模型越不确定
        uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
        
        # 4. 获取预测类别
        _, preds = torch.max(output, 1)
        
        # 5. 计算概率：Dirichlet分布的期望 E[p] = alpha / S
        prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
        
        # 展平张量以便打印和可视化
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        
        # 打印结果
        print("Predict:", preds[0])
        print("Probs:", prob)
        print("Uncertainty:", uncertainty)
        
        # 保存不确定性值用于标题显示
        uncertainty_value = uncertainty.item()
    else:
        # ========== 普通模式 ==========
        # 1. 前向传播
        output = model(img_variable)
        
        # 2. 获取预测类别
        _, preds = torch.max(output, 1)
        
        # 3. 使用softmax计算概率
        prob = F.softmax(output, dim=1)
        
        # 展平张量
        output = output.flatten()
        prob = prob.flatten()
        preds = preds.flatten()
        
        # 打印结果
        print("Predict:", preds[0])
        print("Probs:", prob)
        
        # 普通模式没有不确定性
        uncertainty_value = None
    
    # 创建可视化图表
    labels = np.arange(10)  # 类别标签 0-9
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(1, 2, gridspec_kw={"width_ratios": [1, 3]})
    
    # 设置标题
    if uncertainty:
        plt.title("Classified as: {}, Uncertainty: {:.4f}".format(
            preds[0].item(), uncertainty_value))
    else:
        plt.title("Classified as: {}".format(preds[0].item()))
    
    # 左图：显示原始图片
    axs[0].set_title("Image")
    axs[0].imshow(img, cmap="gray")
    axs[0].axis("off")
    
    # 右图：显示各类别的概率分布（条形图）
    # 注意：需要先将CUDA张量移到CPU并转换为numpy
    axs[1].bar(labels, prob.cpu().detach().numpy(), width=0.5)
    axs[1].set_xlim([0, 9])
    axs[1].set_ylim([0, 1])
    axs[1].set_xticks(np.arange(10))
    axs[1].set_xlabel("Classes")
    axs[1].set_ylabel("Classification Probability")
    
    fig.tight_layout()
    
    # 保存图片
    plt.savefig("./results/{}".format(os.path.basename(img_path)))


def rotating_image_classification(
    model, img, filename, uncertainty=False, threshold=0.5, device=None
):
    """
    旋转图像分类实验
    将一张图片旋转不同角度，观察模型预测和不确定性的变化
    
    Args:
        model: 训练好的模型
        img: 输入图片张量（MNIST数字1）
        filename: 保存结果图片的文件名
        uncertainty: 是否使用不确定性模式
        threshold: 概率阈值，用于筛选显示的类别
        device: 计算设备
    
    功能：
        - 从0度到180度旋转图片（步长10度）
        - 对每个角度进行预测
        - 绘制旋转图片序列、预测类别序列、概率曲线
        - 如果启用不确定性，还绘制不确定性曲线
    """
    if not device:
        device = get_device()
    
    num_classes = 10
    Mdeg = 180      # 最大旋转角度
    Ndeg = int(Mdeg / 10) + 1  # 旋转角度数量（0, 10, 20, ..., 180）
    
    # 用于存储结果的列表
    ldeg = []           # 旋转角度列表
    lp = []              # 概率列表
    lu = []              # 不确定性列表（仅不确定性模式）
    classifications = [] # 预测类别列表
    
    # 用于筛选显示的类别（概率超过阈值的类别）
    scores = np.zeros((1, num_classes))
    
    # 用于拼接所有旋转后的图片
    rimgs = np.zeros((28, 28 * Ndeg))
    
    # 遍历每个旋转角度
    for i, deg in enumerate(np.linspace(0, Mdeg, Ndeg)):
        # 旋转图片
        nimg = rotate_img(img.numpy()[0], deg).reshape(28, 28)
        
        # 限制像素值在[0, 1]范围内
        nimg = np.clip(a=nimg, a_min=0, a_max=1)
        
        # 将旋转后的图片拼接到大图中
        rimgs[:, i * 28 : (i + 1) * 28] = nimg
        
        # 转换为张量并添加batch维度
        trans = transforms.ToTensor()
        img_tensor = trans(nimg)
        img_tensor.unsqueeze_(0)
        img_variable = Variable(img_tensor)
        img_variable = img_variable.to(device)
        
        if uncertainty:
            # ========== 不确定性模式 ==========
            # 1. 前向传播
            output = model(img_variable)
            
            # 2. 计算证据和Dirichlet参数
            evidence = relu_evidence(output)
            alpha = evidence + 1
            
            # 3. 计算不确定性
            uncertainty = num_classes / torch.sum(alpha, dim=1, keepdim=True)
            
            # 4. 获取预测类别
            _, preds = torch.max(output, 1)
            
            # 5. 计算概率
            prob = alpha / torch.sum(alpha, dim=1, keepdim=True)
            
            # 展平张量
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            
            # 记录结果
            classifications.append(preds[0].item())
            # 注意：需要先移到CPU再转换为Python数值
            # uncertainty的形状是[batch_size, 1]，需要先squeeze再转换为数值
            uncertainty_value = uncertainty.squeeze().cpu().item()
            lu.append(uncertainty_value)
        else:
            # ========== 普通模式 ==========
            # 1. 前向传播
            output = model(img_variable)
            
            # 2. 获取预测类别
            _, preds = torch.max(output, 1)
            
            # 3. 使用softmax计算概率
            prob = F.softmax(output, dim=1)
            
            # 展平张量
            output = output.flatten()
            prob = prob.flatten()
            preds = preds.flatten()
            
            # 记录结果
            classifications.append(preds[0].item())
        
        # 统计概率超过阈值的类别
        scores += prob.detach().cpu().numpy() >= threshold
        
        # 记录角度和概率
        ldeg.append(deg)
        # 注意：需要先移到CPU再转换为Python列表
        lp.append(prob.detach().cpu().tolist())
    
    # 筛选出概率超过阈值的类别（用于绘图）
    labels = np.arange(10)[scores[0].astype(bool)]
    lp = np.array(lp)[:, labels]  # 只保留筛选出的类别的概率
    
    # 绘图颜色和标记
    c = ["black", "blue", "red", "brown", "purple", "cyan"]
    marker = ["s", "^", "o"] * 2
    labels = labels.tolist()
    
    # 创建图表：3个子图（旋转图片、预测序列、概率曲线）
    fig = plt.figure(figsize=[6.2, 5])
    fig, axs = plt.subplots(3, gridspec_kw={"height_ratios": [4, 1, 12]})
    
    # 绘制各类别的概率曲线
    for i in range(len(labels)):
        axs[2].plot(ldeg, lp[:, i], marker=marker[i], c=c[i])
    
    # 如果启用不确定性，绘制不确定性曲线
    if uncertainty:
        labels += ["uncertainty"]
        # 注意：lu已经是Python列表，可以直接使用
        axs[2].plot(ldeg, lu, marker="<", c="red")
    
    # 打印预测类别序列
    print(classifications)
    
    # 上子图：显示旋转后的图片序列
    axs[0].set_title('Rotated "1" Digit Classifications')
    axs[0].imshow(1 - rimgs, cmap="gray")  # 1-rimgs用于反转颜色（黑底白字）
    axs[0].axis("off")
    plt.pause(0.001)
    
    # 中子图：显示预测类别序列（表格形式）
    empty_lst = []
    empty_lst.append(classifications)
    axs[1].table(cellText=empty_lst, bbox=[0, 1.2, 1, 1])
    axs[1].axis("off")
    
    # 下子图：概率曲线（和不确定性曲线）
    axs[2].legend(labels)
    axs[2].set_xlim([0, Mdeg])
    axs[2].set_ylim([0, 1])
    axs[2].set_xlabel("Rotation Degree")
    axs[2].set_ylabel("Classification Probability")
    
    # 保存图片
    plt.savefig(filename)
