"""
主程序入口
提供命令行接口，支持训练、测试、查看示例三种模式

使用方法：
    # 训练普通模型
    python main.py --train --dropout --epochs 10
    
    # 训练不确定性模型（MSE损失）
    python main.py --train --dropout --uncertainty --mse --epochs 50
    
    # 测试模型
    python main.py --test --uncertainty --mse
    
    # 查看示例数据
    python main.py --examples
"""
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision.datasets.mnist import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np
import argparse
from matplotlib import pyplot as plt
from PIL import Image

from helpers import get_device, rotate_img, one_hot_embedding
from data import dataloaders, digit_one
from train import train_model
from test import rotating_image_classification, test_single_image
from losses import edl_mse_loss, edl_digamma_loss, edl_log_loss, relu_evidence
from lenet import LeNet


def main():
    """
    主函数：解析命令行参数并执行相应操作
    """
    # ========== 命令行参数解析 ==========
    parser = argparse.ArgumentParser(description="Evidential Deep Learning for MNIST")
    
    # 互斥模式组：必须选择一种模式（训练/测试/示例）
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--train", action="store_true", help="训练网络"
    )
    mode_group.add_argument(
        "--test", action="store_true", help="测试网络"
    )
    mode_group.add_argument(
        "--examples", action="store_true", help="查看MNIST示例数据"
    )
    
    # 训练相关参数
    parser.add_argument(
        "--epochs", default=10, type=int, help="训练轮数"
    )
    parser.add_argument(
        "--dropout", action="store_true", help="是否使用dropout正则化"
    )
    parser.add_argument(
        "--uncertainty", action="store_true", help="是否使用不确定性建模模式"
    )
    
    # 不确定性损失函数选择（互斥，三选一）
    uncertainty_type_group = parser.add_mutually_exclusive_group()
    uncertainty_type_group.add_argument(
        "--mse",
        action="store_true",
        help="使用不确定性时，选择MSE损失函数（论文Eq.5）",
    )
    uncertainty_type_group.add_argument(
        "--digamma",
        action="store_true",
        help="使用不确定性时，选择Digamma损失函数（论文Eq.4）",
    )
    uncertainty_type_group.add_argument(
        "--log",
        action="store_true",
        help="使用不确定性时，选择Log损失函数（论文Eq.3）",
    )
    
    args = parser.parse_args()
    
    # ========== 模式1：查看示例数据 ==========
    if args.examples:
        # 从验证集中获取一个批次的数据
        examples = enumerate(dataloaders["val"])
        batch_idx, (example_data, example_targets) = next(examples)
        
        # 创建图表显示6张示例图片
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2, 3, i + 1)
            plt.tight_layout()
            plt.imshow(example_data[i][0], cmap="gray", interpolation="none")
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        
        # 保存图片
        plt.savefig("./images/examples.jpg")
        print("示例图片已保存到 ./images/examples.jpg")
    
    # ========== 模式2：训练模型 ==========
    elif args.train:
        # 设置训练参数
        num_epochs = args.epochs
        use_uncertainty = args.uncertainty
        num_classes = 10  # MNIST有10个类别
        
        # 创建模型
        model = LeNet(dropout=args.dropout)
        
        # 选择损失函数
        if use_uncertainty:
            # 不确定性模式：必须选择一种损失函数
            if args.digamma:
                criterion = edl_digamma_loss  # 期望交叉熵（Eq.4）
            elif args.log:
                criterion = edl_log_loss      # 负对数期望似然（Eq.3）
            elif args.mse:
                criterion = edl_mse_loss       # 期望均方误差（Eq.5）
            else:
                parser.error("使用 --uncertainty 时必须指定 --mse、--log 或 --digamma 之一")
        else:
            # 普通模式：使用标准交叉熵损失
            criterion = nn.CrossEntropyLoss()
        
        # 创建优化器：Adam优化器，学习率1e-3，权重衰减0.005
        optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=0.005)
        
        # 创建学习率调度器：每7个epoch将学习率乘以0.1
        exp_lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)
        
        # 获取计算设备并显示
        device = get_device()
        print(f"训练设备: {device}")
        model = model.to(device)
        
        # 开始训练
        model, metrics = train_model(
            model,
            dataloaders,
            num_classes,
            criterion,
            optimizer,
            scheduler=exp_lr_scheduler,
            num_epochs=num_epochs,
            device=device,
            uncertainty=use_uncertainty,
        )
        
        # 保存模型状态
        state = {
            "epoch": num_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        }
        
        # 根据模式选择保存路径
        if use_uncertainty:
            if args.digamma:
                torch.save(state, "./results/model_uncertainty_digamma.pt")
                print("模型已保存: ./results/model_uncertainty_digamma.pt")
            if args.log:
                torch.save(state, "./results/model_uncertainty_log.pt")
                print("模型已保存: ./results/model_uncertainty_log.pt")
            if args.mse:
                torch.save(state, "./results/model_uncertainty_mse.pt")
                print("模型已保存: ./results/model_uncertainty_mse.pt")
        else:
            torch.save(state, "./results/model.pt")
            print("模型已保存: ./results/model.pt")
    
    # ========== 模式3：测试模型 ==========
    elif args.test:
        use_uncertainty = args.uncertainty
        device = get_device()
        
        # 创建模型（结构必须与训练时一致）
        model = LeNet()
        model = model.to(device)
        optimizer = optim.Adam(model.parameters())  # 仅用于加载状态，实际不使用
        
        # 根据模式加载对应的模型权重
        if use_uncertainty:
            if args.digamma:
                checkpoint = torch.load(
                    "./results/model_uncertainty_digamma.pt",
                    weights_only=False  # 允许加载完整checkpoint（包含优化器状态）
                )
                filename = "./results/rotate_uncertainty_digamma.jpg"
            if args.log:
                checkpoint = torch.load(
                    "./results/model_uncertainty_log.pt",
                    weights_only=False
                )
                filename = "./results/rotate_uncertainty_log.jpg"
            if args.mse:
                checkpoint = torch.load(
                    "./results/model_uncertainty_mse.pt",
                    weights_only=False
                )
                filename = "./results/rotate_uncertainty_mse.jpg"
        else:
            checkpoint = torch.load(
                "./results/model.pt",
                weights_only=False
            )
            filename = "./results/rotate.jpg"
        
        # 加载模型权重
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        # 设置为评估模式（禁用dropout等）
        model.eval()
        
        # 执行旋转实验：旋转数字1，观察不同角度下的预测
        rotating_image_classification(
            model, digit_one, filename, uncertainty=use_uncertainty
        )
        print(f"旋转实验结果已保存: {filename}")
        
        # 测试单张图片：MNIST数字1
        test_single_image(model, "./data/one.jpg", uncertainty=use_uncertainty)
        print("单图测试结果已保存: ./results/one.jpg")
        
        # 测试单张图片：Yoda图片（分布外样本，用于测试不确定性）
        test_single_image(model, "./data/yoda.jpg", uncertainty=use_uncertainty)
        print("单图测试结果已保存: ./results/yoda.jpg")


if __name__ == "__main__":
    main()
