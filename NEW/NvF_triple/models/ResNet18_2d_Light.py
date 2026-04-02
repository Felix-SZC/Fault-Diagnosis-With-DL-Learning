"""
轻量化 2D ResNet：通道数减半（32/64/128/256）、每阶段 1 个 block，参数量约为 ResNet18 的 1/4～1/8，
适用于 EDL 集成二分类等「小模型」场景，输入输出接口与 ResNet18_2d 一致。
"""
import torch
import torch.nn as nn
from .ResNet18_2d import ResNet, ResidualBlock


def ResNet18_2d_Light(num_classes=10):
    """轻量 ResNet：通道 (32, 32, 64, 128, 256)，每阶段 1 个 block，特征维 256，参数量约为 ResNet18 的 1/4～1/8。"""
    return ResNet(ResidualBlock, [1, 1, 1, 1], num_classes=num_classes, channels=(32, 32, 64, 128, 256))


def ResNet18_2d_Light_Model(num_classes=10):
    """别名，与 ResNet18_2d_Light 相同。"""
    return ResNet18_2d_Light(num_classes=num_classes)
