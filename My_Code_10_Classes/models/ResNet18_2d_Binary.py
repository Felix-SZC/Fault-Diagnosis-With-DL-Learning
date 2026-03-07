import torch
import torch.nn as nn
from .ResNet18_2d import ResNet, ResidualBlock

class ResNet18_Binary(ResNet):
    def __init__(self, num_classes=10, use_main_head=True):
        # 初始化父类 ResNet，使用标准的 ResNet18 结构 [2, 2, 2, 2]
        super(ResNet18_Binary, self).__init__(ResidualBlock, [2, 2, 2, 2], num_classes=num_classes)
        self.use_main_head = use_main_head

        # 定义 K 个二分类头
        # 每个头输入特征维度为 512，输出维度为 2 (属于该类/不属于该类)
        self.binary_heads = nn.ModuleList([
            nn.Linear(512, 2) for _ in range(num_classes)
        ])

    def forward(self, x):
        # 1. 提取特征
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)  # [Batch, 512]

        # 2. 二分类器输出 (K 个 [Batch, 2])
        binary_logits_list = []
        for head in self.binary_heads:
            binary_logits_list.append(head(features))

        # 3. 主分类器：由 config 的 use_main_head 决定是否启用
        if self.use_main_head:
            main_logits = self.fc(features)  # [Batch, num_classes]
            return main_logits, binary_logits_list
        return binary_logits_list

def ResNet18_Binary_Model(num_classes=10, use_main_head=True):
    return ResNet18_Binary(num_classes=num_classes, use_main_head=use_main_head)
