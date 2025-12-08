import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False), 
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.bn1(x)
        out = self.relu(out)
        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        identity_transformed = self.shortcut(identity)

        out += identity_transformed

        return out

class ResNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet, self).__init__()
        
        # Conv(4, 3, /2)
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1, bias=False)
        # RBU(4, 3, /2)
        self.rbu1 = ResidualBlock(4, 4, stride=2)
        # RBU(4, 3)
        self.rbu2 = ResidualBlock(4, 4, stride=1)
        # RBU(8, 3, /2)
        self.rbu3 = ResidualBlock(4, 8, stride=2)
        # RBU(8, 3)
        self.rbu4 = ResidualBlock(8, 8, stride=1)
        # RBU(16, 3, /2)
        self.rbu5 = ResidualBlock(8, 16, stride=2)
        # RBU(16, 3)
        self.rbu6 = ResidualBlock(16, 16, stride=1)
        # BN, ReLU, GAP
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # FC
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x, return_features=False):
        x = self.conv1(x)
        x = self.rbu1(x)
        x = self.rbu2(x)
        x = self.rbu3(x)
        x = self.rbu4(x)
        x = self.rbu5(x)
        x = self.rbu6(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.gap(x)
        features = torch.flatten(x, 1)
        logits = self.fc(features)
        
        if return_features:
            return logits, features
        else:
            return logits

def ResNet_test(num_classes=10):
    return ResNet(num_classes=num_classes)