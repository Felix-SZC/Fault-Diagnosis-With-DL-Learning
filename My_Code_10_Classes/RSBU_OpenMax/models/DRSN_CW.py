import torch
import torch.nn as nn

class SoftThresholding(nn.Module):
    """
    Shrinkage Sub-network (子网络)
    对应论文 Fig. 4 中的 'Absolute' -> 'GAP' -> 'FC' -> 'BN' -> 'Sigmoid' -> 'Soft Thresholding'
    """
    def __init__(self, channels):
        super(SoftThresholding, self).__init__()
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Linear(channels, channels),
            nn.Sigmoid(),
        )
    def forward(self, x):
        x_abs = torch.abs(x)

        # shape: (Batch, Channel, 1)
        abs_mean = self.gap(x_abs)

        # shape: (Batch, Channel)
        abs_mean_flat = abs_mean.view(x.size(0), -1)
        alpha = self.fc(abs_mean_flat)
        
        # shape: (Batch, Channel, 1)
        alpha = alpha.view(x.size(0), x.size(1), 1)
        tau = alpha * abs_mean
        
        return torch.sign(x) * torch.relu(x_abs - tau)

class RSBU_CW(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(RSBU_CW, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)

        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

        self.shrinkage  = SoftThresholding(out_channels)

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

        out = self.shrinkage(out)
  
        out += identity_transformed

        return out

class DRSN_CW(nn.Module):
    def __init__(self, num_classes=10):
        super(DRSN_CW, self).__init__()
        
        # Conv(4, 3, /2)
        self.conv1 = nn.Conv1d(1, 4, kernel_size=3, stride=2, padding=1, bias=False)
        # rsbu(4, 3, /2)
        self.rsbu1 = RSBU_CW(4, 4, stride=2)
        # rsbu(4, 3)
        self.rsbu2 = RSBU_CW(4, 4, stride=1)
        # rsbu(8, 3, /2)
        self.rsbu3 = RSBU_CW(4, 8, stride=2)
        # rsbu(8, 3)
        self.rsbu4 = RSBU_CW(8, 8, stride=1)
        # rsbu(16, 3, /2)
        self.rsbu5 = RSBU_CW(8, 16, stride=2)
        # rsbu(16, 3)
        self.rsbu6 = RSBU_CW(16, 16, stride=1)
        # BN, ReLU, GAP
        self.bn = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.gap = nn.AdaptiveAvgPool1d(1)
        # FC
        self.fc = nn.Linear(16, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.rsbu1(x)
        x = self.rsbu2(x)
        x = self.rsbu3(x)
        x = self.rsbu4(x)
        x = self.rsbu5(x)
        x = self.rsbu6(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

def DRSN_CW_test(num_classes=10):
    return DRSN_CW(num_classes=num_classes)