import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionLayer(nn.Module):
    """对时间步做加权的注意力层（作用于 BiGRU 的输出）。
    
    注意力机制输出加权后的序列（保持时间维度），
    用于后续的全局平均池化。
    """
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention_fc = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, gru_output):
        # gru_output: (B, T, H)  B=批大小, T=序列长度, H=特征维
        scores = self.attention_fc(gru_output).squeeze(-1)      # (B, T)  注意力打分
        weights = F.softmax(scores, dim=1)                      # (B, T)  归一化权重
        
        # 对每个时间步的特征进行加权（保持时间维度），而不是直接求和
        # 这样输出仍然是 (B, T, H)，可以用于全局平均池化
        weighted_seq = gru_output * weights.unsqueeze(-1)      # (B, T, H) 加权后的序列
        
        return weighted_seq, weights  # 返回加权序列和权重


class TimeFreqAttentionModel(nn.Module):
    """基于时频融合和注意力机制的深度学习模型（输入为 WPT 特征）。
    1. 两个并行的卷积分支（每个分支：卷积 → 批归一化）
    2. 双向GRU序列建模
    3. 注意力机制
    4. 全局平均池化
    5. 全连接层 + Dropout + Softmax
    """
    def __init__(self, input_channels, num_classes=10,
                 cnn_out_channels=128, gru_hidden_dim=128,
                 dropout_rate=0.5):
        super().__init__()

        # 时域一维卷积: (B, C_in, L) -> (B, C_out, L)
        self.cnn1 = nn.Conv1d(input_channels, cnn_out_channels, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(cnn_out_channels)
        self.cnn2 = nn.Conv1d(cnn_out_channels, cnn_out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(cnn_out_channels)

        # 双向 GRU：输入维度等于 CNN 输出通道数
        self.bigru = nn.GRU(input_size=cnn_out_channels,
                            hidden_size=gru_hidden_dim,
                            num_layers=2,
                            batch_first=True,
                            bidirectional=True)
        gru_output_dim = gru_hidden_dim * 2

        self.attention = AttentionLayer(gru_output_dim)
        
        self.dropout = nn.Dropout(dropout_rate)

        # 分类器
        self.fc1 = nn.Linear(gru_output_dim, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        # x: (B, C_in, L)
        # CNN特征提取
        x = F.relu(self.bn1(self.cnn1(x)))           # (B, C_out, L)
        cnn_out = F.relu(self.bn2(self.cnn2(x)))    # (B, C_out, L) 保存CNN最后批归一化输出
        
        # 转为 GRU 需要的格式并进行序列建模
        cnn_out = cnn_out.permute(0, 2, 1)                 # (B, T, C_out) 转为 GRU 需要的格式
        gru_out, _ = self.bigru(cnn_out)                   # (B, T, 2H) BiGRU输出
        
        # 注意力机制：输出加权后的序列（保持时间维度）
        attn_out, attn = self.attention(gru_out)     # (B, T, 2H) 加权序列, (B, T) 注意力权重
        
        # 全局平均池化
        # (B, T, 2H) -> (B, 2H)，对时间维度求平均
        context = torch.mean(attn_out, dim=1)  # (B, 2H)
        
        # 分类器
        x = self.fc1(context)                        # (B, 2H) -> (B, 512)
        x = self.dropout(x)
        logits = self.fc2(x)                          # (B, 512) -> (B, num_classes)
        
        return logits, attn


def create_model(input_channels, num_classes=10, **kwargs):
    return TimeFreqAttentionModel(input_channels=input_channels,
                                  num_classes=num_classes,
                                  **kwargs)