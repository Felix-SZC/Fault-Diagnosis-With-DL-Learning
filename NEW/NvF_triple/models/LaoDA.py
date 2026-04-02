import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleMambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_ff=64):
        super(SimpleMambaBlock, self).__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_ff = d_ff

        self.input_proj = nn.Linear(d_model, d_ff)
        self.state_proj = nn.Linear(d_model, d_state)
        self.state_to_ff = nn.Linear(d_state, d_ff)
        self.update_proj = nn.Linear(d_ff, d_model)

        self.activation = nn.GELU()
        self.norm = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.input_proj.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.state_proj.weight, nonlinearity="tanh")
        nn.init.kaiming_normal_(self.state_to_ff.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.update_proj.weight, nonlinearity="relu")
        nn.init.constant_(self.input_proj.bias, 0)
        nn.init.constant_(self.state_proj.bias, 0)
        nn.init.constant_(self.state_to_ff.bias, 0)
        nn.init.constant_(self.update_proj.bias, 0)

    def forward(self, x):
        residual = x
        x_norm = self.norm(x)
        
        state = torch.tanh(self.state_proj(x_norm))
        state = self.state_to_ff(state)

        x = self.activation(self.input_proj(x_norm)) * state
        x = self.update_proj(x)
        return x + residual


class LaoDA(nn.Module):
    def __init__(self, num_classes=2):
        super(LaoDA, self).__init__()
        #input_channels = train_data.shape[1]
        
        input_channels = 64

        self.conv1 = nn.Conv1d(input_channels, 128, kernel_size=5, padding=2)
        self.bn1 = nn.BatchNorm1d(128)
        self.conv2 = nn.Conv1d(128, 256, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(256)

        self.mamba_blocks = nn.Sequential(
            SimpleMambaBlock(d_model=256),
            SimpleMambaBlock(d_model=256),
            SimpleMambaBlock(d_model=256),
        )
        self.fc1 = nn.Linear(256, 512)
        self.fc2 = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))

        x = x.transpose(1, 2)  # [B, C, L] -> [B, L, C]
        x = self.mamba_blocks(x)
        x = x.mean(dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return logits

