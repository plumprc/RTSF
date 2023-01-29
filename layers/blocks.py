import torch
import torch.nn as nn

class MLPBlock(nn.Module):
    def __init__(self, input_dim, mlp_dim) :
        super().__init__()
        self.fc1 = nn.Linear(input_dim, mlp_dim)
        self.gelu = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, input_dim)
    
    def forward(self, x):
        # [B, L, D] or [B, D, L]
        return self.fc2(self.gelu(self.fc1(x)))
        