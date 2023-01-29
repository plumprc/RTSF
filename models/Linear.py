import torch
import torch.nn as nn
from layers.blocks import MLPBlock

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = nn.Linear(configs.seq_len, configs.pred_len)
        self.refine = MLPBlock(configs.pred_len, configs.d_model) if configs.refine else None

    def forward(self, x):
        # x: [B, L, D]
        x = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        x = self.refine(x.transpose(1, 2)).transpose(1, 2) if self.refine else x

        return x
        