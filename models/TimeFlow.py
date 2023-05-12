import torch
import torch.nn as nn
from torch.nn import functional as F
from layers.Invertible import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.seq_len)
        )
        self.projection = nn.Linear(configs.seq_len, configs.pred_len)
        # self.dropout = nn.Dropout(configs.dropout)
        self.rev = RevIN(configs.channel) if configs.rev else None

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x
        x += self.temporal(x.transpose(1, 2)).transpose(1, 2)
        pred = self.projection(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred, self.forward_loss(pred, y)
        
