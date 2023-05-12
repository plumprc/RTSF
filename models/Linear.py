import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = nn.ModuleList([
            nn.Linear(configs.seq_len, configs.pred_len) for _ in range(configs.channel)
        ]) if configs.individual else nn.Linear(configs.seq_len, configs.pred_len)
        
        self.dropout = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.channel) if configs.rev else None
        self.individual = configs.individual

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x
        x = self.dropout(x)
        if self.individual:
            pred = torch.zeros_like(y)
            for idx, proj in enumerate(self.Linear):
                pred[:, :, idx] = proj(x[:, :, idx])
        else:
            pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred, self.forward_loss(pred, y)
