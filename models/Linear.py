import torch
import torch.nn as nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.Linear = nn.Linear(configs.seq_len, configs.pred_len)

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D]
        pred = self.Linear(x.transpose(1, 2)).transpose(1, 2)

        return pred, self.forward_loss(pred, y)
