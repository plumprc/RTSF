import torch
import torch.nn as nn
from torch.nn import functional as F
from layers.Invertible import RevIN

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.transport = nn.Parameter(torch.zeros(configs.seq_len, configs.pred_len))
        self.bias = nn.Parameter(torch.zeros(configs.pred_len, configs.channel))
        # self.bias = nn.Sequential(
        #    nn.Linear(configs.seq_len, configs.d_model),
        #    nn.GELU(),
        #    nn.Linear(configs.d_model, configs.pred_len)
        # )
        self.dropout = nn.Dropout(configs.drop)
        self.rev = RevIN(configs.channel) if configs.rev else None

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true) # + 0.1 * torch.norm(self.transport)

    def forward(self, x, y):
        # x: [B, L, D]
        x = self.rev(x, 'norm') if self.rev else x
        # x = torch.einsum('bij,ik->bkj', x, self.transport)
        self.transport = self.dropout(self.transport)
        pred = torch.matmul(self.transport.T, x)
        # y += self.bias(x.transpose(1, 2)).transpose(1, 2)
        pred += self.bias
        pred = self.rev(pred, 'denorm') if self.rev else pred

        return pred, self.forward_loss(pred, y)
        