import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.transport = nn.Parameter(torch.zeros(configs.seq_len, configs.pred_len))
        self.bias = nn.Parameter(torch.zeros(configs.pred_len, configs.channel))

    def forward(self, x):
        # x: [B, L, D]
        # x = torch.einsum('bij,ik->bkj', x, self.transport)
        x = torch.matmul(self.transport.T, x)
        x += self.bias

        return x
        