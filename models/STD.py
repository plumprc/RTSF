import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.Invertible import RevIN

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):
    """
    Decomposition-Linear
    """
    def __init__(self, configs):
        super(Model, self).__init__()

        # Decompsition Kernel Size
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.Linear_Seasonal = nn.Linear(configs.seq_len, configs.pred_len)
        self.Linear_Trend = nn.Linear(configs.seq_len, configs.pred_len)
        self.rev = RevIN(configs.channel)

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [Batch, Input length, Channel]
        seasonal, trend = self.decompsition(x)
        
        trend = self.rev(trend, 'norm')
        seasonal_output = self.Linear_Seasonal(seasonal.transpose(1, 2)).transpose(1, 2)
        trend_output = self.Linear_Trend(trend.transpose(1, 2)).transpose(1, 2)
        trend_output = self.rev(trend_output, 'denorm')
        pred = seasonal_output + trend_output
        
        return pred, self.forward_loss(pred, y)
