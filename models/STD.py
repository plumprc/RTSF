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
    def __init__(self, configs):
        super(Model, self).__init__()

        # the size of sliding window in moving average
        kernel_size = 25
        self.decompsition = series_decomp(kernel_size)
        self.seasonal = nn.Linear(configs.seq_len, configs.pred_len)
        self.trend = nn.Sequential(
            nn.Linear(configs.seq_len, configs.d_model),
            nn.ReLU(),
            nn.Linear(configs.d_model, configs.pred_len)
        )
        self.rev = RevIN(configs.channel) if configs.rev else None

    def forward_loss(self, pred, true):
        return F.mse_loss(pred, true)

    def forward(self, x, y):
        # x: [B, L, D]
        seasonality, trend = self.decompsition(x)
        trend = self.rev(trend, 'norm') if self.rev else trend
        seasonality = self.seasonal(seasonality.transpose(1, 2)).seasonality(1, 2)
        trend = self.trend(trend.transpose(1, 2)).transpose(1, 2)
        trend = self.rev(trend, 'denorm') if self.ref else trend
        pred = seasonality + trend
        
        return pred, self.forward_loss(pred, y)
