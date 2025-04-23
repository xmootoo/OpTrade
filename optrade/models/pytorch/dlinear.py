import torch
import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.functional as F
from optrade.pytorch.utils.revin import RevIN

from typing import Optional


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    Taken from: https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py
    """

    def __init__(
        self,
        task: str = "forecasting",
        seq_len: int = 512,
        pred_len: int = 96,
        num_channels: int = 7,
        num_classes: int = 2,
        moving_avg: int = 25,
        individual: bool = False,
        return_head: bool = True,
        revin: bool = True,
        revout: bool = False,
        revin_affine: bool = False,
        eps_revin: float = 1e-5,
        target_channels: Optional[list] = None,
    ) -> None:
        """
        Args:
            task (str): Task name among 'classification', 'anomaly_detection', 'imputation', or 'forecasting'.
            seq_len (int): Length of input sequence.
            pred_len (int): Length of output forecasting.
            num_channels (int): Number of input channels (features).
            num_classes (int): Number of classes for classification task.
            moving_avg (int): Window size of moving average.
            individual (bool): Whether shared model among different variates.
        """

        super(Model, self).__init__()
        self.task = task
        self.seq_len = seq_len
        self.return_head = return_head
        self.target_channels = target_channels
        self.revout = revout
        self.revin_affine = revin_affine
        self.eps_revin = eps_revin
        self.num_channels = num_channels

        # RevIN
        if revin:
            self._init_revin()
        else:
            self._revin = None
            self.revout = None

        if self.task == "classification":
            self.pred_len = seq_len
        elif self.task == "forecasting":
            self.pred_len = pred_len
        else:
            raise ValueError(f"Task name '{self.task}' not supported.")

        # Series decomposition block from Autoformer
        self.decomposition = series_decomp(moving_avg)
        self.individual = individual
        self.channels = num_channels

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()

            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
                )
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len])
            )

        if self.task == "classification":
            self.head = nn.Linear(num_channels * seq_len, num_classes)
        elif self.task == "forecasting":
            self.target_channels = target_channels
            in_dim = (
                num_channels * pred_len
                if target_channels is None
                else pred_len * len(target_channels)
            )
            out_dim = (
                num_channels * pred_len
                if target_channels is None
                else pred_len * len(target_channels)
            )
            self.head = nn.Linear(in_dim, out_dim)

    def _init_revin(self):
        self._revin = True
        self.revin = RevIN(
            num_channels=self.num_channels,
            eps=self.eps_revin,
            affine=self.revin_affine,
            target_channels=self.target_channels,
        )

    def encoder(self, x):
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(
            0, 2, 1
        )
        if self.individual:
            seasonal_output = torch.zeros(
                [seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                dtype=seasonal_init.dtype,
            ).to(seasonal_init.device)
            trend_output = torch.zeros(
                [trend_init.size(0), trend_init.size(1), self.pred_len],
                dtype=trend_init.dtype,
            ).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :]
                )
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Ensure correct size (3D tensor)
        if len(x_enc.size()) == 2:
            x_enc = x_enc.unsqueeze(
                1
            )  # (batch_size, seq_len) -> (batch_size, 1, seq_len)
            batch_size, _, seq_len = x_enc.size()
            assert (
                seq_len == self.seq_len
            ), f"Input sequence length {seq_len} is not equal to the model sequence length {self.seq_len}."

        if self._revin:
            x_enc = self.revin(x_enc, mode="norm")

        output = self.encoder(
            x_enc.permute(0, 2, 1)
        )  # (batch_size, seq_len, num_channels)

        if self.target_channels is not None:
            output = output[:, :, self.target_channels]

        if self.return_head:
            output = output.reshape(output.shape[0], -1)
            output = self.head(output)  # (batch_size, num_channels*pred_len)
            output = output.reshape(
                output.shape[0], self.pred_len, -1
            )  # (batch_size, pred_len, num_channels)

        # RevOUT
        if self.revout:
            output = self.revin(output.permute(0, 2, 1), mode="denorm").permute(0, 2, 1)

        return output.permute(0, 2, 1)

    def classification(self, x_enc):
        # Encoder
        output = self.encoder(x_enc)  # (batch_size, seq_len, num_channels)

        if self.target_channels is not None:
            output = output[:, :, self.target_channels]

        if self.return_head:
            output = output.reshape(output.shape[0], -1)
            output = self.head(output)  # (batch_size, num_classes)
        return output

    def forward(self, x_enc):
        if self.task == "forecasting":
            output = self.forecast(x_enc)  # (batch_size, pred_len, num_channels)
        elif self.task == "classification":
            output = self.classification(
                x_enc
            )  # (batch_size, num_classes) or (batch_size,) for binary classification
        else:
            raise ValueError(f"Task name '{self.task}' not supported.")

        return output


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


# Test
if __name__ == "__main__":

    # Forecasting
    batch_size = 32
    seq_len = 512
    num_channels = 7
    task = "forecasting"
    pred_len = 96
    x = torch.randn(batch_size, num_channels, seq_len)

    forecasting_model = Model(
        task=task,
        seq_len=seq_len,
        pred_len=pred_len,
        num_channels=num_channels,
        target_channels=[1, 3],
        return_head=True,
        revin=True,
        revin_affine=True,
        revout=True,
    )

    y = forecasting_model(x)
    print(f"x: {x.shape}")
    print(f"y: {y.shape}")
