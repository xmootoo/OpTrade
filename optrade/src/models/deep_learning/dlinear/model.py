import torch
import torch.nn as nn
import torch.nn.functional as F
from optrade.src.models.deep_learning.dlinear.series_decomp import series_decomp


class DLinear(nn.Module):
    """
    Paper link: https://arxiv.org/pdf/2205.13504.pdf
    Taken from: https://github.com/thuml/Time-Series-Library/blob/main/models/DLinear.py
    """

    def __init__(
        self,
        task="forecasting",
        seq_len=512,
        pred_len=96,
        num_channels=7,
        num_classes=2,
        moving_avg=25,
        individual=False,
        return_head=True):

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

        super(DLinear, self).__init__()
        self.task = task
        self.seq_len = seq_len
        self.return_head = return_head
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
                self.Linear_Seasonal.append(
                    nn.Linear(self.seq_len, self.pred_len))
                self.Linear_Trend.append(
                    nn.Linear(self.seq_len, self.pred_len))

                self.Linear_Seasonal[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
                self.Linear_Trend[i].weight = nn.Parameter(
                    (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.pred_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.pred_len)

            self.Linear_Seasonal.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))
            self.Linear_Trend.weight = nn.Parameter(
                (1 / self.seq_len) * torch.ones([self.pred_len, self.seq_len]))

        if self.task == 'classification':
            self.projection = nn.Linear(
                num_channels * seq_len, num_classes)

    def encoder(self, x):
        x = x.permute(0, 2, 1) # (batch_size, num_channels, seq_len) -> (batch_size, seq_len, num_channels)
        seasonal_init, trend_init = self.decomposition(x)
        seasonal_init, trend_init = seasonal_init.permute(
            0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.pred_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.pred_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](
                    seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](
                    trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)
        x = seasonal_output + trend_output
        return x.permute(0, 2, 1)

    def forecast(self, x_enc):
        # Ensure correct size (3D tensor)
        if len(x_enc.size()) == 2:
            x_enc = x_enc.unsqueeze(1)  # (batch_size, seq_len) -> (batch_size, 1, seq_len)
            batch_size, _, seq_len = x_enc.size()
            assert seq_len == self.seq_len, f"Input sequence length {seq_len} is not equal to the model sequence length {self.seq_len}."

        # Encoder
        return self.encoder(x_enc)

    def classification(self, x_enc):
        # Encoder
        output = self.encoder(x_enc) # (batch_size, seq_len, num_channels)

        if self.return_head:
            output = output.reshape(output.shape[0], -1)
            output = self.projection(output) # (batch_size, num_classes)
        return output

    def forward(self, x_enc):
        if self.task == "forecasting":
            dec_out = self.forecast(x_enc)
            return dec_out[:, -self.pred_len:, :].permute(0, 2, 1)  # (batch_size, num_channels, pred_len)
        if self.task == "classification":
            dec_out = self.classification(x_enc) # (batch_size, num_classes) or (batch_size,) for binary classification
            return dec_out
        return None


# Test
if __name__ == '__main__':

    # Forecasting
    batch_size = 32
    seq_len = 512
    num_channels = 7
    task="forecasting"
    pred_len = 96
    x = torch.randn(batch_size, num_channels, seq_len)

    forecasting_model = DLinear(task=task,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    num_channels=num_channels,)

    y = forecasting_model(x)
    print(f"x: {x.shape}")
    print(f"y: {y.shape}")

    # Classification
    batch_size = 32
    seq_len = 24
    num_channels = 1
    task = "classification"
    pred_len = -1
    num_classes = 1

    x = torch.randn(batch_size, num_channels, seq_len)

    classification_model = DLinear(task=task,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    num_classes=num_classes,
                    num_channels=num_channels,
                    moving_avg=13)

    y = classification_model(x)

    print(f"x: {x.shape}")
    print(f"y: {y.shape}")
