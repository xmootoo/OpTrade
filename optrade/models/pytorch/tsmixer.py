import torch
import torch.nn as nn
from optrade.pytorch.utils.weight_init import xavier_init
from optrade.pytorch.utils.revin import RevIN
from optrade.pytorch.utils.utils import Reshape

from typing import Optional

# Taken from: https://github.com/thuml/Time-Series-Library/blob/main/models/TSMixer.py


class Model(nn.Module):
    def __init__(
        self,
        seq_len: int,
        pred_len: int,
        num_enc_layers: int,
        d_model: int,
        num_channels: int,
        dropout: float = 0.0,
        revin: bool = True,
        revin_affine: bool = True,
        revout: bool = False,
        eps_revin: float = 1e-5,
        return_head: bool = True,
        target_channels: Optional[list] = None,
        channel_independent: bool = False,
    ) -> None:
        super(Model, self).__init__()

        # Parameters
        self.num_enc_layers = num_enc_layers
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        self.revout = revout
        self.target_channels = target_channels
        self.num_channels = num_channels
        self.pred_len = pred_len
        self.seq_len = seq_len
        self.return_head = return_head

        # Layers
        self.backbone = nn.ModuleList(
            [
                ResBlock(seq_len, d_model, dropout, num_channels)
                for _ in range(num_enc_layers)
            ]
        )

        if channel_independent:
            self.head = nn.Linear(seq_len, pred_len)
        else:
            if target_channels is not None:
                num_output_channels = len(target_channels)
            else:
                num_output_channels = num_channels

            self.head = nn.Sequential(
                Reshape(-1, num_output_channels * seq_len),
                nn.Linear(
                    num_output_channels * seq_len, num_output_channels * pred_len
                ),
                Reshape(-1, num_output_channels, pred_len),
            )

        # Initialize layers
        if revin:
            self._init_revin()
        else:
            self._revin = None
            self.revout = None

        self.apply(xavier_init)

    def _init_revin(self):
        self._revin = True
        self.revin = RevIN(
            num_channels=self.num_channels,
            eps=self.eps_revin,
            affine=self.revin_affine,
            target_channels=self.target_channels,
        )

    def forward(self, x):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        x = x.permute(
            0, 2, 1
        )  # (batch_size, num_channels, seq_len) => (batch_size, seq_len, num_channels)
        for i in range(self.num_enc_layers):
            x = self.backbone[i](x)

        if self.target_channels is not None:
            x = x[:, :, self.target_channels].transpose(
                1, 2
            )  # (batch_size, len(target_channels), seq_len)
        else:
            x = x.transpose(1, 2)  # (batch_size, num_channels, seq_len)

        if self.return_head:
            out = self.head(
                x
            )  # (batch_size, seq_len, len(target_channels)) => (batch_size, len(target_channels), pred_len)
        else:
            out = x

        # RevOUT
        if self.revout:
            out = self.revin(out, mode="denorm")

        return out


class ResBlock(nn.Module):
    def __init__(
        self,
        seq_len,
        d_model,
        dropout,
        num_channels,
    ):
        super(ResBlock, self).__init__()

        self.temporal = nn.Sequential(
            nn.Linear(seq_len, d_model),
            nn.GELU(),
            nn.Linear(d_model, seq_len),
            nn.Dropout(dropout),
        )

        self.channel = nn.Sequential(
            nn.Linear(num_channels, d_model),
            nn.GELU(),
            nn.Linear(d_model, num_channels),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch_size, seq_len, num_vars)

        Returns:
            torch.Tensor: Shape (batch_size, seq_len, num_vars)
        """
        x = x + self.temporal(x.transpose(1, 2)).transpose(1, 2)
        x = x + self.channel(x)

        return x


# Test
if __name__ == "__main__":
    batch_size = 32
    seq_len = 512
    pred_len = 96
    num_channels = 7
    num_enc_layers = 3
    d_model = 16
    dropout = 0.1
    revin = True
    revin_affine = True
    revout = True
    eps_revin = 1e-5

    x = torch.rand(batch_size, num_channels, seq_len)

    model = Model(
        seq_len=seq_len,
        pred_len=pred_len,
        num_enc_layers=num_enc_layers,
        d_model=d_model,
        dropout=dropout,
        num_channels=num_channels,
        revin=revin,
        revin_affine=revin_affine,
        revout=revout,
        eps_revin=eps_revin,
        target_channels=[4, 5],
        channel_independent=True,
    )

    y = model(x)
    print(f"x: {x.shape} => y: {y.shape}")
