import torch
import torch.nn as nn
from typing import Optional

from optrade.pytorch.utils.revin import RevIN
from optrade.pytorch.utils.weight_init import xavier_init
from optrade.pytorch.utils.utils import Reshape


class Model(nn.Module):
    def __init__(
        self,
        seq_len,
        pred_len,
        num_channels,
        norm_mode="layer",
        revin=True,
        revout=False,
        revin_affine=False,
        eps_revin=1e-5,
        channel_independent: bool = False,
        target_channels: Optional[list] = None,
    ) -> None:
        super(Model, self).__init__()

        # Normalization
        self.d_model = num_channels
        self.num_channels = num_channels
        self.target_channels = target_channels
        self.revout = revout
        self.revin_affine = revin_affine
        self.eps_revin = eps_revin
        self.channel_independent = channel_independent

        # RevIN
        if revin:
            self._init_revin()
        else:
            self._revin = None
            self.revout = None

        if channel_independent:
            self.backbone = nn.Linear(seq_len, pred_len)
        else:
            self.backbone = nn.ModuleList(
                [
                    Reshape(-1, num_channels * seq_len),
                    nn.Linear(num_channels * seq_len, num_channels * pred_len),
                    Reshape(-1, num_channels, pred_len),
                ]
            )

        if target_channels is not None:
            self.target_channels = target_channels

            if channel_independent:
                self.head = nn.Linear(pred_len, pred_len)
            else:
                self.head = nn.ModuleList(
                    [
                        Reshape(-1, len(target_channels) * pred_len),
                        nn.Linear(
                            len(target_channels) * pred_len,
                            len(target_channels) * pred_len,
                        ),
                        Reshape(-1, len(target_channels), pred_len),
                    ]
                )

        # Weight initialization
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

        if not self.channel_independent:
            for module in self.backbone:
                x = module(x)
            out = x
        else:
            out = self.backbone(x)  # (batch_size, num_channels, pred_len)

        # Head
        if self.target_channels is not None:
            out = out[
                :, self.target_channels, :
            ]  # (batch_size, len(target_channels), pred_len)
            B, C, L = out.size()

            if not self.channel_independent:
                for module in self.head:
                    out = module(out)
            else:
                out = self.head(out)
            out = out.reshape(B, C, -1)  # (batch_size, len(target_channels), pred_len)

        # RevOUT
        if self.revout:
            out = self.revin(out, mode="denorm")

        return out


if __name__ == "__main__":
    # Forecasting
    batch_size = 32
    seq_len = 512
    num_channels = 7
    pred_len = 96
    x = torch.randn(batch_size, num_channels, seq_len)

    model = Model(
        seq_len=seq_len,
        pred_len=pred_len,
        num_channels=num_channels,
        target_channels=[1, 3],
        revin=True,
        revout=True,
        revin_affine=False,
    )

    output = model(x)
    print(f"Model input: {x.shape}")
    print(f"Model output: {output.shape}")
