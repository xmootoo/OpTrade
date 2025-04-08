import torch
import torch.nn as nn

from typing import Optional, List


class RevIN(nn.Module):
    def __init__(
        self,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        target_channels: Optional[List[int]] = None,
    ) -> None:
        """
        Kim et al. (2022): Reversible instance normalization for accurate time-series forecasting against
        distribution shift. Provides a learnable instance normalization layer that is reversible. Code is
        from https://github.com/yuqinie98/PatchTST/blob/main/PatchTST_supervised/layers/RevIN.py with modifications,
        which was originally taken from https://github.com/ts-kim/RevIN.

        Args:
            num_channels: The number of features or channels.
            eps: A value added for numerical stability.
            affine: If True, RevIN has learnable affine parameters (e.g., like LayerNorm).
            target_channels: List of target channels for the head layer. It not None, it will denormalize only the target channels corresponding
                             to an input shape of (batch_size, len(target_channels), pred_len) for the model output.
        """

        super(RevIN, self).__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if target_channels is not None:
            self.target_channels_idx = target_channels
            self.affine_weight = nn.Parameter(torch.ones(len(target_channels)))
            self.affine_bias = nn.Parameter(torch.zeros(len(target_channels)))
        else:
            self.target_channels_idx = None
            self.affine_weight = nn.Parameter(torch.ones(self.num_channels))
            self.affine_bias = nn.Parameter(torch.zeros(self.num_channels))

    def forward(self, x, mode: str):
        """
        Forward pass for normalization or denormalizating the time series with learnable affine transformations.

        Args:
            x: Input tensor of shape (batch_size, num_channels, seq_len).
            mode: 'norm' for normalization and 'denorm' for denormalization.
        Returns:
            Normalized or denormalized tensor of same shape as input tensor.
        """

        if x.dim() != 3:
            x = x.unsqueeze(1)

        if mode == "norm":
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == "denorm":
            x = self._denormalize(x)
        else:
            raise NotImplementedError
        return x

    def _get_statistics(self, x):
        self.mean = torch.mean(x, dim=2, keepdim=True).detach()
        self.stdev = torch.sqrt(
            torch.var(x, dim=2, keepdim=True, unbiased=False) + self.eps
        ).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            if self.target_channels_idx is not None:
                x[:, self.target_channels_idx, :] = x[
                    :, self.target_channels_idx, :
                ] * self.affine_weight.unsqueeze(0).unsqueeze(
                    -1
                )  # Reshape: (1, num_channels, 1)
                x[:, self.target_channels_idx, :] = x[
                    :, self.target_channels_idx, :
                ] + self.affine_bias.unsqueeze(0).unsqueeze(
                    -1
                )  # Reshape: (1, num_channels, 1)
            else:
                x = x * self.affine_weight.unsqueeze(0).unsqueeze(
                    -1
                )  # Reshape: (1, num_channels, 1)
                x = x + self.affine_bias.unsqueeze(0).unsqueeze(
                    -1
                )  # Reshape: (1, num_channels, 1)
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias.unsqueeze(0).unsqueeze(-1)
            x = x / (
                self.affine_weight.unsqueeze(0).unsqueeze(-1) + self.eps * self.eps
            )

        if self.target_channels_idx is not None:
            x = x * self.stdev[:, self.target_channels_idx, :]
            x = x + self.mean[:, self.target_channels_idx, :]
        else:
            x = x * self.stdev
            x = x + self.mean

        return x


if __name__ == "__main__":
    batch_size = 32
    num_channels = 7
    seq_len = 100
    pred_len = 48
    target_channels = [1, 3]
    x = torch.randn(batch_size, num_channels, seq_len)
    revin = RevIN(
        num_channels=num_channels,
        eps=1e-5,
        affine=True,
        target_channels=target_channels,
    )
    x_norm = revin(x, mode="norm")

    print(f"x_norm shape: {x_norm.shape}")

    y_hat = torch.randn(batch_size, len(target_channels), pred_len)

    x_denorm = revin(y_hat, mode="denorm")

    print(f"x_denorm shape: {x_denorm.shape}")
