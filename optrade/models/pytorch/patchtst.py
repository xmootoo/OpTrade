import torch
import torch.nn as nn
from torch import Tensor

from optrade.pytorch.utils.utils import Norm
from optrade.pytorch.utils.utils import Reshape
from optrade.pytorch.utils.revin import RevIN
from optrade.pytorch.utils.patcher import Patcher
from optrade.pytorch.utils.pos_enc import PositionalEncoding
from optrade.pytorch.utils.weight_init import xavier_init

from typing import Optional, List


class Model(nn.Module):
    def __init__(
        self,
        num_enc_layers: int,
        d_model: int,
        d_ff: int,
        num_heads: int,
        input_channels: List[str],
        seq_len: int,
        pred_len: int,
        attn_dropout: float = 0.0,
        ff_dropout: float = 0.0,
        pred_dropout: float = 0.0,
        batch_first: bool = True,
        norm_mode: str = "batch1d",
        revin: bool = True,
        revout: bool = True,
        revin_affine: bool = True,
        eps_revin: float = 1e-5,
        patch_dim: int = 16,
        stride: int = 1,
        return_head: bool = True,
        head_type: str = "linear",
        channel_independent: bool = False,  # Head only
        target_channels: Optional[list] = None,  # Head only
    ) -> None:
        super(Model, self).__init__()

        # Parameters
        self.num_patches = int((seq_len - patch_dim) / stride) + 2
        self.num_channels = len(input_channels)
        self.input_channels = input_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        self.target_channels = target_channels
        self.revout = revout

        # Initialize layers
        if revin:
            self._init_revin()
        else:
            self._revin = None
            self.revout = None

        self.patcher = Patcher(patch_dim, stride)
        self.pos_enc = PositionalEncoding(patch_dim, d_model, self.num_patches)
        self.backbone = PatchTSTBackbone(
            input_channels=input_channels,
            num_enc_layers=num_enc_layers,
            d_model=d_model,
            d_ff=d_ff,
            num_heads=num_heads,
            num_channels=self.num_channels,
            num_patches=self.num_patches,
            pred_len=pred_len,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            pred_dropout=pred_dropout,
            batch_first=batch_first,
            norm_mode=norm_mode,
            return_head=return_head,
            head_type=head_type,
            channel_independent=channel_independent,
            target_channels=target_channels,
        )

        # Weight initialization
        self.apply(xavier_init)

    def _init_revin(self):
        self._revin = True
        self.revin = RevIN(
            num_channels=self.num_channels,
            input_channels=self.input_channels,
            eps=self.eps_revin,
            affine=self.revin_affine,
            target_channels=self.target_channels,
        )

    def forward(self, x, y=None, ch_ids=None):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        # Patcher
        x = self.patcher(x)

        # Project + Positional Encoding
        x = self.pos_enc(x)

        # Transformer + Linear Head
        x = self.backbone(x, y, ch_ids)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return x


class PatchTSTBackbone(nn.Module):
    def __init__(
        self,
        num_enc_layers,
        d_model,
        d_ff,
        num_heads,
        num_channels,
        num_patches,
        pred_len,
        input_channels,
        attn_dropout=0.0,
        ff_dropout=0.0,
        pred_dropout=0.0,
        batch_first=True,
        norm_mode="batch1d",
        return_head=True,
        head_type="linear",
        channel_independent=False,
        target_channels: Optional[list] = None,
    ):
        super(PatchTSTBackbone, self).__init__()

        # Parameters
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.d_model = d_model
        self.return_head = return_head
        self.pred_len = pred_len
        self.target_channels = target_channels
        if target_channels is not None:
            self.target_channels_idx = [
                input_channels.index(ch) for ch in target_channels
            ]

        # Encoder
        self.enc = nn.Sequential(
            *(
                EncoderBlock(
                    d_model,
                    d_ff,
                    num_heads,
                    num_channels,
                    num_patches,
                    attn_dropout,
                    ff_dropout,
                    batch_first,
                    norm_mode,
                )
                for i in range(num_enc_layers)
            )
        )

        # Prediction head
        self.flatten = nn.Flatten(start_dim=-2)
        num_output_channels = (
            len(target_channels) if target_channels is not None else num_channels
        )
        if head_type == "linear":
            if channel_independent:
                self.head = nn.Linear(num_patches * d_model, pred_len)
            elif target_channels is not None or not channel_independent:
                self.head = nn.Sequential(
                    Reshape(-1, num_output_channels * num_patches * d_model),
                    nn.Linear(
                        num_output_channels * num_patches * d_model,
                        num_output_channels * pred_len,
                    ),
                    Reshape(-1, num_output_channels, pred_len),
                )
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(num_patches * d_model, num_patches * d_model),
                nn.GELU(),
                nn.Dropout(pred_dropout),
                nn.Linear(num_patches * d_model, pred_len),
            )
        else:
            raise ValueError(f"Invalid head type: {head_type}")

    def forward(
        self, x: torch.Tensor, y: torch.Tensor = None, ch_ids: torch.Tensor = None
    ) -> torch.Tensor:

        # Encoding
        batch_size = x.shape[0]
        x = x.view(
            batch_size * self.num_channels, self.num_patches, -1
        )  # (batch_size * num_channels, num_patches, d_model)
        x = self.enc(x)  # (batch_size * num_channels, num_patches, d_model)

        if self.return_head:
            x = x.reshape(
                batch_size, self.num_channels, self.num_patches * self.d_model
            )

            if self.target_channels_idx is not None:
                x = x[:, self.target_channels_idx, :]

            out = self.head(x)
        else:
            out = x.view(batch_size, self.num_channels, self.num_patches, -1)

        return out


class SupervisedHead(nn.Module):
    def __init__(self, linear_dim, pred_len, dropout=0.0):
        super().__init__()
        """
        Flattens and applies a linear layer to each channel independently to form a prediction.
        Args:
            num_channels (int): The number of channels in the input.
            linear_dim (int): The dimension of the linear layer, should be num_patches * d_model.
            pred_len (int): The length of the forecast window.
            dropout (float): The dropout value.
        """

        self.dropout = nn.Dropout(dropout)
        self.linear = nn.Linear(linear_dim, pred_len)

    def forward(self, x) -> torch.Tensor:
        """
        Applies a linear layer to each channel independently to form a prediction, optional dropout.
        Args:
            x (torch.Tensor): The input of shape (batch_size, num_channels, num_patches, d_model)
        Returns:
            x (torch.Tensor): The output of shape (batch_size, num_channels, pred_len).
        """
        x = self.linear(x)
        x = self.dropout(x)
        return x


class EncoderBlock(nn.Module):
    """
    Args:
        d_model: The embedding dimension.
        num_heads: The number of heads in the multi-head attention models.
        dropout: The dropout value.
        batch_first: If True, then the input and output tensors are provided as (batch, seq, feature).
        norm: The type of normalization to use. Either "batch1d", "batch2d", or "layer".
    """

    def __init__(
        self,
        d_model,
        d_ff,
        num_heads,
        num_channels,
        num_patches,
        attn_dropout=0.0,
        ff_dropout=0.0,
        batch_first=True,
        norm_mode="batch1d",
    ):
        super(EncoderBlock, self).__init__()

        # Layers
        self.attn = _MultiheadAttention(
            num_heads=num_heads,
            d_model=d_model,
            dropout=attn_dropout,
            batch_first=batch_first,
        )
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(ff_dropout),
            nn.Linear(d_ff, d_model),
        )

        # Normalization
        self.norm = Norm(norm_mode, num_channels, num_patches, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, num_patches, d_model).
        Returns:
            fc_out: Output of the transformer block, a tensor of shape (batch_size, num_patches, d_model).
        """

        # Multihead Attention -> Add & Norm
        attn_out, _ = self.attn(x, x, x)
        attn_norm = self.norm(
            attn_out + x
        )  # Treat the input as the query, key and value for MHA.

        # Feedforward layer -> Add & Norm
        fc_out = self.ff(attn_norm)
        fc_norm = self.norm(fc_out + attn_out)

        return fc_norm


class _MultiheadAttention(nn.Module):
    """
    Multihead Attention mechanism from the Vanilla Transformer, with some preset parameters for the PatchTST model.
    """

    def __init__(self, num_heads: int, d_model: int, dropout=0.0, batch_first=True):
        super(_MultiheadAttention, self).__init__()

        # Layers
        self.attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=batch_first,
        )

    def forward(
        self, Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            Q: Query embedding of shape: (batch_size, num_patches, d_model).
            K: Key embedding of shape (batch_size, num_patches, d_model).
            V: Value embedding of shape (batch_size, num_patches, d_model).
            batch_size: The batch size.
            num_patches: The sequence length.
            d_model: The embedding dimension.
        Returns:
            x: The output of the attention layer of shape (batch_size, num_patches, d_model).
        """
        return self.attn(query=Q, key=K, value=V, need_weights=False)


if __name__ == "__main__":
    batch_size = 64
    num_channels = 7
    seq_len = 512
    patch_dim = 16
    patch_stride = 8
    pred_len = 96

    model = Model(
        num_enc_layers=3,
        d_model=128,
        d_ff=512,
        num_heads=4,
        num_channels=num_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        attn_dropout=0.0,
        ff_dropout=0.0,
        pred_dropout=0.0,
        batch_first=True,
        norm_mode="batch1d",
        revin=True,
        revout=False,
        revin_affine=True,
        eps_revin=1e-5,
        patch_dim=patch_dim,
        stride=patch_stride,
        return_head=True,
        head_type="linear",
        channel_independent=False,
        target_channels=[1, 3, 0],
    )

    x = torch.randn(batch_size, num_channels, seq_len)  # (B, M, L)
    out = model(x)
    print(f"Input shape: {x.shape}, Output shape: {out.shape}")
