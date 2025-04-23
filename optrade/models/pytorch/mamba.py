import torch
import torch.nn as nn
from mambapy.mamba import Mamba as MambaBackbone
from mambapy.mamba import MambaConfig
from optrade.pytorch.utils.revin import RevIN
from optrade.pytorch.utils.patcher import Patcher
from optrade.pytorch.utils.weight_init import xavier_init
from optrade.pytorch.utils.pos_enc import PositionalEncoding
from optrade.pytorch.utils.utils import Reshape

from typing import Optional


class Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        num_enc_layers,
        pred_len,
        num_channels: int = 1,
        revin: bool = False,
        revout: bool = False,
        revin_affine: bool = False,
        eps_revin: float = 1e-5,
        head_type: str = "linear",
        norm_mode: str = "layer",
        patching: bool = False,
        patch_dim: int = 16,
        patch_stride: int = 8,
        seq_len: int = 512,
        last_state: bool = True,
        dropout: float = 0.0,
        channel_independent: bool = False,
        target_channels: Optional[list] = None,
    ) -> None:
        super(Mamba, self).__init__()

        # Parameters
        self.num_patches = int((seq_len - patch_dim) / patch_stride) + 2
        self.num_channels = num_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        self.revout = revout
        self.target_channels = target_channels
        self.last_state = last_state

        # Initialize layers
        if revin:
            self._init_revin()
        else:
            self._revin = None
            self.revout = None

        # Patching (only works for FIXED sequence length)
        if patching:
            self._patching = True
            self.patcher = Patcher(patch_dim, patch_stride)
            self.pos_enc = PositionalEncoding(patch_dim, d_model, self.num_patches)
        else:
            self._patching = None

        # Mamba Backbone
        config = MambaConfig(
            d_model=d_model,
            n_layers=num_enc_layers,
        )
        self.backbone = MambaBackbone(config)

        # Head
        # head_dim = self.num_patches * d_model if patching else d_model
        num_output_channels = (
            len(target_channels) if target_channels is not None else num_channels
        )
        if head_type == "linear":
            if channel_independent:
                self.head = nn.Linear(seq_len, pred_len)
            elif target_channels is not None or not channel_independent:
                self.head = nn.Sequential(
                    Reshape(-1, num_output_channels * seq_len),
                    nn.Linear(
                        num_output_channels * seq_len, num_output_channels * pred_len
                    ),
                    Reshape(-1, num_output_channels, pred_len),
                )
                print("Selecting correct head")
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(head_dim, head_dim), nn.GELU(), nn.Linear(head_dim, pred_len)
            )

        # Final Normalization Layer
        self.norm = nn.LayerNorm(d_model) if norm_mode == "layer" else nn.Identity()

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
        """
        Computes the forward pass of the Mamba model. There are two possible modes:

            Patched Version: This is meant for univariate or multivariate time series forecasting, which applies a
            patching mechanism to the input sequence. The input tensor should have shape (B, M, L), where B is the batch size, M
            is the number of channels, and L is the sequence length. The output tensor will have shape (B, pred_len), where
            pred_len is the prediction length.

            Non-Patched Version: This is meant for univariate variable-length time series classification (SOZ localization), where the input
            tensor should have shape (B, L, 1), where B is the batch size, and L is the sequence length which can change from batch to batch,
            and is padded accordingly. The output tensor will have shape (B, pred_len), where pred_len is the prediction length (usually set to
            pred_len=1 for binary classification).

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, seq_len, num_channels)

        Legend:
            B: batch_size, M: num_channels, L: seq_len, N: num_patches, P: patch_dim, D: d_model.
        """

        if not self._patching:
            x = x.transpose(1, 2)  # (B, M, L) -> (B, L, M)

        # RevIN
        if self._revin:
            x = self.revin(
                x, mode="norm"
            )  # Patched version:(B, M, L). Non-patched version: (B, L, M)

        # Patching
        if self._patching:
            x = self.patcher(x)  # (B, M, N, P)
            x = self.pos_enc(x)  # (B, M, N, D)
            B, M, N, D = x.shape
            x = x.view(B * M, N, D)

        # Mamba forward pass
        print(f"Shape before backbone: {x.shape}")
        x = self.backbone(
            x
        )  # Patched version: (B*M, N, D). Non-patched version: (B, L, M)

        # Normalization
        x = self.norm(x)  # Patched version: (B*M, N, D). Non-patched version: (B, L, M)

        # Apply head
        if self.last_state and not self._patching:
            x = (
                x[:, :, self.target_channels] if self.target_channels is not None else x
            )  # (B, L, num_output_channels)
            x = x.transpose(1, 2)  # (B, num_output_channels, L)
        elif self._patching:
            x = x.view(B, M, N * D)  # (B, M, N*D)
        else:
            raise ValueError(
                "Invalid configuration for the Mamba model. Please check the parameters."
            )

        # Head
        print(f"Shape before head: {x.shape}")
        print(f"Parameter sizes of head: {[p.size() for p in self.head.parameters()]}")
        x = self.head(
            x
        )  # Patched version: ? Non-patched version: (B, pred_len, num_output_channels)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        # if not self._patching:
        #     x = x.transpose(1, 2) # (B, num_output_channels, pred_len)

        return x


if __name__ == "__main__":
    # <---Non-patched version--->
    # Define model parameters
    batch_size = 32
    num_enc_layers = 2
    pred_len = 96
    seq_len = 512
    num_channels = d_model = 7

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create an instance of the LSTM model
    model = Mamba(
        d_model=d_model,
        num_enc_layers=num_enc_layers,
        pred_len=pred_len,
        seq_len=seq_len,
        num_channels=num_channels,
        revin=True,
        head_type="linear",
        patching=False,
        last_state=True,
        channel_independent=False,
        target_channels=[0, 3, 5],
    ).to(device)

    x = torch.randn(batch_size, num_channels, seq_len).to(device)

    # Pass the data through the model
    output = model(x)
    output = output.to(device)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # #<--Patched Version (forecasting)--->
    # # Define model parameters
    # patch_dim = 16
    # patch_stride = 8
    # batch_size = 32
    # d_model = 128
    # input_size = d_model
    # num_enc_layers = 5
    # pred_len = 96
    # seq_len = 512
    # num_channels = 7

    # # Device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # # Create an instance of the LSTM model
    # model = Mamba(
    #     d_model=d_model,
    #     num_enc_layers=num_enc_layers,
    #     pred_len=pred_len,
    #     seq_len=seq_len,
    #     num_channels=num_channels,
    #     revin=True,
    #     revin_affine=True,
    #     revout=True,
    #     head_type="linear",
    #     patching=True,
    #     last_state=False,
    #     target_channels=None,
    # ).to(device)

    # # Create sample input data
    # x = torch.randn(batch_size, num_channels, seq_len).to(device)  # (B, M, L)

    # # Pass the data through the model
    # output = model(x)
    # output = output.to(device)

    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {output.shape}")
