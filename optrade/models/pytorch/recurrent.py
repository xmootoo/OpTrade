import torch
import torch.nn as nn

from typing import Optional

from optrade.pytorch.utils.revin import RevIN
from optrade.pytorch.utils.patcher import Patcher
from optrade.pytorch.utils.pos_enc import PositionalEncoding
from optrade.pytorch.utils.weight_init import xavier_init
from optrade.pytorch.utils.utils import Reshape


# TODO: Reimplement patching
class Model(nn.Module):
    def __init__(
        self,
        d_model,
        num_enc_layers,
        pred_len,
        backbone_id,
        bidirectional=False,
        dropout=0.0,
        seq_len=512,
        patching=False,
        patch_dim=16,
        patch_stride=8,
        num_channels=1,
        head_type="linear",
        norm_mode="layer",
        revin=False,
        revout=False,
        revin_affine=False,
        eps_revin=1e-5,
        last_state=True,
        avg_state=False,
        return_head=True,
        channel_independent=False,
        target_channels: Optional[list] = None,
    ) -> None:
        super(Model, self).__init__()

        """
        A Recurrent Neural Network (RNN) class that host a variety of different recurrent architectures including LSTM, Mamba, GRU, and the classic RNN.

        Args:
            d_model (int): The number of expected features in the input (required).
            num_enc_layers (int): Number of recurrent layers (required).
            pred_len (int): The number of expected features in the output (required).
            backbone_id (str): The type of recurrent architecture to use (required). Options: "LSTM", "Mamba",
            bidirectional (bool): If True, becomes a bidirectional RNN. Default: False.
            dropout (float): If non-zero, introduces a Dropout layer on the outputs of each RNN layer except the last layer, with dropout probability equal to dropout. Default: 0.
            seq_len (int): The length of the input sequence. Default: 512.
            patching (bool): If True, the input sequence is patched. Default: False.
            patch_dim (int): The dimension of the patch. Default: 16.
            patch_stride (int): The stride of the patch. Default: 8.
            num_channels (int): The number of channels in the input data. Default: 1.
            head_type (str): The type of head to use Options: "linear", "mlp". Default: "linear".
            norm_mode (str): The type of normalization to use. Default: "layer".
            revin (bool): If True, applies RevIN to the input sequence. Default: False.
            revout (bool): If True, applies RevIN to the output sequence. Default: False.
            revin_affine (bool): If True, applies an affine transformation to the RevIN layer. Default: False.
            eps_revin (float): The epsilon value for RevIN. Default: 1e-5.
            last_state (bool): If True, returns the last state of the RNN. Default: True.
            avg_state (bool): If True, returns the average state of the RNN. Default: False.
        """

        # Parameters
        self.backbone_id = backbone_id
        self.num_patches = int((seq_len - patch_dim) / patch_stride) + 2
        self.patch_dim = patch_dim
        self.patch_stride = patch_stride
        self.num_channels = num_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        self.revout = revout
        self.target_channels = target_channels
        self.last_state = last_state
        self.avg_state = avg_state
        self.input_size = d_model if patching else num_channels
        self.return_head = return_head

        # RevIN
        if revin:
            self._init_revin()
        else:
            self._revin = None
            self.revout = None

        # Patching (only works for FIXED sequence length)
        if patching:
            self._patching = True
            self.patcher = Patcher(patch_dim, patch_stride)
            self.pos_enc = (
                nn.Linear(patch_dim, d_model)
                if avg_state
                else PositionalEncoding(patch_dim, d_model, self.num_patches)
            )
        else:
            self._patching = None

        # Backbone
        if self.backbone_id == "LSTM":
            self.backbone = nn.LSTM(
                input_size=self.input_size,
                hidden_size=d_model,
                num_layers=num_enc_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif self.backbone_id == "RNN":
            self.backbone = nn.RNN(
                input_size=self.input_size,
                hidden_size=d_model,
                num_layers=num_enc_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        elif self.backbone_id == "GRU":
            self.backbone = nn.GRU(
                input_size=self.input_size,
                hidden_size=d_model,
                num_layers=num_enc_layers,
                batch_first=True,
                dropout=dropout,
                bidirectional=bidirectional,
            )
        else:
            raise ValueError("Invalid backbone_id. Options: 'LSTM', 'RNN', 'GRU'.")

        # Head
        self.dropout = nn.Dropout(dropout)

        if patching and not avg_state:
            head_dim = self.num_patches * d_model
        elif last_state or avg_state:
            head_dim = d_model

        num_output_channels = (
            len(target_channels) if target_channels is not None else num_channels
        )
        if head_type == "linear":
            self.head = nn.Sequential(
                nn.Linear(head_dim, num_output_channels * pred_len),
                Reshape(-1, num_output_channels, pred_len),
            )
        elif head_type == "mlp":
            self.head = nn.Sequential(
                nn.Linear(head_dim, head_dim // 2),
                nn.GELU(),
                nn.Linear(head_dim // 2, num_output_channels * pred_len),
                Reshape(-1, num_output_channels, pred_len),
            )

        if not (last_state or avg_state):
            self.head = nn.Sequential(
                Reshape(-1, seq_len * d_model),
                self.head,
            )
        self.flatten = nn.Flatten(start_dim=-2)

        # Final Normalization Layer
        norm_dim = d_model
        self.norm = nn.LayerNorm(norm_dim) if norm_mode == "layer" else nn.Identity()

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

    def compute_backbone(self, x):
        if self.backbone_id in {"RNN", "GRU"}:
            out, hn = self.backbone(x)
            last_hn = hn[-1]
        elif self.backbone_id == "LSTM":
            out, (hn, _) = self.backbone(x)
            last_hn = hn[-1]
        else:
            raise ValueError("Invalid backbone_id. Options: 'LSTM', 'RNN', 'GRU',.")

        return out, last_hn

    def forward(self, x):
        """
        Forward pass of the model.

        Args:
            x (torch.Tensor): The input data. Shape: (batch_size, num_channels, seq_len) = (B, M, L).
        """

        # Ensure input is correct
        if len(x.shape) == 2:
            x = (
                x.unsqueeze(-2) if self._patching else x.unsqueeze(-1)
            )  #: (B, L) -> (B, L, 1)

        # RevIN
        if self._revin:
            x = self.revin(
                x, mode="norm"
            )  # Patched version:(B, M, L). Non-patched version: (B, L, 1)

        x = x.transpose(1, 2)

        # # Patching
        # if self._patching:
        #     x = self.patcher(x) # (B, M, N, P)
        #     x = self.pos_enc(x) # (B, M, N, D)
        #     B, M, N, D = x.shape
        #     x = x.view(B*M, N, D) # (B*M, N, D)

        # Backbone forward pass
        out, last_hn = self.compute_backbone(x)

        # Normalization
        if self.last_state:
            x = self.norm(last_hn)  # Select last hidden state: (B, D)
        elif self.avg_state:
            x = self.norm(
                torch.mean(out, dim=1)
            )  # Average over sequence length. Patched version: (B*M, D). Non-patched version: (B, D).
        else:
            x = self.norm(
                out
            )  # Patched version: (B*M, N, D). Non-patched version: (B, L, D)

        # # Reshape for patching
        # if self._patching:
        #     x = x.view(B, M, -1) # avg state: (B, M, D). Non-avg state: (B, M, N*D)

        # Head
        if self.return_head:
            # x = x.transpose(0,1)
            print(f"Shape before head: {x.shape}")
            x = self.head(self.dropout(x))  # (B, pred_len)

        print(f"x after head: {x.shape}")
        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return x


if __name__ == "__main__":
    # <---Non-patched version (classification)--->
    # Define model parameters
    batch_size = 32
    num_channels = 7
    seq_len = 512
    pred_len = 96
    d_model = 64
    num_enc_layers = 5

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.randn(batch_size, num_channels, seq_len).to(device)  # (B, M, L)

    model = Model(
        d_model=d_model,
        num_enc_layers=num_enc_layers,
        pred_len=pred_len,
        backbone_id="GRU",
        bidirectional=False,
        dropout=0.1,
        seq_len=seq_len,
        patching=False,
        # patch_dim=16,
        # patch_stride=8,
        num_channels=num_channels,
        head_type="linear",
        norm_mode="layer",
        revin=True,
        revout=True,
        revin_affine=True,
        last_state=False,
        avg_state=True,
        return_head=True,
        target_channels=[0, 3, 5],
    ).to(device)

    # Pass the data through the model
    output = model(x)
    output = output.to(device)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")

    # #<--Patched Version (forecasting)--->
    # # Define model parameters
    # patch_dim = 64
    # patch_stride = 16
    # batch_size = 1
    # d_model = 128
    # num_enc_layers = 5
    # pred_len = 1
    # seq_len = 16031
    # num_channels = 1

    # # Device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print(f"Using device: {device}")

    # # Create an instance of the LSTM model
    # model = RecurrentModel(
    #     d_model=d_model,
    #     backbone_id="Mamba",
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
    #     avg_state=True,
    # ).to(device)

    # # Create sample input data
    # x = torch.randn(batch_size, num_channels, seq_len).to(device)  # (B, M, L)
    # print(f"Input shape: {x.shape}")

    # # Pass the data through the model
    # output = model(x)
    # output = output.to(device)

    # print(f"Output shape: {output.shape}")
