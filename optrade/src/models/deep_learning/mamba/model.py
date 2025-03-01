import torch
import torch.nn as nn
from mambapy.mamba import Mamba as MambaBackbone
from mambapy.mamba import MambaConfig
from optrade.src.models.deep_learning.utils.revin import RevIN
from optrade.src.models.deep_learning.utils.patcher import Patcher
from optrade.src.models.deep_learning.utils.weight_init import xavier_init
from optrade.src.models.deep_learning.utils.pos_enc import PositionalEncoding

class Mamba(nn.Module):
    def __init__(self, d_model,
                       num_enc_layers,
                       pred_len,
                       num_channels=1,
                       revin=False,
                       revout=False,
                       revin_affine=False,
                       eps_revin=1e-5,
                       head_type="linear",
                       norm_mode="layer",
                       patching=False,
                       patch_dim=16,
                       patch_stride=8,
                       seq_len=512,
                       last_state=True,
                       dropout=0.):
        super(Mamba, self).__init__()

        # Parameters
        self.num_patches = int((seq_len - patch_dim) / patch_stride) + 2
        self.num_channels = num_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        self.last_state = last_state

        # RevIN
        if revin:
            self._init_revin(revout, revin_affine)
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
        config = MambaConfig(d_model=d_model,
                                n_layers=num_enc_layers,
        )
        self.backbone = MambaBackbone(config)

        # Head
        head_dim = self.num_patches * d_model if patching else d_model
        if head_type=="linear":
            self.head = nn.Linear(head_dim, pred_len)
        elif head_type=="mlp":
            self.head = nn.Sequential(
                nn.Linear(head_dim, head_dim),
                nn.GELU(),
                nn.Linear(head_dim, pred_len)
            )
        self.flatten = nn.Flatten(start_dim=-2)

        # Final Normalization Layer
        self.norm = nn.LayerNorm(d_model) if norm_mode=="layer" else nn.Identity()

        # Weight initialization
        self.apply(xavier_init)

    def _init_revin(self, revout:bool, revin_affine:bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

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

        Legend:
            B: batch_size, M: num_channels, L: seq_len, N: num_patches, P: patch_dim, D: d_model.
        """

        # Ensure input is correct
        if len(x.shape) == 2:
            x = x.unsqueeze(-1) #: (B, L) -> (B, L, 1)

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm") # Patched version:(B, M, L). Non-patched version: (B, L, 1)

        # Patching
        if self._patching:
            x = self.patcher(x) # (B, M, N, P)
            x = self.pos_enc(x) # (B, M, N, D)
            B, M, N, D = x.shape
            x = x.view(B*M, N, D)

        # Mamba forward pass
        x = self.backbone(x) # Patched version: (B*M, N, D). Non-patched version: (B, L, D)

        # Normalization
        x = self.norm(x) # Patched version: (B*M, N, D). Non-patched version: (B, L, D)

        # Apply head
        if self.last_state and not self._patching:
            x = x[:, -1, :] # Patched-version: (B*M, D). Non-patched version: (B, D)
        elif self._patching:
            x = x.view(B, M, N*D) # (B, M, N*D)
        else:
            pass

        # Head
        x = self.head(x)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return x


if __name__=="__main__":
    # # <---Non-patched version (classification)--->
    # # Define model parameters
    # batch_size = 32
    # input_size = 1  # for univariate time series
    # d_model = 64
    # num_enc_layers = 5
    # pred_len = 1
    # seq_len = 512
    # num_channels = 1

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
    #     head_type="linear",
    #     patching=False,
    #     last_state=True,
    # ).to(device)

    # # Create sample input data
    # true_seq_len = 600
    # x = torch.randn(batch_size, true_seq_len, input_size).to(device)  # (B, L_true, input_dim)

    # # Pass the data through the model
    # output = model(x)
    # output = output.to(device)

    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {output.shape}")


    #<--Patched Version (forecasting)--->
    # Define model parameters
    patch_dim = 16
    patch_stride = 8
    batch_size = 32
    d_model = 128
    input_size = d_model
    num_enc_layers = 5
    pred_len = 96
    seq_len = 512
    num_channels = 7

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
        revin_affine=True,
        revout=True,
        head_type="linear",
        patching=True,
        last_state=False,
    ).to(device)

    # Create sample input data
    x = torch.randn(batch_size, num_channels, seq_len).to(device)  # (B, M, L)

    # Pass the data through the model
    output = model(x)
    output = output.to(device)

    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
