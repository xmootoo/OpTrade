import torch
import torch.nn as nn

from mambapy.mamba import Mamba as MambaBackbone
from mambapy.mamba import MambaConfig

from optrade.src.models.deep_learning.utils.revin import RevIN
from optrade.src.models.deep_learning.utils.patcher import Patcher
from optrade.src.models.deep_learning.utils.pos_enc import PositionalEncoding
from optrade.src.models.deep_learning.utils.weight_init import xavier_init

class RecurrentModel(nn.Module):
    def __init__(self, d_model,
                       num_enc_layers,
                       pred_len,
                       backbone_id,
                       bidirectional=False,
                       dropout=0.,
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
                       return_head=True):
        super(RecurrentModel, self).__init__()

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
        self.last_state = last_state
        self.avg_state = avg_state
        self.input_size = d_model if patching else num_channels
        self.return_head = return_head

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
            self.pos_enc = nn.Linear(patch_dim, d_model) if avg_state else PositionalEncoding(patch_dim, d_model, self.num_patches)
        else:
            self._patching = None

        # Backbone
        if self.backbone_id=="LSTM":
            self.backbone = nn.LSTM(input_size=self.input_size,
                                hidden_size=d_model,
                                num_layers=num_enc_layers,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=bidirectional)
        elif self.backbone_id=="RNN":
            self.backbone = nn.RNN(input_size=self.input_size,
                                hidden_size=d_model,
                                num_layers=num_enc_layers,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=bidirectional)
        elif self.backbone_id=="GRU":
            self.backbone = nn.GRU(input_size=self.input_size,
                                hidden_size=d_model,
                                num_layers=num_enc_layers,
                                batch_first=True,
                                dropout=dropout,
                                bidirectional=bidirectional)
        elif self.backbone_id=="Mamba":
            config = MambaConfig(
                d_model=d_model,
                n_layers=num_enc_layers,
            )
            self.backbone = MambaBackbone(config)
        else:
            raise ValueError("Invalid backbone_id. Options: 'LSTM', 'RNN', 'GRU', 'Mamba'.")

        # Head
        self.dropout = nn.Dropout(dropout)

        if patching and not avg_state:
            head_dim = self.num_patches * d_model
        else:
            head_dim = d_model

        head_dim = 2*head_dim if bidirectional else head_dim

        if head_type=="linear":
            self.head = nn.Linear(head_dim, pred_len)
        elif head_type=="mlp":
            self.head = nn.Sequential(
                nn.Linear(head_dim, head_dim//2),
                nn.GELU(),
                nn.Linear(head_dim//2, pred_len)
            )
        self.flatten = nn.Flatten(start_dim=-2)

        # Final Normalization Layer
        norm_dim = 2*d_model if bidirectional else d_model
        self.norm = nn.LayerNorm(norm_dim) if norm_mode=="layer" else nn.Identity()

        # Weight initialization
        self.apply(xavier_init)

    def _init_revin(self, revout:bool, revin_affine:bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

    def compute_backbone(self, x):
        if self.backbone_id in {"RNN", "GRU"}:
            out, hn = self.backbone(x)
            last_hn = hn[-1]
        elif self.backbone_id =="LSTM":
            out, (hn, _) = self.backbone(x)
            last_hn = hn[-1]
        elif self.backbone_id=="Mamba":
            out = self.backbone(x)
            last_hn = out[-1]
        else:
            raise ValueError("Invalid backbone_id. Options: 'LSTM', 'RNN', 'GRU', 'Mamba'.")

        return out, last_hn


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
            pred_len=1 for binary classification). You may also use this method forecasting too, but it is not recommended.

        Legend:
            B: batch_size, M: num_channels, L: seq_len, N: num_patches, P: patch_dim, D: d_model.
        """

        # Ensure input is correct
        if len(x.shape) == 2:
            x = x.unsqueeze(-2) if self._patching else x.unsqueeze(-1) #: (B, L) -> (B, L, 1)

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm") # Patched version:(B, M, L). Non-patched version: (B, L, 1)

        # Patching
        if self._patching:
            x = self.patcher(x) # (B, M, N, P)
            x = self.pos_enc(x) # (B, M, N, D)
            B, M, N, D = x.shape
            x = x.view(B*M, N, D) # (B*M, N, D)

        # Backbone forward pass
        out, last_hn = self.compute_backbone(x)

        # Normalization
        if self.last_state:
            x = self.norm(last_hn) # Select last hidden state: (B, D)
        elif self.avg_state:
            x = self.norm(torch.mean(out, dim=1)) # Average over sequence length. Patched version: (B*M, D). Non-patched version: (B, D).
        else:
            x = self.norm(out) # Patched version: (B*M, N, D). Non-patched version: (B, L, D)

        # Reshape for patching
        if self._patching:
            x = x.view(B, M, -1) # avg state: (B, M, D). Non-avg state: (B, M, N*D)

        # Head
        if self.return_head:
            x = self.head(self.dropout(x)) # (B, pred_len)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return x

if __name__ == '__main__':
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
    # model = LSTM(
    #     input_size=input_size,
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
    # true_seq_len = 10000
    # x = torch.randn(batch_size, true_seq_len, input_size).to(device)  # (B, L_true, input_dim)

    # # Pass the data through the model
    # output = model(x)
    # output = output.to(device)

    # print(f"Input shape: {x.shape}")
    # print(f"Output shape: {output.shape}")


    #<--Patched Version (forecasting)--->
    # Define model parameters
    patch_dim = 64
    patch_stride = 16
    batch_size = 1
    d_model = 128
    num_enc_layers = 5
    pred_len = 1
    seq_len = 16031
    num_channels = 1

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create an instance of the LSTM model
    model = RecurrentModel(
        d_model=d_model,
        backbone_id="Mamba",
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
        avg_state=True,
    ).to(device)

    # Create sample input data
    x = torch.randn(batch_size, num_channels, seq_len).to(device)  # (B, M, L)
    print(f"Input shape: {x.shape}")

    # Pass the data through the model
    output = model(x)
    output = output.to(device)


    print(f"Output shape: {output.shape}")
