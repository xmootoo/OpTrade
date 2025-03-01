import torch
from torch import Tensor
import torch.nn as nn

from backbone import PatchTSTBackbone

from optrade.src.models.deep_learning.utils.revin import RevIN
from optrade.src.models.deep_learning.utils.patcher import Patcher
from optrade.src.models.deep_learning.utils.pos_enc import PositionalEncoding
from optrade.src.models.deep_learning.utils.weight_init import xavier_init


class PatchTST(nn.Module):
    def __init__(self, num_enc_layers, d_model, d_ff, num_heads, num_channels, seq_len, pred_len, attn_dropout=0.0,
        ff_dropout=0.0, pred_dropout=0.0, batch_first=True, norm_mode="batch1d", revin=True, revout=True, revin_affine=True,
        eps_revin=1e-5, patch_dim=16, stride=1, return_head=True, head_type="linear"):
        super(PatchTST, self).__init__()

        # Parameters
        self.num_patches = int((seq_len - patch_dim) / stride) + 2
        self.num_channels = num_channels
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine

        # Initialize layers
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        self.patcher = Patcher(patch_dim, stride)
        self.pos_enc = PositionalEncoding(patch_dim, d_model, self.num_patches)
        self.backbone = PatchTSTBackbone(num_enc_layers, d_model, d_ff, num_heads, num_channels, self.num_patches, pred_len,
                                         attn_dropout,ff_dropout, pred_dropout, batch_first, norm_mode, return_head, head_type)

        # Weight initialization
        self.apply(xavier_init)

    def _init_revin(self, revout:bool, revin_affine:bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

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

if __name__ == "__main__":
    batch_size = 64
    num_channels = 1
    seq_len = 96
    patch_dim = 16
    patch_stride = 8
    pred_len = 1

    model = PatchTST(num_enc_layers=3,
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
                            return_head=False,
                            head_type="linear",)

    # Count number of parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of parameters: {num_params}")

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()
    x = torch.randn(batch_size, num_channels, seq_len) # (B, M, L)
    ch_ids = torch.randint(0, 4, (batch_size,))
    targets = torch.randint(0, 2, (batch_size,))
