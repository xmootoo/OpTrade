import torch
import torch.nn as nn

from optrade.src.models.deep_learning.utils.revin import RevIN
from optrade.src.models.deep_learning.utils.weight_init import xavier_init

class Linear(nn.Module):
    def __init__(self, in_features,
                       out_features,
                       norm_mode="layer",
                       revin=True,
                       revout=False,
                       revin_affine=False,):
        super(Linear, self).__init__()

        # Normalization
        if norm_mode=="layer":
            self.norm = nn.LayerNorm(self.d_model)
        elif norm_mode=="batch1d":
            self.norm = nn.BatchNorm1d(self.d_model)
        else:
            self.norm = nn.Identity()


        # RevIN
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        self.linear = nn.Linear(in_features, out_features)

        # Weight initialization
        self.apply(xavier_init)

    def _init_revin(self, revout:bool, revin_affine:bool):
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

    def forward(self, x):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        x = self.norm(x.squeeze())
        out = self.linear(x)

        # RevOUT
        if self.revout:
            x = self.revin(x, mode="denorm")

        return out
