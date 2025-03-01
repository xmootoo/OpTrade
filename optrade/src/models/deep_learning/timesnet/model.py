import torch
import torch.nn as nn
import torch.nn.functional as F

from timesblock import TimesBlock
from embed import DataEmbedding

from optrade.src.models.deep_learning.utils.revin import RevIN

class TimesNet(nn.Module):
    """
    Paper link: https://openreview.net/pdf?id=ju_Uqw384Oq
    """
    def __init__(self, seq_len,
                       pred_len,
                       num_channels,
                       d_model,
                       d_ff,
                       num_enc_layers,
                       num_kernels,
                       c_out,
                       top_k,
                       dropout,
                       task,
                       revin=False,
                       revin_affine=False,
                       revout=False,
                       eps_revin=1e-5,
                       return_head=True):
        super(TimesNet, self).__init__()

        # Parameters
        self.task = task
        self.seq_len = seq_len
        self.pred_len = 0 if task=="classification" else pred_len
        self.num_channels = num_channels
        self.num_enc_layers = num_enc_layers
        self.eps_revin = eps_revin
        self.return_head = return_head

        # RevIN
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        # Backbone
        self.backbone = nn.ModuleList([TimesBlock(seq_len=seq_len,
                                                  pred_len=self.pred_len,
                                                  top_k=top_k,
                                                  d_model=d_model,
                                                  d_ff=d_ff,
                                                  num_kernels=num_kernels)
                                    for _ in range(num_enc_layers)])

        # Embedding
        max_len = seq_len if seq_len > 5000 else 5000
        self.enc_embedding = DataEmbedding(c_in=num_channels, d_model=d_model, dropout=dropout, max_len=max_len)

        # Normalization
        self.layer_norm = nn.LayerNorm(d_model)

        # Head
        if self.task == "forecasting":
            self.predict_linear = nn.Linear(self.seq_len, self.pred_len + self.seq_len)
            self.head = nn.Linear(d_model, c_out, bias=True)
        if self.task == 'classification':
            self.act = F.gelu
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(d_model * seq_len, pred_len)

    def _init_revin(self, revout:bool, revin_affine:bool):
            self._revin = True
            self.revout = revout
            self.revin_affine = revin_affine
            self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

    def forecast(self, x):

        """
        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, num_channels, seq_len)
        """

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        # embedding
        x = self.enc_embedding(x.permute(0, 2, 1))  # (batch_size, seq_len, num_channels)
        x = self.predict_linear(x.permute(0, 2, 1)).permute(0, 2, 1)  # align temporal dimension

        # TimesNet
        for i in range(self.num_enc_layers):
            x = self.layer_norm(self.backbone[i](x))

        # Project back
        if self.return_head:
            out = self.head(x) # (batch_size, pred_len, num_channels)

        # RevOUT
        if self.revout:
            out = self.revin(out.permute(0, 2, 1), mode="denorm") # (batch_size, num_channels, pred_len)

        return out

    def classification(self, x):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm") # (batch_size, num_channels, seq_len)

        # Embedding
        x = self.enc_embedding(x.permute(0, 2, 1), None)  # (batch_size, seq_len, num_channels)

        # TimesNet
        for i in range(self.num_enc_layers):
            x = self.layer_norm(self.backbone[i](x))

        # Output
        out = self.act(x)
        out = self.dropout(out)

        if self.return_head:
            out = out.reshape(out.shape[0], -1) # (batch_size, seq_length * d_model)
            out = self.head(out)  # (batch_size, num_classes)

        return out

    def forward(self, x):
        if self.task == "forecasting":
            out = self.forecast(x)[:, :, -self.pred_len:] # (batch_size, num_channels, pred_len)
        elif self.task == "classification":
            out = self.classification(x) # (batch_size, num_classes)
        else:
            raise ValueError("Invalid task name")

        return out



if __name__ == "__main__":


    # <-----Forecasting---->
    batch_size = 32
    seq_len = 512
    pred_len = 48
    num_channels = 7
    d_model = 128
    d_ff = 256
    num_enc_layers = 3
    num_kernels = 3
    c_out = 1
    top_k = 4
    dropout = 0.1
    task = "forecasting"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesNet(seq_len=seq_len,
                        pred_len=pred_len,
                        num_channels=num_channels,
                        d_model=d_model,
                        d_ff=d_ff,
                        num_enc_layers=num_enc_layers,
                        num_kernels=num_kernels,
                        c_out=c_out,
                        top_k=top_k,
                        dropout=dropout,
                        task=task,
                        revin=True,
                        revin_affine=True,
                        revout=True).to(device)

    x = torch.randn(batch_size, num_channels, seq_len).to(device)
    out = model(x)
    print(f"Output (forecasting): {out.shape}")


    #<----Classification---->
    batch_size = 32
    seq_len = 512
    pred_len = 9
    num_channels = 1
    d_model = 128
    d_ff = 256
    num_enc_layers = 3
    num_kernels = 4
    c_out = 1
    top_k = 4
    dropout = 0.1
    task = "classification"


    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TimesNet(seq_len=seq_len,
                        pred_len=pred_len,
                        num_channels=num_channels,
                        d_model=d_model,
                        d_ff=d_ff,
                        num_enc_layers=num_enc_layers,
                        num_kernels=num_kernels,
                        c_out=c_out,
                        top_k=top_k,
                        dropout=dropout,
                        task=task,
                        revin=True,
                        revin_affine=True,
                        revout=False).to(device)

    x = torch.randn(batch_size, num_channels, seq_len).to(device)
    out = model(x)
    print(f"Output (classification) {out.shape}")
