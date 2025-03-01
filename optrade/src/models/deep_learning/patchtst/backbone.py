import torch
import torch.nn as nn
from enc_block import SupervisedHead, EncoderBlock

class PatchTSTBackbone(nn.Module):
    def __init__(self, num_enc_layers, d_model, d_ff, num_heads, num_channels, num_patches, pred_len, attn_dropout=0.0,
        ff_dropout=0.0, pred_dropout=0.0, batch_first=True, norm_mode="batch1d", return_head=True, head_type="linear"):
        super(PatchTSTBackbone, self).__init__()

        # Parameters
        self.num_channels = num_channels
        self.num_patches = num_patches
        self.d_model = d_model
        self.return_head = return_head
        self.pred_len = pred_len

        # Encoder
        self.enc = nn.Sequential(*(EncoderBlock(d_model, d_ff, num_heads, num_channels, num_patches, attn_dropout, ff_dropout,
                                                batch_first, norm_mode) for i in range(num_enc_layers)))

        # Prediction head
        head_in_dim = num_patches*d_model*2 if self.clm else num_patches*d_model
        head_hidden_dim = num_patches*d_model // 2 if self.clm else num_patches*d_model
        self.flatten = nn.Flatten(start_dim=-2)
        if head_type=="linear":
            self.head = SupervisedHead(head_in_dim, pred_len, pred_dropout)
        elif head_type=="mlp":
            self.head = nn.Sequential(nn.Linear(head_in_dim, head_hidden_dim),
                                        nn.GELU(),
                                        nn.Dropout(pred_dropout),
                                        nn.Linear(head_hidden_dim, pred_len))
        else:
            raise ValueError(f"Invalid head type: {head_type}")

    def forward(self, x:torch.Tensor, y:torch.Tensor=None, ch_ids:torch.Tensor=None) -> torch.Tensor:

        # Encoding
        batch_size = x.shape[0]
        x = x.view(batch_size * self.num_channels, self.num_patches, -1) # (batch_size * num_channels, num_patches, d_model)
        x = self.enc(x) # (batch_size * num_channels, num_patches, d_model)

        if self.return_head:
            x = self.flatten(x) # (batch_size, num_channels, num_patches*d_model)
            out = self.head(x)
        else:
            out = x.view(batch_size, self.num_channels, self.num_patches, -1)

        return out
