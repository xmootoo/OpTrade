import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

from inception_block import InceptionBlock


def compute_periods(x, k=2):
    """
    Computes the period of the input time series using FFT.

    Taken from: https://github.com/thuml/Time-Series-Library/blob/main/models/TimesNet.py.

    Args:
        x (torch.Tensor): Shape (batch_size, num_channels, seq_len)
        k: int, number of top frequencies to consider

    """

    # Compute fourier transform
    xf = fft.rfft(x, dim=1)


    # Calculate the amplitudes of fourier coefficients
    # and averaging them along the batch dimension and the channel dimension
    frequency_list = abs(xf).mean(0).mean(-1)
    frequency_list[0] = 0 # remove the DC component

    # Find the top k frequencies
    _, top_list = torch.topk(frequency_list, k)
    top_list = top_list.detach().cpu().numpy()
    periods = x.shape[1] // top_list

    return periods, abs(xf).mean(-1)[:, top_list]


class TimesBlock(nn.Module):
    def __init__(self, seq_len,
                       pred_len,
                       top_k,
                       d_model,
                       d_ff,
                       num_kernels,):
        super(TimesBlock, self).__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.k = top_k
        # parameter-efficient design
        self.conv = nn.Sequential(
            InceptionBlock(d_model, d_ff,
                               num_kernels=num_kernels),
            nn.GELU(),
            InceptionBlock(d_ff, d_model,
                               num_kernels=num_kernels)
        )

    def forward(self, x):
        B, T, N = x.size()
        period_list, period_weight = compute_periods(x, self.k)

        res = []
        for i in range(self.k):
            period = period_list[i]

            # padding
            if (self.seq_len + self.pred_len) % period != 0:
                length = (
                                 ((self.seq_len + self.pred_len) // period) + 1) * period
                padding = torch.zeros([x.shape[0], (length - (self.seq_len + self.pred_len)), x.shape[2]]).to(x.device)
                out = torch.cat([x, padding], dim=1)
            else:
                length = (self.seq_len + self.pred_len)
                out = x

            # reshape
            out = out.reshape(B, length // period, period,
                              N).permute(0, 3, 1, 2).contiguous()
            # 2D conv: from 1d Variation to 2d Variation
            out = self.conv(out)
            # reshape back
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N)
            res.append(out[:, :((self.seq_len + self.pred_len)), :])
        res = torch.stack(res, dim=-1)

        # adaptive aggregation
        period_weight = F.softmax(period_weight, dim=1)
        period_weight = period_weight.unsqueeze(
            1).unsqueeze(1).repeat(1, T, N, 1)
        res = torch.sum(res * period_weight, -1)

        # residual connection
        res = res + x
        return res


if __name__ == "__main__":
    d_model = 128
    batch_size = 32
    seq_len = 512
    pred_len = 96
    top_k = 3
    d_ff = 32
    num_kernels = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TimesBlock(seq_len,
                       pred_len,
                       top_k,
                       d_model,
                       d_ff,
                       num_kernels).to(device)
    x = torch.randn(batch_size, seq_len+pred_len, d_model).to(device)

    out = model(x).to(device)
    print(out.shape)
