import torch
import torch.nn as nn
from backbone import ModernTCNBackbone

class ModernTCN(nn.Module):
    def __init__(self,
                       # Modifiable (everything below)
                       seq_len,
                       pred_len,
                       patch_dim,
                       patch_stride,
                       num_classes,
                       num_channels,
                       task,
                       return_head=True,
                       dropout=0.0,
                       class_dropout=0.0,
                       ffn_ratio=1,
                       num_enc_layers=[2],
                       large_size=[13],
                       d_model=[32],
                       revin=True,
                       affine=False,

                       # Non-modifiable (everything below)
                       small_size=[5],
                       stem_ratio=6,
                       downsample_ratio=2,
                       dw_dims=[256],
                       small_kernel_merged=False,
                       head_dropout=0.0,
                       use_multi_scale=False,
                       subtract_last=False,
                       individual=False):
        super(ModernTCN, self).__init__()

        """
        ModernTCN: https://openreview.net/forum?id=vpJMJerXHU#.

        This is configured for classification purposes. Please see the original implementation for forecasting (and which args to modify).
        Arguments listed as modifable where hyperparameter tuned in the original experiments for classification, whereas nonmodifable arguments
        were fixed on all experiments.

        Args (can modify):
            ffn_ratio (int): The expansion factor for the feed-forward networks in each block, d_ffn = d_model*ffn_ratio. Choose from {1, 2, 4, 8}
            num_enc_layers (int): Choose from {1, 2, 3} and can make it a list for multistaging with 5 possible stages [a,b,c,d,e] with each element
                                  from {1, 2, 3}. For exameple [1, 1] or [2, 2, 3].
            patch_dim (int): Choose in {8, 16, 32, 48}
            patch_stride (int): Choose in {patch_size, patch_size//2}
            large_size (int): Size of the large kernel. Choose from {13, 31, 51, 71}. Make a list for multistaging, length equal to number of stages.
            small_size (int): Size of the small kernel Set to 5 for all experiments. Make a list for multistaging, length equal to number of stages.
            d_model (int): The model dimension (i.e. Conv1D channel dimension) for each stage. Choose from {32, 64, 128, 256, 512}. Make a list for multistaging, length equal to number of stages.
            dropout (float): Dropout rate for the model. Choose from {0.1, 0.3, 0.5}.
            class_dropout (float): Dropout before the final linear head (classification only). Choose from {0.0, 0.1}.
            affine (bool): Whether to use affine parameters for RevIN.

        Args (do not modify):
            head_dropout (float): Set to 0.0.
            use_multi_scale (bool): Set to False.
            revin (bool): Set to True.
            subtract_last (bool): Set to False.
            kernel_size (int): Useless argument. Set to 25.
            individual (bool): Set to False.
            stem_ratio (int): Set to 6.
            downsample_ratio (int): Set to 2.
        """

        # hyper param
        self.task_name = task
        self.stem_ratio = stem_ratio
        self.downsample_ratio = downsample_ratio
        self.ffn_ratio = ffn_ratio
        self.num_enc_layers = num_enc_layers
        self.large_size = large_size
        self.small_size = small_size
        self.d_model = d_model
        self.dw_dims = dw_dims

        self.nvars = self.c_in = num_channels
        self.small_kernel_merged = small_kernel_merged
        self.drop_backbone = dropout
        self.drop_head = head_dropout
        self.use_multi_scale = use_multi_scale
        self.revin = revin
        self.affine = affine
        self.subtract_last = subtract_last

        self.seq_len = seq_len
        self.individual = individual
        self.target_window = pred_len

        self.patch_dim = patch_dim
        self.patch_stride = patch_stride

        #classification
        self.class_dropout = class_dropout
        self.num_classes = num_classes
        self.return_head = return_head


        self.model = ModernTCNBackbone(task_name=self.task_name,patch_dim=self.patch_dim, patch_stride=self.patch_stride, stem_ratio=self.stem_ratio,
                           downsample_ratio=self.downsample_ratio, ffn_ratio=self.ffn_ratio, num_enc_layers=self.num_enc_layers,
                           large_size=self.large_size, small_size=self.small_size, dims=self.d_model, dw_dims=self.dw_dims, nvars=self.nvars,
                           small_kernel_merged=self.small_kernel_merged, backbone_dropout=self.drop_backbone, head_dropout=self.drop_head,
                           use_multi_scale=self.use_multi_scale, revin=self.revin, affine=self.affine, subtract_last=self.subtract_last, seq_len=self.seq_len, c_in=self.c_in,
                           individual=self.individual, target_window=self.target_window, class_drop = self.class_dropout, num_classes = self.num_classes,
                           return_head=self.return_head)

        self.head_in_dim = self.model.head_in_dim

    def forward(self, x):
        x = self.model(x)
        return x


# Test
if __name__ == '__main__':
    seq_len = 512
    pred_len = 96
    patch_dim = 16
    patch_stride = 8
    num_classes = 2
    num_channels = 2
    task = "classification"

    batch_size = 32
    x = torch.randn(32, num_channels, seq_len)

    model = ModernTCN(seq_len=seq_len,
                 pred_len=pred_len,
                 patch_dim=patch_dim,
                 patch_stride=patch_stride,
                 num_classes=num_classes,
                 num_channels=num_channels,
                 task=task)

    y = model(x)
    print(f"x: {x.shape}")
    print(f"y: {y.shape}")
