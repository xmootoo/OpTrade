import torch
import torch.nn as nn
from typing import List, Optional
from pydantic import BaseModel

# Layers and models
from optrade.pytorch.models.dlinear import Model as DLinear
from optrade.pytorch.models.tsmixer import Model as TSMixer

# Weight initialization
from optrade.pytorch.utils.weight_init import xavier_init

# Util Layers
from optrade.pytorch.utils.revin import RevIN
from optrade.pytorch.utils.patcher import Patcher


class Model(nn.Module):
    def __init__(
        self,
        args: BaseModel,
        seed: int = 42,
        seq_len: int = 336,
        pred_len: int = 96,
        num_channels: int = 1,
        revin: bool = True,
        revout: bool = True,
        revin_affine: bool = True,
        eps_revin: float = 1e-5,
        patch_model_id: str = "TSMixer",
        patch_norm: str = "none",
        patch_act: str = "GeLU",
        patch_dim: int = 24,
        patch_stride: int = 12,
        patch_embed_dim: int = 128,
        pos_enc: str = "none",
        return_head: bool = True,
        target_channels: Optional[list] = None,
    ) -> None:
        super(Model, self).__init__()

        """
        Patched-based forecasting model (univariate only).

        Args:
            seq_len (int): Length of input sequence.
            pred_len (int): Length of output forecasting.
            num_channels (int): Number of input channels (features).
            revin (bool): Whether to use reversible instance normalization (input).
            revout (bool): Whether to use reversible instance normalization (output).
            revin_affine (bool): Whether to use affine transformation for reversible instance normalization.
            eps_revin (float): Epsilon value for standard deviation numerical stability in reversible instance normalization.
            patch_model_id (str): Model used for patch embedding.
            patch_norm (str): Normalization layer used in patch embedding.
            patch_act (str): Activation function used in patch embedding.
            patch_dim (int): Patch dimension.
            patch_stride (int): Stride used in patching.
            patch_embed_dim (int): Patch embedding dimension.
            d_model (int): Dimension of the model.
            patching_on (bool): Only used for extensions of PatchedForecaster when patching is done externally.
            num_patches (int): Only used for extensions of PatchedForecaster when patching is done externally.
        """

        # Parameters
        self.args = args
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_channels = num_channels
        self.patch_embed_dim = patch_embed_dim
        self.patch_model_id = patch_model_id
        self.patch_dim = patch_dim
        self.return_head = return_head
        self.target_channels = target_channels
        self.revout = revout
        self.revin_affine = revin_affine
        self.eps_revin = eps_revin
        self.num_channels = num_channels

        if patch_stride == -1:
            self.patch_stride = self.patch_dim // 2
        elif patch_stride == -2:
            self.patch_stride = self.patch_dim
        else:
            self.patch_stride = patch_stride

        self.num_patches = int((seq_len - patch_dim) / self.patch_stride) + 2

        # RevIN
        if revin:
            self._init_revin()
        else:
            self._revin = None
            self.revout = None

        # Layers
        self.patcher = Patcher(self.patch_dim, self.patch_stride)
        self.patch_model = self.get_patch_model()

        # Positional encoding
        if pos_enc == "learnable":
            self.pos_enc = nn.Parameter(
                torch.randn(1, self.num_channels, self.num_patches * patch_embed_dim)
            )
        elif pos_enc == "none":
            self.pos_enc = 0

        # Activation function (patching)
        if patch_act == "relu":
            self.patch_act = nn.ReLU()
        elif patch_act == "gelu":
            self.patch_act = nn.GELU()
        else:
            self.patch_act = nn.Identity()

        # Fixed code:
        if target_channels is None:
            num_output_channels = self.num_channels
        else:
            num_output_channels = len(target_channels)

        self.head = nn.ModuleList(
            [
                nn.Linear(self.num_patches * self.patch_embed_dim, self.pred_len)
                for i in range(num_output_channels)
            ]
        )

        # Weight initialization
        self.apply(lambda m: xavier_init(m, seed=seed))

    def forward_patch_model(self, x):
        # Process each channel through its dedicated model
        outputs = []
        for i in range(len(self.patch_model)):
            channel_input = x[:, i, :, :]  # (batch_size, num_patches, patch_dim)
            channel_output = self.patch_model[i](
                channel_input
            )  # (batch_size, num_patches, patch_embed_dim)
            outputs.append(
                channel_output.view(-1, 1, self.num_patches, self.patch_embed_dim)
            )

        # Stack the outputs along the channel dimension
        x = torch.stack(
            outputs, dim=1
        )  # (batch_size, num_channels, num_patches, patch_embed_dim)

        return x

    def forward(self, x):

        # RevIN
        if self._revin:
            x = self.revin(x, mode="norm")

        # Patching
        x = self.patcher(
            x
        ).squeeze()  # (batch_size, num_channels, num_patches, patch_dim)

        # Activation function (optional)
        x = self.patch_act(x)  # (batch_size, num_channels, num_patches, patch_dim)

        # Patch model
        x = self.forward_patch_model(
            x
        )  # (batch_size, num_channels, num_patches, patch_embed_dim)

        if self.return_head:
            # Flatten
            x = x.view(
                -1, self.num_channels, self.num_patches * self.patch_embed_dim
            )  # (B, 1, num_patches * patch_embed_dim)

            # Positional Encoding
            x = (
                x + self.pos_enc
            )  # (B, self.num_channels, num_patches * patch_embed_dim)

            # Base Model + Linear Head
            outputs = []
            for i in range(len(self.head)):
                channel_input = x[
                    :, i, :
                ]  # (batch_size, num_patches * patch_embed_dim)
                channel_output = self.head[i](channel_input)  # (batch_size, pred_len)
                outputs.append(
                    channel_output.view(-1, 1, self.pred_len)
                )  # (batch_size, 1, pred_len)
            x = torch.stack(outputs, dim=1).squeeze(
                2
            )  # (batch_size, num_output_channels, pred_len)

            # RevOUT
            if self.revout:
                x = self.revin(x, mode="denorm")

        return x

        return map[self.patch_model_id]

    def _init_revin(self):
        self._revin = True
        self.revin = RevIN(
            num_channels=self.num_channels,
            target_channels=self.target_channels,
            eps=self.eps_revin,
            affine=self.revin_affine,
        )

    def get_patch_model(self) -> nn.Module:
        if self.patch_model_id == "DLinear":
            return nn.ModuleList(
                [
                    nn.Sequential(
                        DLinear(
                            task="forecasting",
                            seq_len=self.patch_dim,
                            pred_len=self.patch_embed_dim,
                            num_channels=self.num_patches,
                            moving_avg=self.args.emf.moving_avg,
                            individual=False,
                            return_head=False,
                        )
                    )
                    for i in range(self.num_channels)
                ]
            )
        elif self.patch_model_id == "TSMixer":
            return nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(self.patch_dim, self.patch_embed_dim),
                        TSMixer(
                            seq_len=self.patch_embed_dim,
                            pred_len=1,
                            num_enc_layers=self.args.emf.num_enc_layers,
                            d_model=self.args.emf.d_model,
                            num_channels=self.num_patches,
                            dropout=self.args.emf.dropout,
                            revin=False,
                            revin_affine=False,
                            revout=False,
                            return_head=False,
                        ),
                    )
                    for i in range(self.num_channels)
                ]
            )
        elif self.patch_model_id == "Linear":
            return NotImplementedError
        else:
            raise NotImplementedError


if __name__ == "__main__":
    from optrade.config.config import Global

    args = Global()

    # Multivariate Test
    batch_size = 32
    seq_len = 512
    pred_len = 96
    num_channels = 7
    x = torch.randn(batch_size, num_channels, seq_len)

    model = Model(
        args=args,
        seq_len=seq_len,
        pred_len=pred_len,
        num_channels=num_channels,
        patch_model_id="TSMixer",
        revin=True,
        revout=True,
        revin_affine=True,
        target_channels=[6, 3, 1],
    )

    y = model(x)
    print(f"Model output: {y.shape}")
