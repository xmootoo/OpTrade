import torch
import torch.nn as nn
from typing import List
from pydantic import BaseModel

# Layers and models
from optrade.src.models.deep_learning.dlinear.model import DLinear
from optrade.src.models.deep_learning.tsmixer.model import TSMixer

# Weight initialization
from optrade.src.models.deep_learning.utils.weight_init import xavier_init

# Util Layers
from optrade.src.models.deep_learning.utils.revin import RevIN
from optrade.src.models.deep_learning.utils.patcher import Patcher

class EMForecaster(nn.Module):
    def __init__(
        self,
        args:BaseModel,
        seed:int = 42,
        seq_len:int = 336,
        pred_len:int = 96,
        num_channels:int = 1,
        revin:bool = True,
        revout:bool = True,
        revin_affine:bool = True,
        eps_revin:float = 1e-5,
        patch_model_id:str = "DLinear",
        backbone_id:str = "DLinear",
        patch_norm:str = "layer",
        patch_act:str = "GeLU",
        patch_dim:int = 24,
        patch_stride:int = 12,
        patch_embed_dim: int = 128,
        independent_patching:bool = False,
        pos_enc:str = "learnable",
        patching_on:bool = True,
        num_patches:int = 0,
        return_head:bool = True) -> None:
        super(EMForecaster, self).__init__()

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
            patch_model_config (BaseModel): Configuration for patch embedding model.
            patch_norm (str): Normalization layer used in patch embedding.
            patch_act (str): Activation function used in patch embedding.
            patch_dim (int): Patch dimension.
            patch_stride (int): Stride used in patching.
            patch_embed_dim (int): Patch embedding dimension.
            independent_patching (bool): Whether to process patches independently with the patch embedding model, or dependently.
            d_model (int): Dimension of the model.
            patching_on (bool): Only used for extensions of PatchedForecaster when patching is done externally.
            num_pathes (int): Only used for extensions of PatchedForecaster when patching is done externally.
        """

        # Parameters
        self.args = args
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.num_channels = num_channels
        self.patch_embed_dim = patch_embed_dim
        self.patch_model_id = patch_model_id
        self.independent_patching = independent_patching
        self.patch_dim = patch_dim
        self.patching_on = patching_on
        self.return_head = return_head

        if self.patching_on:
            if patch_stride==-1:
                self.patch_stride = self.patch_dim //2
            elif patch_stride==-2:
                self.patch_stride = self.patch_dim
            else:
                self.patch_stride = patch_stride

            self.num_patches = int((seq_len - patch_dim) / self.patch_stride) + 2
        else:
            self.num_patches = num_patches

        # Initialize layers
        self.patch_model_config = self.get_patch_model_config()
        self.backbone_id = backbone_id

        # RevIN
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        # Layers
        self.patcher = Patcher(self.patch_dim, self.patch_stride)
        self.patch_model = self.get_patch_model()

        # Positional encoding
        if pos_enc == "learnable":
            self.pos_enc = nn.Parameter(torch.randn(1, 1, self.num_patches * patch_embed_dim))
        elif pos_enc == "none":
            self.pos_enc = 0

        # Activation function (patching)
        if patch_act == "relu":
            self.patch_act = nn.ReLU()
        elif patch_act == "gelu":
            self.patch_act = nn.GELU()
        else:
            self.patch_act = nn.Identity()

        # Normalization (patching)
        if patch_norm == "layer" and independent_patching:
            self.patch_norm = nn.LayerNorm(self.patch_embed_dim)
        else:
            self.patch_norm = nn.Identity()

        # Base model
        self.base_model = self.get_base_model()

        # Weight initialization
        self.apply(lambda m: xavier_init(m, seed=seed))

    def forward(self, x):

        if self.patching_on:
            # RevIN
            B = x.size(0)
            if self._revin:
                x = self.revin(x, mode="norm")

            # Patch embedding
            x = self.patcher(x).squeeze() # (B, num_patches, patch_dim)
        else:
            pass

        if self.independent_patching:
            x = x.view(B*self.num_patches, 1, self.patch_dim)
            x = self.patch_model(x).view(B, self.num_patches, self.patch_embed_dim) # (B, num_patches, patch_embed_dim)
        else:
            x = self.patch_model(x) # (B, patch_embed_dim, num_patches)

        # Activation function (optional)
        x = self.patch_act(x)

        # Normalization (optional)
        x = self.patch_norm(x)


        if self.return_head:

            # Flatten
            x = x.view(B, 1, self.num_patches*self.patch_embed_dim) # (B, 1, num_patches * patch_embed_dim)

            # # Project + Positional Encoding
            x = x + self.pos_enc # (B, 1, num_patches * patch_embed_dim)

            # Base Model + Linear Head
            x = self.base_model(x) # (B, 1, pred_len)

            # RevOUT
            if self.revout:
                x = self.revin(x, mode="denorm")

        return x

    def get_patch_model_config(self):
        map = {
            "DLinear": self.args.dlinear,
            "TSMixer": self.args.tsmixer
        }

        return map[self.patch_model_id]

    def _init_revin(
        self,
        revout:bool,
        revin_affine:bool) -> None:
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(self.num_channels, self.eps_revin, self.revin_affine)

    def get_base_model(self) -> nn.Module:
        if self.backbone_id == "DLinear":
            return DLinear(
                task="forecasting",
                seq_len=self.num_patches*self.patch_embed_dim,
                pred_len=self.pred_len,
                num_channels=1,
                moving_avg=self.args.dlinear.final_moving_avg,
                individual=True,
                return_head=True)
        elif self.backbone_id == "Linear":
            return nn.Linear(self.num_patches*self.patch_embed_dim, self.pred_len)
        else:
            raise ValueError(f"Invalid backbone_id: {self.backbone_id}")

    def get_patch_model(self) -> nn.Module:
        if self.independent_patching:
            num_channels = 1 # Processes (1, patch_dim) independently for each patch (patch-independent)
        else:
            num_channels = self.num_patches # Processes (num_patches, patch_dim) entirely (patch-dependent)
            individual = False

        if self.patch_model_id == "DLinear":
            return DLinear(
                task="forecasting",
                seq_len=self.patch_dim,
                pred_len=self.patch_embed_dim,
                num_channels=num_channels,
                moving_avg=self.patch_model_config.moving_avg,
                individual=False,
                return_head=False)
        elif self.patch_model_id == "TSMixer":
            embedder = nn.Linear(self.patch_dim, self.patch_embed_dim)
            backbone = TSMixer(
                seq_len=self.patch_embed_dim, # seq_len
                pred_len=1,
                num_enc_layers=self.patch_model_config.num_enc_layers,
                d_model=self.patch_model_config.d_model, # Hidden dim
                num_channels=num_channels, # nvars
                dropout=self.patch_model_config.dropout,
                revin=False,
                revin_affine=False,
                revout=False,
                return_head=False)
            return nn.Sequential(embedder, backbone)
        elif self.patch_model_id == "Linear":
            return NotImplementedError
        else:
            raise NotImplementedError

class CyclicalPatchedForecaster(nn.Module):
    def __init__(
        self,
        args:BaseModel,
        seed:int = 42,
        seq_len:int = 336,
        pred_len:int = 96,
        num_channels:int = 1,
        revin:bool = True,
        revout:bool = True,
        revin_affine:bool = True,
        eps_revin:float = 1e-5,
        patch_model_id:str = "DLinear",
        backbone_id:str = "DLinear",
        patch_norm:str = "layer",
        patch_act:str = "GeLU",
        patch_dim:int = 24,
        patch_stride:int = 12,
        patch_embed_dim: int = 128,
        independent_patching:bool = False,
        pos_enc:str = "learnable",
        datetime_features:List[str] = ["hour", "day"],
        datetime_backbone:str = "PatchedForecaster",
        head_type:str = "linear",
        num_head_layers:int = 2,
        pred_dropout:float = 0.,
        head_d_model:int = 128) -> None:
        super(CyclicalPatchedForecaster, self).__init__()

        self.num_datetime_feat = 2*len(datetime_features)
        self.num_channels = self.num_datetime_feat + 1
        self.datetime_backbone = datetime_backbone
        self.patch_embed_dim = patch_embed_dim
        self.patch_dim = patch_dim

        # Patching
        if patch_stride==-1:
            self.patch_stride = self.patch_dim //2
        elif patch_stride==-2:
            self.patch_stride = self.patch_dim
        else:
            self.patch_stride = patch_stride

        self.num_patches = int((seq_len - patch_dim) / self.patch_stride) + 2

        # RevIN
        self.eps_revin = eps_revin
        self.revin_affine = revin_affine
        if revin:
            self._init_revin(revout, revin_affine)
        else:
            self._revin = None
            self.revout = None

        self.backbone = nn.Sequential(PatchedForecaster(
            args=args,
            seed=seed,
            seq_len=seq_len,
            pred_len=pred_len,
            num_channels=num_channels,
            revin=False,
            revout=False,
            revin_affine=False,
            eps_revin=eps_revin,
            patch_model_id=patch_model_id,
            backbone_id=backbone_id,
            patch_norm=patch_norm,
            patch_act=patch_act,
            patch_dim=patch_dim,
            patch_stride=patch_stride,
            patch_embed_dim=patch_embed_dim,
            independent_patching=independent_patching,
            pos_enc=pos_enc,
            patching_on=True,
            return_head=False),
            nn.Flatten(start_dim=-2))

        # Datetime backbone
        if self.datetime_backbone=="PatchedForecaster": # An individual backbone for each datetime feature
            self.patch_dim = patch_dim
            self.dt_backbone = nn.ModuleList([
                PatchedForecaster(
                    args=args,
                    seed=seed,
                    seq_len=seq_len,
                    pred_len=pred_len,
                    num_channels=num_channels,
                    revin=False,
                    revout=False,
                    revin_affine=False,
                    eps_revin=eps_revin,
                    patch_model_id=patch_model_id,
                    backbone_id=backbone_id,
                    patch_norm=patch_norm,
                    patch_act=patch_act,
                    patch_dim=patch_dim,
                    patch_stride=patch_stride,
                    patch_embed_dim=patch_embed_dim,
                    independent_patching=independent_patching,
                    pos_enc=pos_enc,
                    patching_on=True,
                    return_head=False)
                for i in range(self.num_datetime_feat)
            ])
        elif self.datetime_backbone=="LinearPatcher":
            self.dt_backbone = nn.ModuleList([
                nn.Sequential(
                    Patcher(self.patch_dim, self.patch_stride), # (B, 1, L) -> (B, 1, N, P)
                    nn.Linear(self.patch_dim, self.patch_embed_dim), # (B, 1, N, P) -> (B, 1 N, D)
                    nn.Flatten(start_dim=-2), # (B, 1, N, D) -> (B, 1, N*D)
                )
                for i in range (self.num_datetime_feat)])
        else:
            raise NotImplementedError("OnlyPatchedForecaster datetime_backbone supported for now.")

        # Final head
        if head_type == "TSMixer":
            in_dim = self.num_channels
            self.head = nn.Sequential(
            TSMixer(
                seq_len=self.num_patches*self.patch_embed_dim,
                pred_len=seq_len,
                num_enc_layers=num_head_layers,
                d_model=head_d_model,
                num_channels=self.num_channels,
                dropout=pred_dropout,
                revin=False,
                revout=False,
                return_head=False,
                ),
                nn.Flatten(start_dim=-2),
                nn.Linear(self.num_channels*self.num_patches*self.patch_embed_dim, pred_len)
                )
        elif head_type == "MLP":
            raise NotImplementedError("MLP head not implemented yet.")
        elif head_type == "Linear":
            self.head = nn.Sequential(
                nn.Flatten(start_dim=-2),
                nn.Linear(self.num_channels*self.num_patches*self.patch_embed_dim, pred_len)
            )
        else:
            raise ValueError(f"head_type {head_type} not recognized.")

    def forward(self, x):
        """
        x (torch.Tensor): Shape (batch_size, seq_len, num_channels) where num_channels = 1 + 2*num_datetime_features
        """

        # RevIN
        if self.datetime_backbone in {"LinearPatcher", "PatchedForecaster"}:
            batch_size = x.size(0)
            ts = x[:, 0, :].unsqueeze(1)
            dt = x[:, 1:, :]

            if self._revin:
                ts = self.revin(ts, mode="norm")

            # Forward pass in backbone
            ts_embed = self.backbone(ts)
            dt_embed = torch.zeros(batch_size, self.num_datetime_feat, self.num_patches*self.patch_embed_dim).to(dt.device) # (B, M, N*D)
            for i in range(self.num_datetime_feat):
                dt_out = self.dt_backbone[i](dt[:, i, :].unsqueeze(1))
                dt_embed[:, i, :] = dt_out.flatten(-2)

            # Final Head
            out = torch.cat([ts_embed.unsqueeze(1), dt_embed], dim=1) # Shape (batch_size, num_channels, num_patches*patch_embed_dim)
            out = self.head(out)

            if self.revout:
                out = self.revin(out, mode="denorm")
        else:
            raise ValueError(f"datetime_backbone {self.datetime_backbone} not recognized.")

        return out


    def _init_revin(
        self,
        revout:bool,
        revin_affine:bool) -> None:
        self._revin = True
        self.revout = revout
        self.revin_affine = revin_affine
        self.revin = RevIN(1, self.eps_revin, self.revin_affine) # Only num_channels = 1, applied to the TS channel


if __name__=="__main__":
    from optrade.src.config.config import Global

    args = Global()

    # Univariate case
    batch_size = 32
    seq_len = 512
    pred_len = 96
    x = torch.randn(batch_size, 1, seq_len)


    # Dlinear PatchModel Test
    args.dlinear.moving_avg = 5
    args.dlinear.individual = False
    args.dlinear.final_moving_avg = 25

    model = EMForecaster(
    args=args,
    seq_len=seq_len,
    pred_len=pred_len,
    num_channels=1,
    revin=True,
    revout = True,
    revin_affine = True,
    eps_revin = 1e-5,
    patch_model_id = "DLinear",
    backbone_id = "DLinear",
    patch_norm = "layer",
    patch_act = "GeLU",
    patch_dim = 24,
    patch_stride = -1,
    patch_embed_dim = 128,
    independent_patching = False,
    pos_enc = "learnable",)
    y = model(x)
    print(f"Shape for DLinear PatchModel: {y.shape}")


    # TSMixer PatchModel Test
    args.tsmixer.num_enc_layers = 2
    args.tsmixer.d_model = 256
    args.tsmixer.dropout = 0.1

    model = EMForecaster(
    args=args,
    seq_len=seq_len,
    pred_len=pred_len,
    num_channels=1,
    revin=True,
    revout = True,
    revin_affine = True,
    eps_revin = 1e-5,
    patch_model_id = "TSMixer",
    backbone_id = "Linear",
    patch_norm = "none",
    patch_act = "GeLU",
    patch_dim = 24,
    patch_stride = -1,
    patch_embed_dim = 24,
    independent_patching = False,
    pos_enc = "none",)
    y = model(x)
    print(f"Shape for TSMixer PatchModel: {y.shape}")
