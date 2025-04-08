import warnings
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from pydantic import BaseModel

warnings.filterwarnings("ignore", message="h5py not installed")

from typing import List

# Supervised Models
from optrade.pytorch.models.patchtst import Model as PatchTST
from optrade.pytorch.models.recurrent import Model as RecurrentModel
from optrade.pytorch.models.linear import Model as Linear
from optrade.pytorch.models.dlinear import Model as DLinear
from optrade.pytorch.models.tsmixer import Model as TSMixer
from optrade.pytorch.models.emforecaster import Model as EMForecaster

# Optimizers and Schedulers
from torch import optim

from optrade.personal.utils.schedulers import WarmupCosineSchedule
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


def get_model(
    args: BaseModel,
    input_channels: List[str],
    target_channels_idx: List[int],
) -> nn.Module:
    if args.exp.model_id == "PatchTST":
        model = PatchTST(
            input_channels=input_channels,
            num_enc_layers=args.patchtst.num_enc_layers,
            d_model=args.patchtst.d_model,
            d_ff=args.patchtst.d_ff,
            num_heads=args.patchtst.num_heads,
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            attn_dropout=args.patchtst.attn_dropout,
            ff_dropout=args.patchtst.ff_dropout,
            pred_dropout=args.patchtst.pred_dropout,
            batch_first=args.train.batch_first,
            norm_mode=args.train.norm_mode,
            revin=args.train.revin,
            revout=args.train.revout,
            revin_affine=args.train.revin_affine,
            eps_revin=args.train.eps_revin,
            patch_dim=args.data.patch_dim,
            stride=args.data.patch_stride,
            return_head=args.train.return_head,
            head_type=args.train.head_type,
            channel_independent=args.train.channel_independent,  # Head only
            target_channels=target_channels_idx,  # Head only
        )
    elif args.exp.model_id == "RecurrentModel":
        model = RecurrentModel(
            d_model=args.rnn.d_model,
            num_enc_layers=args.rnn.num_enc_layers,
            pred_len=args.data.pred_len,
            backbone_id=args.rnn.backbone_id,
            bidirectional=args.rnn.bidirectional,
            dropout=args.train.dropout,
            seq_len=args.data.seq_len,
            patching=args.data.patching,
            patch_dim=args.data.patch_dim,
            patch_stride=args.data.patch_stride,
            num_channels=args.data.num_channels,
            head_type=args.train.head_type,
            norm_mode=args.train.norm_mode,
            revin=args.train.revin,
            revout=args.train.revout,
            revin_affine=args.train.revin_affine,
            eps_revin=args.train.eps_revin,
            last_state=args.rnn.last_state,
            avg_state=args.rnn.avg_state,
            return_head=args.train.return_head,
            channel_independent=args.train.channel_independent,
            target_channels=target_channels_idx,
        )
    elif args.exp.model_id == "Linear":
        model = Linear(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            norm_mode=args.train.norm_mode,
            revin=args.train.revin,
            revout=args.train.revout,
            revin_affine=args.train.revin_affine,
            eps_revin=args.train.eps_revin,
            channel_independent=args.train.channel_independent,
            target_channels=target_channels_idx,
        )
    elif args.exp.model_id == "DLinear":
        model = DLinear(
            task=args.exp.task,
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            num_classes=args.data.num_classes,
            moving_avg=args.dlinear.moving_avg,
            individual=args.dlinear.individual,
            return_head=args.train.return_head,
            revin=args.train.revin,
            revout=args.train.revout,
            revin_affine=args.train.revin_affine,
            eps_revin=args.train.eps_revin,
            target_channels=target_channels_idx,
        )
    elif args.exp.model_id == "EMForecaster":
        model = EMForecaster(
            args=args,
            seed=args.exp.seed,
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            revin=args.train.revin,
            revout=args.train.revout,
            revin_affine=args.train.revin_affine,
            eps_revin=args.train.eps_revin,
            patch_model_id=args.emf.patch_model_id,
            patch_norm=args.emf.patch_norm,
            patch_act=args.emf.patch_act,
            patch_dim=args.data.patch_dim,
            patch_stride=args.data.patch_stride,
            patch_embed_dim=args.emf.patch_embed_dim,
            pos_enc=args.emf.pos_enc,
            return_head=args.train.return_head,
            target_channels=target_channels_idx,
        )
    elif args.exp.model_id == "TSMixer":
        model = TSMixer(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_enc_layers=args.tsmixer.num_enc_layers,
            d_model=args.tsmixer.d_model,
            num_channels=args.data.num_channels,
            dropout=args.train.dropout,
            revin=args.train.revin,
            revin_affine=args.train.revin_affine,
            revout=args.train.revout,
            eps_revin=args.train.eps_revin,
            return_head=args.train.return_head,
            target_channels=target_channels_idx,
            channel_independent=args.train.channel_independent,
        )
    else:
        raise ValueError("Please select a valid model_id.")
    return model


def get_optim(args, model):
    optimizer_classes = {"adam": optim.Adam, "adamw": optim.AdamW}

    if args.train.optimizer not in optimizer_classes:
        raise ValueError("Please select a valid optimizer.")
    optimizer_class = optimizer_classes[args.train.optimizer]

    param_groups = exclude_weight_decay(
        model, args
    )  # Exclude bias and normalization parameters from weight decay

    optimizer = optimizer_class(param_groups)  # Set optimizer

    return optimizer


def exclude_weight_decay(model, args):
    # Separate parameters into those that will use weight decay and those that won't
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if "bias" in name or isinstance(
                param, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)
            ):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {"params": decay_params, "weight_decay": args.train.weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]

    return param_groups


def get_scheduler(args: BaseModel, optimizer, num_batches: int = 0):
    if args.train.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.train.epochs,
            eta_min=args.train.lr * 1e-2,
            last_epoch=args.scheduler.last_epoch,
        )
    elif args.train.scheduler == "onecycle":
        scheduler = OneCycleLR(
            optimizer=optimizer,
            steps_per_epoch=num_batches,
            pct_start=args.scheduler.pct_start,
            epochs=args.train.epochs,
            max_lr=args.train.lr,
        )
    else:
        raise ValueError("Please select a valid scheduler_type.")
    return scheduler


def get_criterion(args: BaseModel) -> nn.Module:
    if args.train.criterion == "MSE":
        return nn.MSELoss()
    elif args.train.criterion == "SmoothL1":
        return nn.SmoothL1Loss()
    elif args.train.criterion == "BCE":
        return nn.BCEWithLogitsLoss()
    elif args.train.criterion == "BCE_normal":
        return nn.BCELoss()
    elif args.train.criterion == "CE":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Please select a valid criterion.")


def forward_pass(args, model, batch, model_id, device):
    if model_id in {
        "PatchTST",
        "TSMixer",
        "RecurrentModel",
        "Linear",
        "DLinear",
        "ModernTCN",
        "TimesNet",
        "EMForecaster",
    }:
        x = batch[0]
        x = x.to(device)

        if args.data.difference_input:
            x = x.diff(dim=1)
            # Pad the last dimension with the last value of the sequence
            x = torch.cat([x, x[:, -1].unsqueeze(1)], dim=1)

        output = model(x)
    else:
        raise ValueError("Please select a valid model_id.")

    return output


def compute_loss(output, batch, criterion, model_id, args, device):
    if model_id in {
        "PatchTST",
        "TSMixer",
        "RecurrentModel",
        "Linear",
        "DLinear",
        "ModernTCN",
        "TimesNet",
        "EMForecaster",
    }:
        target = batch[1].to(device)
        loss = criterion(output, target)
    else:
        raise ValueError("Please select a valid model_id.")

    return loss


def model_update(model, loss, optimizer, model_id):
    if model_id in {
        "PatchTST",
        "TSMixer",
        "RecurrentModel",
        "Linear",
        "DLinear",
        "ModernTCN",
        "TimesNet",
        "EMForecaster",
    }:
        loss.backward()
        optimizer.step()
    else:
        raise ValueError(f"{model_id} is not a valid model_id.")


def check_gradients(model, threshold_low=1e-5, threshold_high=1e2):
    vanishing = []
    exploding = []
    normal = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.norm().item()
            if grad_norm < threshold_low:
                vanishing.append((name, grad_norm))
            elif grad_norm > threshold_high:
                exploding.append((name, grad_norm))
            else:
                normal.append((name, grad_norm))

    print(f"Gradient statistics:")
    print(
        f"  Total parameters with gradients: {len(vanishing) + len(exploding) + len(normal)}"
    )
    print(
        f"  Vanishing gradients: {len(vanishing)} ({len(vanishing) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)"
    )
    print(
        f"  Exploding gradients: {len(exploding)} ({len(exploding) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)"
    )
    print(
        f"  Normal gradients: {len(normal)} ({len(normal) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)"
    )

    if vanishing:
        print("\nVanishing gradients:")
        for name, grad_norm in vanishing[:10]:  # Print first 10
            print(f"  {name}: {grad_norm}")
        if len(vanishing) > 10:
            print(f"  ... and {len(vanishing) - 10} more")

    if exploding:
        print("\nExploding gradients:")
        for name, grad_norm in exploding[:10]:  # Print first 10
            print(f"  {name}: {grad_norm}")
        if len(exploding) > 10:
            print(f"  ... and {len(exploding) - 10} more")

    # Compute and print gradient statistics
    all_grads = [
        param.grad.norm().item()
        for name, param in model.named_parameters()
        if param.grad is not None
    ]
    if all_grads:
        print("\nGradient norm statistics:")
        print(f"  Mean: {np.mean(all_grads):.6f}")
        print(f"  Median: {np.median(all_grads):.6f}")
        print(f"  Std: {np.std(all_grads):.6f}")
        print(f"  Min: {np.min(all_grads):.6f}")
        print(f"  Max: {np.max(all_grads):.6f}")
