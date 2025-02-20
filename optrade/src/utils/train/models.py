import warnings
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
warnings.filterwarnings("ignore", message="h5py not installed")

# Supervised Models
from time_series.models.patchtst_blind import PatchTST
from time_series.models.recurrent import RecurrentModel
from time_series.models.linear import Linear
from time_series.models.dlinear import DLinear
from time_series.models.modern_tcn import ModernTCN
from time_series.models.timesnet import TimesNet
from time_series.models.tsmixer import TSMixer
from time_series.models.patched_forecaster import EMForecaster

# Optimizers and Schedulers
from torch import optim
from time_series.utils.schedulers import WarmupCosineSchedule, PatchTSTSchedule
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR


def get_model(args, generator=torch.Generator()):
    if args.exp.model_id == "PatchTST":
        model = PatchTST(num_enc_layers=args.sl.num_enc_layers,
                              d_model=args.sl.d_model,
                              d_ff=args.sl.d_ff,
                              num_heads=args.sl.num_heads,
                              num_channels=args.data.num_channels,
                              seq_len=args.data.seq_len,
                              pred_len=args.data.pred_len,
                              attn_dropout=args.sl.attn_dropout,
                              ff_dropout=args.sl.ff_dropout,
                              pred_dropout=args.sl.pred_dropout,
                              batch_first=args.sl.batch_first,
                              norm_mode=args.sl.norm_mode,
                              revin=args.sl.revin,
                              revout=args.sl.revout,
                              revin_affine=args.sl.revin_affine,
                              eps_revin=args.sl.eps_revin,
                              patch_dim=args.data.patch_dim,
                              stride=args.data.patch_stride,
                              head_type=args.sl.head_type,
                              ch_aggr=args.open_neuro.ch_aggr,
                              ch_reduction=args.open_neuro.ch_reduction,
                              cla_mix=args.clm.clm,
                              cla_mix_layers=args.clm.num_enc_layers,
                              cla_combination=args.clm.combo,
                              qwa=args.qwa.qwa,
                              qwa_num_networks=args.qwa.num_networks,
                              qwa_network_type=args.qwa.network_type,
                              qwa_hidden_dim=args.qwa.hidden_dim,
                              qwa_mlp_dropout=args.qwa.mlp_dropout,
                              qwa_attn_dropout=args.qwa.attn_dropout,
                              qwa_ff_dropout=args.qwa.ff_dropout,
                              qwa_norm_mode=args.qwa.norm_mode,
                              qwa_num_heads=args.qwa.num_heads,
                              qwa_num_enc_layers=args.qwa.num_enc_layers,
                              qwa_upper_quantile=args.qwa.upper_quantile,
                              qwa_lower_quantile=args.qwa.lower_quantile,)
    elif args.exp.model_id == "RecurrentModel":
        model = RecurrentModel(d_model=args.sl.d_model,
                               backbone_id=args.exp.backbone_id,
                               num_enc_layers=args.sl.num_enc_layers,
                               pred_len=args.data.pred_len,
                               bidirectional=args.sl.bidirectional,
                               dropout=args.sl.dropout,
                               seq_len=args.data.seq_len,
                               patching=args.data.patching,
                               patch_dim=args.data.patch_dim,
                               patch_stride=args.data.patch_stride,
                               num_channels=args.data.num_channels,
                               head_type=args.sl.head_type,
                               norm_mode=args.sl.norm_mode,
                               revin=args.sl.revin,
                               revout=args.sl.revout,
                               revin_affine=args.sl.revin_affine,
                               eps_revin=args.sl.eps_revin,
                               last_state=args.sl.last_state,
                               avg_state=args.sl.avg_state)
    elif args.exp.model_id == "Linear":
        model = Linear(in_features=args.data.seq_len,
                       out_features=args.data.pred_len,
                       norm_mode=args.sl.norm_mode)
    elif args.exp.model_id == "DLinear":
        model = DLinear(task=args.exp.task,
                        seq_len=args.data.seq_len,
                        pred_len=args.data.pred_len,
                        num_channels=args.data.num_channels,
                        num_classes=args.data.pred_len,
                        moving_avg=args.dlinear.moving_avg,
                        individual=args.dlinear.individual,
                        return_head=args.sl.return_head,
        )
    elif args.exp.model_id == "EMForecaster":
        model = EMForecaster(
            args=args,
            seed=args.exp.seed,
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            revin=args.sl.revin,
            revout=args.sl.revout,
            revin_affine=args.sl.revin_affine,
            eps_revin=args.sl.eps_revin,
            patch_model_id=args.exp.patch_model_id,
            backbone_id=args.exp.backbone_id,
            patch_norm=args.sl.patch_norm,
            patch_act=args.sl.patch_act,
            patch_dim=args.data.patch_dim,
            patch_stride=args.data.patch_stride,
            patch_embed_dim=args.sl.patch_embed_dim,
            independent_patching=args.sl.independent_patching,
            pos_enc=args.sl.pos_enc)
    elif args.exp.model_id == "ModernTCN":
        model = ModernTCN(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            patch_dim=args.data.patch_dim,
            patch_stride=args.data.patch_stride,
            num_classes=args.data.pred_len,
            num_channels=args.data.num_channels,
            task=args.exp.task,
            return_head=args.sl.return_head,
            dropout=args.sl.dropout,
            class_dropout=args.moderntcn.class_dropout,
            ffn_ratio=args.moderntcn.ffn_ratio,
            num_enc_layers=args.moderntcn.num_enc_layers,
            large_size=args.moderntcn.large_size,
            d_model=args.moderntcn.d_model,
            revin=args.sl.revin,
            affine=args.sl.revin_affine,
            small_size=args.moderntcn.small_size,
            dw_dims=args.moderntcn.dw_dims,
        )

    elif args.exp.model_id == "TimesNet":
        model = TimesNet(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_channels=args.data.num_channels,
            d_model=args.timesnet.d_model,
            d_ff=args.timesnet.d_ff,
            num_enc_layers=args.sl.num_enc_layers,
            num_kernels=args.timesnet.num_kernels,
            c_out=args.timesnet.c_out,
            top_k=args.timesnet.top_k,
            dropout=args.sl.dropout,
            task=args.exp.task,
            revin=args.sl.revin,
            revin_affine=args.sl.revin_affine,
            revout=args.sl.revout,
            eps_revin=args.sl.eps_revin,
            return_head=args.sl.return_head,
        )
    elif args.exp.model_id == "TSMixer":
        model = TSMixer(
            seq_len=args.data.seq_len,
            pred_len=args.data.pred_len,
            num_enc_layers=args.sl.num_enc_layers,
            d_model=args.sl.d_model,
            num_channels=args.data.num_channels,
            dropout=args.sl.dropout,
            revin=args.sl.revin,
            revin_affine=args.sl.revin_affine,
            revout=args.sl.revout,
            eps_revin=args.sl.eps_revin,
        )
    else:
        raise ValueError("Please select a valid model_id.")
    return model

def get_optim(args, model, optimizer_type="adamw", flag="sl"):
    if args.exp.sklearn:
        return None

    optimizer_classes = {"adam": optim.Adam, "adamw": optim.AdamW}
    if optimizer_type not in optimizer_classes:
        raise ValueError("Please select a valid optimizer.")
    optimizer_class = optimizer_classes[optimizer_type]

    param_groups = exclude_weight_decay(model, args, flag) # Exclude bias and normalization parameters from weight decay

    optimizer = optimizer_class(param_groups) # Set optimizer

    return optimizer

def exclude_weight_decay(model, args, flag="sl"):
    # Separate parameters into those that will use weight decay and those that won't
    decay_params = []
    no_decay_params = []

    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bias' in name or isinstance(param, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, RevIN)):
                no_decay_params.append(param)
            else:
                decay_params.append(param)

    # Create parameter groups
    param_groups = [
        {'params': decay_params, 'weight_decay': eval(f"args.{flag}.weight_decay")},
        {'params': no_decay_params, 'weight_decay': 0.0}
    ]

    return param_groups

def get_scheduler(args, scheduler_type, training_mode, optimizer, num_batches=0):
    if args.exp.sklearn:
        return None

    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(optimizer,
                                    T_max=args.sl.epochs,
                                    eta_min=args.sl.lr*1e-2,
                                    last_epoch=args.scheduler.last_epoch)
    elif scheduler_type == "patchtst" and training_mode=="supervised":
            scheduler = PatchTSTSchedule(optimizer, args, num_batches)
    elif scheduler_type == "onecycle" and training_mode=="supervised":
        scheduler = OneCycleLR(optimizer=optimizer,
                               steps_per_epoch=num_batches,
                               pct_start=args.scheduler.pct_start,
                               epochs = args.sl.epochs,
                               max_lr = args.sl.lr)
    elif scheduler_type is None:
        return None
    else:
        raise ValueError("Please select a valid scheduler_type.")
    return scheduler

def get_criterion(args, criterion_type):
    if criterion_type == "MSE":
        return nn.MSELoss()
    elif criterion_type == "SmoothL1":
        return nn.SmoothL1Loss()
    elif criterion_type == "BCE":
        return nn.BCEWithLogitsLoss()
    elif criterion_type == "BCE_normal":
        return nn.BCELoss()
    elif criterion_type == "CE":
        return nn.CrossEntropyLoss()
    else:
        raise ValueError("Please select a valid criterion_type.")

def forward_pass(args, model, batch, model_id, device):

    if model_id in {"SSS", "PatchTSTBlind", "PatchTSTOG", "TSMixer", "MLPMixerCI", "RecurrentModel", "Linear", "DLinear", "ModernTCN",
                    "TimesNet", "RF_EMF_CNN", "RF_EMF_MLP", "RF_EMF_LSTM", "RF_EMF_Transformer", "EMForecaster", "CyclicalEMForecaster"}:

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
    if model_id in {"SSS", "PatchTSTBlind", "PatchTSTOG", "TSMixer", "MLPMixerCI", "RecurrentModel", "Linear", "DLinear", "ModernTCN",
                    "TimesNet", "RF_EMF_CNN", "RF_EMF_MLP", "RF_EMF_LSTM", "RF_EMF_Transformer", "EMForecaster", "CyclicalEMForecaster"}:

        target = batch[1].to(device)
        loss = criterion(output, target)
    else:
        raise ValueError("Please select a valid model_id.")

    return loss

def model_update(model, loss, optimizer, model_id, alpha=0.6):
    if model_id in {"SSS", "PatchTSTBlind", "PatchTSTOG", "TSMixer", "MLPMixerCI", "RecurrentModel", "Linear", "DLinear", "ModernTCN",
                    "TimesNet", "RF_EMF_CNN", "RF_EMF_MLP", "RF_EMF_LSTM", "RF_EMF_Transformer", "EMForecaster", "CyclicalEMForecaster"}:
        loss.backward()
        optimizer.step()
    else:
        raise ValueError("Please select a valid model_id.")

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
    print(f"  Total parameters with gradients: {len(vanishing) + len(exploding) + len(normal)}")
    print(f"  Vanishing gradients: {len(vanishing)} ({len(vanishing) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)")
    print(f"  Exploding gradients: {len(exploding)} ({len(exploding) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)")
    print(f"  Normal gradients: {len(normal)} ({len(normal) / (len(vanishing) + len(exploding) + len(normal)) * 100:.2f}%)")

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
    all_grads = [param.grad.norm().item() for name, param in model.named_parameters() if param.grad is not None]
    if all_grads:
        print("\nGradient norm statistics:")
        print(f"  Mean: {np.mean(all_grads):.6f}")
        print(f"  Median: {np.median(all_grads):.6f}")
        print(f"  Std: {np.std(all_grads):.6f}")
        print(f"  Min: {np.min(all_grads):.6f}")
        print(f"  Max: {np.max(all_grads):.6f}")
