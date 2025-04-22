from optrade.config.config import Global

from optrade.exp.forecasting import Experiment
from optrade.dev.utils.models import (
    get_model,
    get_optim,
    get_criterion,
    get_scheduler,
)


def run_forecasting_experiment(args: Global, ablation_id: int) -> None:

    exp = Experiment(
        log_dir=args.exp.log_dir,
        logging="neptune" if args.exp.neptune else "offline",
        seed=args.exp.seed,
        ablation_id=ablation_id,
        exp_id=args.exp.id,
        neptune_project_name=args.exp.project_name,
        neptune_api_token=args.exp.api_token,
        download_only=args.data.download_only,
    )

    # Initialize device
    exp.init_device(mps=args.exp.mps, gpu_id=args.exp.gpu_id)

    # Step 2: Initialize data loaders with specified configuration
    exp.init_loaders(
        root=args.contracts.root,
        start_date=args.contracts.start_date,
        end_date=args.contracts.end_date,
        contract_stride=args.contracts.stride,
        interval_min=args.contracts.interval_min,
        right=args.contracts.right,
        target_tte=args.contracts.target_tte,
        tte_tolerance=args.contracts.tte_tolerance,
        moneyness=args.contracts.moneyness,
        train_split=args.data.train_split,
        val_split=args.data.val_split,
        seq_len=args.data.seq_len,
        pred_len=args.data.pred_len,
        scaling=args.feats.scaling,
        dtype=args.data.dtype,
        core_feats=args.feats.core,
        tte_feats=args.feats.tte,
        datetime_feats=args.feats.datetime,
        keep_datetime=args.feats.keep_datetime,
        target_channels=args.feats.target_channels,
        target_type=args.feats.target_type,
        strike_band=args.contracts.strike_band,
        volatility_type=args.contracts.volatility_type,
        volatility_scaled=args.contracts.volatility_scaled,
        volatility_scalar=args.contracts.volatility_scalar,
        batch_size=args.data.batch_size,
        shuffle=args.data.shuffle,
        drop_last=args.data.drop_last,
        num_workers=args.data.num_workers,
        prefetch_factor=args.data.prefetch_factor,
        pin_memory=args.data.pin_memory,
        persistent_workers=args.data.persistent_workers,
        clean_up=args.data.clean_up,
        offline=args.data.offline,
        save_dir=args.data.save_dir,
        verbose=args.data.verbose,
        validate_contracts=args.data.validate_contracts,
        dev_mode=args.data.dev_mode,
        download_only=args.data.download_only,
    )

    if args.data.download_only:
        return

    # Select Model, Optimizer, and Loss function
    input_channels = (
        args.feats.core + args.feats.tte + args.feats.datetime
    )
    target_channel_idx = [
        input_channels.index(channel) for channel in args.feats.target_channels
    ]
    model = get_model(
        args=args, input_channels=input_channels, target_channels_idx=target_channel_idx
    )
    optimizer = get_optim(args=args, model=model)
    criterion = get_criterion(args=args)

    if args.train.scheduler is not None:
        scheduler = get_scheduler(
            args=args, optimizer=optimizer, num_batches=len(exp.train_loader)
        )
    else:
        scheduler = None

    # Train the model
    model = exp.train(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        num_epochs=args.train.epochs,
        early_stopping=args.train.early_stopping,
        patience=args.train.early_stopping_patience,
        metrics=args.eval.metrics,
    )

    # Step 5: Evaluate model on test set
    exp.test(
        model=model,
        criterion=criterion,
        metrics=args.eval.metrics,  # Metrics to compute
        target_type=args.feats.target_type,
    )

    # Step 6: Save model and logs
    exp.save_logs()  # Save experiment logs to disk or neptune


if __name__ == "__main__":
    pass
