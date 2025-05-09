import yaml
from pathlib import Path
from typing import Optional
import numpy as np
from torch.utils.data import DataLoader
from optrade.config.config import Global
from optrade.exp.forecasting import Experiment
from optrade.dev.utils.models import (
    get_model,
    get_optim,
    get_criterion,
    get_scheduler,
)
from optrade.models.classical._sklearn import get_sklearn_model
from optrade.dev.utils.ablations import load_ablation_config
from optrade.dev.utils.logger import log_pydantic

def run_forecasting_experiment(
    args: Global,
    ablation_id: int,
    job_dir: Path,
    train_loader: Optional[DataLoader] = None,
    val_loader: Optional[DataLoader] = None,
    test_loader: Optional[DataLoader] = None,
    train_x: Optional[np.ndarray] = None,
    train_y: Optional[np.ndarray] = None,
    test_x: Optional[np.ndarray] = None,
    test_y: Optional[np.ndarray] = None
) -> None:
    """
    Run a forecasting experiment using the specified arguments through PyDantic base model Global,
    contain all possible experimental configurations. The function initializes the experiment,
    sets up the data loaders, and trains the model. It also handles both Scikit-Learn and PyTorch
    models based on the provided configuration.

    Args:
        args (Global): Global configuration object containing all the arguments.
        ablation_id (int): ID of the ablation experiment.
        job_dir (Path): Directory where the experiment logs will be saved.
        train_loader (Optional[DataLoader]): DataLoader for the training set.
        val_loader (Optional[DataLoader]): DataLoader for the validation set.
        test_loader (Optional[DataLoader]): DataLoader for the test set.
        train_x (Optional[np.ndarray]): Input features for the training set.
        train_y (Optional[np.ndarray]): Target values for the training set.
        test_x (Optional[np.ndarray]): Input features for the test set.
        test_y (Optional[np.ndarray]): Target values for the test set.

    Returns:
        None
    """

    if args.exp.model_id.startswith("sklearn_"):
        args.exp.sklearn = True

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

    # Initialize data loaders. Optional: download data only and return

    if all(x is None for x in (train_loader, val_loader, test_loader, train_x, train_y, test_x, test_y)):
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
            vol_feats=args.feats.vol,
            rolling_volatility_range=args.feats.rolling_volatility_range,
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
            validate_contracts=args.contracts.validate,
            modify_contracts=args.contracts.modify,
            dev_mode=args.data.dev_mode,
            download_only=args.data.download_only,
        )

    if args.data.download_only:
        return

    #<--------Scikit-Learn------->
    if args.exp.sklearn:
        model_id = args.exp.model_id.replace("sklearn_", "")
        model_args = eval(f"args.{args.exp.model_id}")

        # Get model
        model = get_sklearn_model(
            model_id=model_id,
            target_type=args.feats.target_type,
            model_args=model_args.dict(),
        )

        # Get param_dict
        ablation_path = job_dir / "ablation.yaml"
        ablation_config = load_ablation_config(ablation_path)

        with open(ablation_path, "r") as file:
            ablation_config = yaml.safe_load(file)
        param_dict = ablation_config.get("params", {})

        exp.train_sklearn(
            model=model,
            train_x=train_x,
            train_y=train_y,
            param_dict=param_dict,
            tuning_method=args.sklearn.tuning_method,
            n_splits=args.sklearn.n_splits,
            verbose=args.sklearn.verbose,
            n_jobs=args.sklearn.n_jobs,
            n_iter=args.sklearn.n_iter,
            target_type=args.feats.target_type,
        )

        exp.test_sklearn(
            train_x=train_x,
            train_y=train_y,
            test_x=test_x,
            test_y=test_y,
            metrics=args.eval.metrics,
            target_type=args.feats.target_type)

    #<--------PyTorch------->
    else:
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
        model = exp.train_torch(
            train_loader=train_loader,
            val_loader=val_loader,
            model=model,
            optimizer=optimizer,
            criterion=criterion,
            scheduler=scheduler,
            num_epochs=args.train.epochs,
            early_stopping=args.train.early_stopping,
            patience=args.train.early_stopping_patience,
            metrics=args.eval.metrics,
            best_model_metric=args.eval.best_model_metric,
        )

        # Step 5: Evaluate model on test set
        exp.test_torch(
            test_loader=test_loader,
            model=model,
            criterion=criterion,
            metrics=args.eval.metrics,  # Metrics to compute
            target_type=args.feats.target_type,
        )

    # Step 6: Save model and logs
    log_pydantic(exp.logger, args, key="parameters") # Save Global args to logs
    exp.save_logs()  # Save experiment logs to disk or neptune


if __name__ == "__main__":
    pass
