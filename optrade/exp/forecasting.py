import json
import warnings
import random
import time
import neptune
from typing import Optional, List, Tuple, Union
from pathlib import Path
from rich.console import Console
from sklearn.base import BaseEstimator

# Torch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

# Custom Modules
from optrade.data.contracts import get_contract_datasets
from optrade.data.forecasting import get_forecasting_dataset, get_forecasting_loaders
from optrade.utils.train import EarlyStopping
from optrade.utils.misc import format_time_dynamic, generate_random_id

warnings.filterwarnings(
    "ignore", message="h5py not installed, hdf5 features will not be supported."
)


class Experiment:
    def __init__(
        self,
        log_dir: str = "logs",
        logging: str = "offline",
        seed: int = 42,
        ablation_id: Optional[int] = None,
        exp_id: str = generate_random_id(length=6),
        neptune_project_name: Optional[str] = None,
        neptune_api_token: Optional[str] = None,
        download_only: bool = False,
    ) -> None:
        """

        Experiment class for training and evaluating forecasting models in PyTorch.

        Args:
            logdir: The directory to save logs.
            logging: The logging method to use. Options: {"offline", "neptune"}.
            seed: The random seed to use for reproducibility.
            ablation_id: The ablation ID.
            exp_id: The experiment ID.
            neptune_project_name: The Neptune project name.
            neptune_api_token: The Neptune API token.

        Returns:
            None
        """
        self.start_time = time.time()

        # Rich Console
        self.ctx = Console()
        self.seed = seed
        self.logging = logging
        self.neptune_project_name = neptune_project_name
        self.neptune_api_token = neptune_api_token
        self.exp_id = exp_id
        self.log_dir = Path(log_dir)

        if ablation_id:
            self.log_dir = self.log_dir / "logs" / exp_id / str(ablation_id)
        else:
            self.log_dir = self.log_dir / "logs" / exp_id

        # Set up logging directory
        if not download_only:
            self.init_logger(exp_id=exp_id)

        self.set_seed()

    def save_logs(self):
        # Stop Logger
        if self.logging == "neptune":
            self.logger.stop()
        elif self.logging == "offline":
            # Save offline logging to JSON file
            with open(self.log_file, "w") as f:
                json.dump(self.logger, f, indent=2)
        else:
            raise ValueError(f"Invalid logging method: {self.logging}.")

    # Initialize device
    def init_device(self, gpu_id: int = 0, mps: bool = False) -> None:
        """
        Initialize CUDA (or MPS) devices.

        Args:
            gpu_id: The GPU ID to use.
            mps: Whether to use MPS (Metal Performance Shaders) for macOS.

        Returns:
            None
        """
        if mps:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self.print_master("MPS hardware acceleration activated.")
        elif torch.cuda.is_available():
            self.device = torch.device(f"cuda:{gpu_id}")
        else:
            self.device = torch.device("cpu")
            self.print_master("CUDA not available. Running on CPU.")

    def init_logger(
        self,
        exp_id: str,
    ) -> None:
        """
        Initialize the logger.

        Args:
            logdir: The directory to save logs.
            ablation_id: The ablation ID.
            exp_id: The experiment ID.

        Returns:
            None
        """

        if self.logging == "neptune":
            # Initialize Neptune run with the time-based ID
            self.logger = neptune.init_run(
                project=self.neptune_project_name,
                api_token=self.neptune_api_token,
                custom_run_id=exp_id,
            )
            self.print_master("Neptune logger initialized.")
        elif self.logging == "offline":
            self.logger = dict()
            self.log_file = self.log_dir / "log.json"

            if Path(self.log_dir).exists():
                self.print_master(f"Using existing log directory: {self.log_dir}")
            else:
                Path(self.log_dir).mkdir(parents=True, exist_ok=True)
                self.print_master(f"Created new log directory: {self.log_dir}")

            self.print_master("Offline logger initialized.")
        else:
            raise ValueError(f"Invalid logging method: {self.logging}.")

    def init_loaders(
        self,
        root: str,
        start_date: str,
        end_date: str,
        contract_stride: int,
        interval_min: int,
        right: str,
        target_tte: int,
        tte_tolerance: Tuple[int, int],
        moneyness: str,
        train_split: float,
        val_split: float,
        seq_len: int,
        pred_len: int,
        scaling: bool = False,
        dtype: str = "float32",
        core_feats: List[str] = ["option_returns"],
        tte_feats: Optional[List[str]] = None,
        datetime_feats: Optional[List[str]] = None,
        keep_datetime: bool = False,
        target_channels: Optional[List[str]] = None,
        target_type: str = "multistep",
        strike_band: Optional[float] = 0.05,
        volatility_type: Optional[str] = "period",
        volatility_scaled: bool = False,
        volatility_scalar: Optional[float] = 1.0,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = torch.cuda.is_available(),
        persistent_workers: bool = True,
        clean_up: bool = False,
        offline: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        validate_contracts: bool = True,
        dev_mode: bool = False,
        download_only: bool = False,
    ) -> None:
        """

        Initializes the data loaders for training, validation, and testing.

        Args:
            root: The root directory containing the data.
            start_date: The start date for the data in YYYYMMDD format.
            end_date: The end date for the data in YYYYMMDD format.
            contract_stride: The stride for the contracts.
            interval_min: The interval in minutes for the data.
            right: The option type (C for call, P for put).
            target_tte: The target time to expiration in minutes.
            moneyness: The moneyness type (ATM, OTM, ITM).
            train_split: The fraction of data to use for training.
            val_split: The fraction of data to use for validation.
            strike_band: The band to select the strike price for OTM or ITM options.
            volatility_type: The type of historical volatility to use.
            volatility_scaled: Whether to scale strike selection by the volatility.
            volatility_scalar: The scalar to multiply the volatility.
            validate_contracts: Whether to validate contracts by requesting the data from ThetaData API.
            seq_len: The sequence length of the lookback widow.
            pred_len: The prediction length of the forecast window.
            scaling: Whether to apply normalization.
            core_feats: The core features to use for the model.
            tte_feats: The time-to-expiration features to use for the model.
            datetime_feats: The datetime features to use for the model.
            keep_datetime: Whether to keep the datetime column in the dataset.
            target_channels: The target channels to use for the model.
            target_type: The type of target. Options: "multistep", "average", or "average_direction".
            batch_size: The batch size for the data loaders.
            shuffle: Whether to shuffle the data.
            drop_last: Whether to drop the last batch if it is smaller than the batch size.
            num_workers: The number of workers for the data loaders.
            prefetch_factor: The number of batches to prefetch.
            pin_memory: Whether to pin memory for the data loaders.
            persistent_workers: Whether to use persistent workers for the data loaders.
            clean_up: Whether to clean up the data after loading.
            offline: Whether to use offline mode.
            save_dir: The directory to save the data.
            verbose: Whether to print verbose output.
            dev_mode: Whether to run in development mode.
            download_only: Whether to only download the data without running an experiment.

        Returns:
            None
        """

        if download_only:
            offline = clean_up = validate_contracts = False
            self.print_master(f"Downloading data only without running an experiment.")

        self.print_master("Generating contract datasets...")
        (
            self.train_contract_dataset,
            self.val_contract_dataset,
            self.test_contract_dataset,
        ) = get_contract_datasets(
            root=root,
            start_date=start_date,
            end_date=end_date,
            contract_stride=contract_stride,
            interval_min=interval_min,
            right=right,
            target_tte=target_tte,
            tte_tolerance=tte_tolerance,
            moneyness=moneyness,
            strike_band=strike_band,
            volatility_type=volatility_type,
            volatility_scaled=volatility_scaled,
            volatility_scalar=volatility_scalar,
            train_split=train_split,
            val_split=val_split,
            clean_up=clean_up,
            offline=offline,
            save_dir=save_dir,
            verbose=verbose,
            dev_mode=dev_mode,
        )

        if validate_contracts or download_only:
            self.print_master("Validating contracts with ThetaData API...")
            self.train_contract_dataset = get_forecasting_dataset(
                contract_dataset=self.train_contract_dataset,
                tte_tolerance=tte_tolerance,
                validate_contracts=validate_contracts,
                download_only=download_only,
                verbose=verbose,
                save_dir=save_dir,
                dev_mode=dev_mode,
            )

            self.val_contract_dataset = get_forecasting_dataset(
                contract_dataset=self.val_contract_dataset,
                tte_tolerance=tte_tolerance,
                validate_contracts=validate_contracts,
                download_only=download_only,
                verbose=verbose,
                save_dir=save_dir,
                dev_mode=dev_mode,
            )
            self.test_contract_dataset = get_forecasting_dataset(
                contract_dataset=self.test_contract_dataset,
                tte_tolerance=tte_tolerance,
                validate_contracts=validate_contracts,
                download_only=download_only,
                verbose=verbose,
                save_dir=save_dir,
                dev_mode=dev_mode,
            )

        if download_only:
            return

        self.train_loader, self.val_loader, self.test_loader, self.scaler = (
            get_forecasting_loaders(
                train_contract_dataset=self.train_contract_dataset,
                val_contract_dataset=self.val_contract_dataset,
                test_contract_dataset=self.test_contract_dataset,
                seq_len=seq_len,
                pred_len=pred_len,
                tte_tolerance=tte_tolerance,
                core_feats=core_feats,
                tte_feats=tte_feats,
                datetime_feats=datetime_feats,
                keep_datetime=keep_datetime,
                target_channels=target_channels,
                target_type=target_type,
                batch_size=batch_size,
                shuffle=shuffle,
                drop_last=drop_last,
                num_workers=num_workers,
                prefetch_factor=prefetch_factor,
                pin_memory=pin_memory,
                persistent_workers=persistent_workers,
                clean_up=clean_up,
                offline=offline,
                save_dir=save_dir,
                verbose=verbose,
                scaling=scaling,
                intraday=False,
                dtype=dtype,
                dev_mode=dev_mode,
            )
        )

        self.print_master(
            f"PyTorch Dataloaders initialized. Training examples: {len(self.train_loader.dataset)}. Validation examples: {len(self.val_loader.dataset)}. Test examples: {len(self.test_loader.dataset)}."
        )

    def set_seed(self) -> None:
        """Fixes a seed for reproducibility purposes."""
        # Reproducibility
        torch.manual_seed(self.seed)  # CPU
        np.random.seed(self.seed)  # Numpy
        random.seed(self.seed)  # Python
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)  # GPU
            torch.cuda.manual_seed_all(self.seed)  # multi-GPU

    def init_earlystopping(self, path: str, patience: int) -> None:
        self.early_stopping = EarlyStopping(
            patience=patience,
            path=path,
        )

    def train(
        self,
        model: Union[nn.Module, BaseEstimator],
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        device=None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        metrics: List[str] = ["loss"],
        best_model_path: Optional[str] = None,
        early_stopping: bool = False,
        patience: Optional[int] = None,
        scheduler: Optional[_LRScheduler] = None,
    ) -> nn.Module:
        """
            Trains a model.

        Args:
            model: The model to train.
            optimizer: The optimizer to use.
            criterion: The loss function.
            train_loader: The training data loader.
            num_epochs: The number of epochs to train.
            val_loader: The validation data loader.
            metrics: The evaluation metrics to track.
            best_model_path: The path to save the best model.
            early_stopping: Whether to use early stopping.
            patience: The number of epochs to wait before stopping.
            scheduler: The learning rate scheduler.
        """

        if device is None:
            device = self.device

        if train_loader is None:
            assert hasattr(
                self, "train_loader"
            ), "A train_loader must be given or initialized using init_loaders() method."
            train_loader = self.train_loader

        if val_loader is None:
            assert hasattr(
                self, "val_loader"
            ), "A val_loader must be given or initialized using init_loaders() method."
            val_loader = self.val_loader

        if best_model_path is None:
            self.best_model_path = self.log_dir / "model.pth"
        else:
            self.best_model_path = best_model_path

        # Deep learning (PyTorch) pipeline
        num_examples = len(train_loader.dataset)
        self.print_master(f"Training on {num_examples} examples...")

        self.best_val_metric = float("inf")
        if early_stopping:
            assert (
                patience is not None
            ), "Patience must be specified for early stopping."
            self.init_earlystopping(patience=patience, path=self.best_model_path)
            self.print_master("Early stopping initialized.")

        # <--------------- Training --------------->
        for epoch in range(num_epochs):
            model.train()
            total_loss = torch.tensor(0.0, device=device)
            running_loss = torch.tensor(0.0, device=device)
            running_num_examples = torch.tensor(0.0, device=device)
            start_time = time.time()

            for i, batch in enumerate(train_loader):
                x = batch[0].to(device)
                y = batch[1].to(device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)

                # Metrics
                num_batch_examples = torch.tensor(batch[0].shape[0], device=device)
                total_loss += loss * num_batch_examples
                running_loss += loss * num_batch_examples
                running_num_examples += num_batch_examples

                # Update model parameters
                loss.backward()
                optimizer.step()

                # Periodic Logging
                if (i + 1) % 100 == 0:

                    # Loss metrics
                    loss_tensor = running_loss.to(device)
                    num_examples_tensor = running_num_examples.to(device)
                    end_time = time.time()

                    # Only rank 0 prints and logs details
                    average_loss = loss_tensor.item() / num_examples_tensor.item()
                    self.print_master(
                        f"[Epoch {epoch}, Batch ({i+1}/{len(train_loader)})]: {end_time - start_time:.3f}s. Loss: {average_loss:.6f}"
                    )

                    # Reset trackers
                    running_loss = torch.tensor(0.0, device=device)
                    running_num_examples = torch.tensor(0.0, device=device)
                    start_time = time.time()

                if scheduler:
                    scheduler.step()

            # Average Loss + Logging
            epoch_loss = total_loss.item() / num_examples
            self.print_master(f"Epoch {epoch}. Training loss: {epoch_loss:.6f}.")
            self.epoch_logger(self.logger, "train/loss", epoch_loss)

            # <--------------- Validation --------------->
            if val_loader:
                self.validate(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    metrics=metrics,
                    epoch=epoch,
                    best_model_path=self.best_model_path,
                    early_stopping=early_stopping,
                    device=device,
                )

            # Early stopping
            if early_stopping and self.early_stopping.early_stop:
                self.print_master("EarlyStopping activated, ending training.")
                break

            # Checkpoint (online)
            if self.logging == "neptune":
                pass
            else:
                run_time = time.time() - self.start_time
                self.logger["parameters/running_time"] = format_time_dynamic(run_time)

        # Upload best model to Neptune
        if self.logging == "neptune":
            self.print_master("Uploading best model to Neptune.")
            self.logger[f"model_checkpoints/{self.exp_id}"].upload(best_model_path)

        return model

    def validate(
        self,
        model: Union[nn.Module, BaseEstimator],
        val_loader: DataLoader,
        criterion: nn.Module,
        device,
        best_model_metric: str = "loss",
        metrics: List[str] = ["loss"],
        epoch: Optional[int] = None,
        best_model_path: Optional[str] = None,
        early_stopping: Optional[bool] = False,
    ) -> None:
        """
        Validates the model.

        Args:
            model: The model to validate.
            val_loader: The validation data loader.
            criterion: The loss function.
            best_model_metric: The metric to use for the best model.
            metrics: The evaluation metrics to track.
            epoch: The current epoch.
            best_model_path: The path to save the best model.
            early_stopping: Whether to use early stopping

        Returns:
            None

        """
        stats = self.evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            metrics=metrics,
            device=device,
        )

        val_loss = stats["loss"]
        self.log_stats(stats=stats, metrics=metrics, mode="val")

        if best_model_metric == "loss":
            val_metric = val_loss
        elif best_model_metric in {"acc"}:
            val_metric = -stats[best_model_metric]
        else:
            raise ValueError(f"Invalid best model metric: {best_model_metric}")

        # Save best model and apply early stopping
        if early_stopping:
            self.early_stopping(val_metric, model)
        else:
            if val_metric < self.best_val_metric:
                if best_model_metric == "loss":
                    self.print_master(
                        f"Validation loss decreased ({self.best_val_metric:.6f} --> {val_metric:.6f})."
                    )
                path_dir = Path(best_model_path).parent.absolute()
                if not path_dir.is_dir():
                    path_dir.mkdir(parents=True)
                torch.save(model.state_dict(), best_model_path)
                self.print_master(f"Saving Model Weights at: {best_model_path}...")
                self.best_val_metric = val_metric

        self.print_master("Validation complete")

    def test(
        self,
        model: Union[nn.Module, BaseEstimator],
        criterion: nn.Module,
        device=None,
        test_loader: Optional[DataLoader] = None,
        metrics: List[str] = ["loss"],
    ) -> None:
        """

        Tests the model.

        Args:
            model: The model to test.
            test_loader: The test data loader.
            criterion: The loss function.
            metrics: The evaluation metrics to track.

        Returns:
            None

        """
        if device is None:
            device = self.device

        if test_loader is None:
            assert hasattr(
                self, "test_loader"
            ), "A test_loader must be given or initialized using init_loaders() method."
            test_loader = self.test_loader

        # Load best model
        model_weights = torch.load(self.best_model_path)
        model.load_state_dict(model_weights)

        # Test set evaluation
        stats = self.evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            metrics=metrics,
            device=device,
        )

        self.log_stats(stats=stats, metrics=metrics, mode="test")

    def evaluate(
        self,
        model: Union[nn.Module, BaseEstimator],
        loader: DataLoader,
        criterion: nn.Module,
        device,
        metrics: List[str] = ["loss"],
    ) -> dict:
        """
        Evaluates the model and returns metrics. Used in validation and testing.

        Args:
            model: The model to evaluate.
            loader: The data loader.
            criterion: The loss function.
            metrics: The evaluation metrics to track.

        Returns:
            stats: A dictionary of evaluation metrics.
        """
        self.print_master("Evaluating...")
        stats = dict()
        mae_loss = nn.L1Loss()
        num_examples = len(loader.dataset)  # Total number of examples across all ranks

        model.eval()
        with torch.no_grad():

            # Initialize Metrics
            total_loss = torch.tensor(0.0, device=device)
            total_mae = torch.tensor(0.0, device=device)

            for i, batch in enumerate(loader):
                x = batch[0].to(device)
                y = batch[1].to(device)
                output = model(x)
                loss = criterion(output, y)
                num_batch_examples = torch.tensor(batch[0].shape[0], device=device)
                total_loss += loss * num_batch_examples

                # MAE
                if "mae" in metrics:
                    total_mae += (
                        mae_loss(output, batch[1].to(device)) * num_batch_examples
                    )

        # Loss
        if "loss" in metrics:
            stats["loss"] = total_loss.item() / num_examples

        # MAE
        if "mae" in metrics:
            stats["mae"] = total_mae.item() / num_examples

        return stats

    def log_stats(self, stats: dict, metrics: List[str], mode: str):
        modes = {"val": "Validation", "test": "Test"}
        Mode = modes[mode]

        loss = stats["loss"]
        self.print_master(f"Model {Mode} Loss: {loss:.6f}")
        self.epoch_logger(self.logger, f"{mode}/loss", loss)

        if "mae" in metrics:
            mae_value = stats["mae"]
            self.print_master(f"Model {Mode} MAE: {mae_value:.6f}")
            self.epoch_logger(self.logger, f"{mode}/mae", mae_value)

    def epoch_logger(self, logger, key: str, value: str) -> None:
        if self.logging == "neptune":
            logger[key].append(value)
        elif self.logging == "offline":
            if key not in logger:
                logger[key] = []
            logger[key].append(value)
        else:
            raise ValueError(f"Invalid logging method: {self.logging}.")

    def print_master(self, message: str):
        """
        Prints statements to the rank 0 node.
        """
        self.ctx.log(message)
