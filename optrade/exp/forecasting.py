import json
import warnings
import random
import time
import neptune
import joblib
from typing import Optional, List, Tuple, Union
from pathlib import Path
from rich.console import Console
from rich.table import Table
from sklearn.base import BaseEstimator
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.model_selection import TimeSeriesSplit

# Torch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

# Custom Modules
from optrade.data.contracts import get_contract_datasets
from optrade.data.forecasting import get_forecasting_dataset, get_forecasting_loaders
from optrade.utils.train import EarlyStopping
from optrade.utils.misc import format_time_dynamic, generate_random_id
from optrade.exp.evaluate import get_metrics
from optrade.data.universe import Universe

warnings.filterwarnings(
    "ignore", message="h5py not installed, hdf5 features will not be supported."
)


class Experiment:
    def __init__(
        self,
        log_dir: Union[str, Path] = "logs",
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
        vol_feats: Optional[List[str]] = None,
        rolling_volatility_range: Optional[List[int]] = None,
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
        prefetch_factor: Optional[int] = 2,
        pin_memory: bool = torch.cuda.is_available(),
        persistent_workers: bool = True,
        clean_up: bool = False,
        offline: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        validate_contracts: bool = True,
        modify_contracts: bool = False,
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
            modify_contracts: Whether to overwite contracts .pkl files if certain contracts are invalid.
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
            self.print_master("Download only mode (no experiments)")
            offline = clean_up = validate_contracts = False

            if not modify_contracts:
                self.print_master(
                    "modify_contracts is set to False. You must disable it to activately modify contracts .pkl files."
                )

        action = "Loading" if offline else "Generating"
        self.ctx.log(f"{action} contract datasets")
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

        self.ctx.log(f"Train contracts: {len(self.train_contract_dataset)}")
        self.ctx.log(f"Validation contracts:{len(self.val_contract_dataset)}")
        self.ctx.log(f"Test contracts:{len(self.test_contract_dataset)}")

        # Warning: this will overwite existing contract .pkl files
        if not modify_contracts:
            self.ctx.log(
                "Safe mode enabled (modify_contracts=False): contracts will not be modified."
            )
        else:
            self.ctx.log(
                "Warning (modify_contracts=True): contracts will be overwritten if invalid."
            )

        if validate_contracts or download_only:
            action = "Validating contracts" if validate_contracts else "Downloading data"
            self.ctx.log(f"{action} with ThetaData API...")
            self.train_contract_dataset = get_forecasting_dataset(
                contract_dataset=self.train_contract_dataset,
                tte_tolerance=tte_tolerance,
                validate_contracts=validate_contracts,
                download_only=download_only,
                modify_contracts=modify_contracts,
                verbose=verbose,
                save_dir=save_dir,
                dev_mode=dev_mode,
            )

            self.val_contract_dataset = get_forecasting_dataset(
                contract_dataset=self.val_contract_dataset,
                tte_tolerance=tte_tolerance,
                validate_contracts=validate_contracts,
                download_only=download_only,
                modify_contracts=modify_contracts,
                verbose=verbose,
                save_dir=save_dir,
                dev_mode=dev_mode,
            )
            self.test_contract_dataset = get_forecasting_dataset(
                contract_dataset=self.test_contract_dataset,
                tte_tolerance=tte_tolerance,
                validate_contracts=validate_contracts,
                download_only=download_only,
                modify_contracts=modify_contracts,
                verbose=verbose,
                save_dir=save_dir,
                dev_mode=dev_mode,
            )

        self.ctx.log(f"Train contracts (validated): {len(self.train_contract_dataset)}")
        self.ctx.log(f"Validation contracts (validated):{len(self.val_contract_dataset)}")
        self.ctx.log(f"Test contracts (validated):{len(self.test_contract_dataset)}")

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
                vol_feats=vol_feats,
                rolling_volatility_range=rolling_volatility_range,
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
                modify_contracts=modify_contracts,
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

    def get_sklearn_data(
        self,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        test_loader: Optional[DataLoader] = None,
        has_datetime: bool = False,
    ) -> None:

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
        if test_loader is None:
            assert hasattr(
                self, "test_loader"
            ), "A test_loader must be given or initialized using init_loaders() method."
            test_loader = self.test_loader

        # Iterate through the ConcatDataset list of datasets
        train_x = []; train_y = []; train_dtx = []; train_dty = []
        for dataset in train_loader.dataset.datasets:
            numpy_dataset = dataset.to_numpy()
            train_x.append(numpy_dataset[0]); train_y.append(numpy_dataset[1])
            if has_datetime:
                train_dtx.append(numpy_dataset[2]); train_dty.append(numpy_dataset[3])

        val_x = []; val_y = []; val_dtx = []; val_dty = []
        for dataset in val_loader.dataset.datasets:
            numpy_dataset = dataset.to_numpy()
            val_x.append(numpy_dataset[0]); val_y.append(numpy_dataset[1])
            if has_datetime:
                val_dtx.append(numpy_dataset[2]); val_dty.append(numpy_dataset[3])

        test_x = []; test_y = []; test_dtx = []; test_dty = []
        for dataset in test_loader.dataset.datasets:
            numpy_dataset = dataset.to_numpy()
            test_x.append(numpy_dataset[0]); test_y.append(numpy_dataset[1])
            if has_datetime:
                test_dtx.append(numpy_dataset[2]); test_dty.append(numpy_dataset[3])

        # Concatenate the numpy arrays
        train_x = np.concatenate(train_x + val_x, axis=0)
        train_y = np.concatenate(train_y + val_y, axis=0)
        test_x = np.concatenate(test_x, axis=0)
        test_y = np.concatenate(test_y, axis=0)

        # Flatten the last two dimensions of the input: (num_windows, seq_len, num_features) -> (num_windows, seq_len * num_features)
        if train_x.ndim == 3:
            train_x = train_x.reshape(train_x.shape[0], -1)
            test_x = test_x.reshape(test_x.shape[0], -1)

        # For multi-feature targets, flatten the last two dimensions: (num_windows, pred_len, num_target_features) -> (num_windows, pred_len * num_target_features)
        if train_y.ndim == 3:
            train_y = train_y.reshape(train_y.shape[0], -1)
            test_y = test_y.reshape(test_y.shape[0], -1)

        self.sklearn_data = {
            "train_x": train_x,
            "train_y": train_y,
            "test_x": test_x,
            "test_y": test_y,
        }

        if has_datetime:
            train_dtx = np.concatenate(train_dtx + val_dtx, axis=0)
            train_dty = np.concatenate(train_dty + val_dty, axis=0)
            test_dtx = np.concatenate(test_dtx, axis=0)
            test_dty = np.concatenate(test_dty, axis=0)

            self.sklearn_dt_data = {
                "train_dtx": train_dtx,
                "train_dty": train_dty,
                "test_dtx": test_dtx,
                "test_dty": test_dty,
            }

        return

    def train_sklearn(
        self,
        model: BaseEstimator,
        param_dict: dict,
        tuning_method: str = "grid",
        n_splits: int = 5,
        verbose: int = 1,
        n_jobs: int = -1,
        n_iter: int = 100,
        train_x: Optional[np.ndarray] = None,
        train_y: Optional[np.ndarray] = None,
        target_type: str = "multistep",
        best_model_path: Optional[str] = None,
        early_stopping: bool = False,
        patience: Optional[int] = None,
    ) -> None:

        # Temporally split the data into train and validation sets
        cv = TimeSeriesSplit(n_splits=n_splits)

        # Convert PyTorch DataLoader objects to single NumPy arrays for scikit-learn training
        if all(v is None for v in [train_x, train_y]):

            with self.ctx.status("Converting PyTorch DataLoader to NumPy arrays..."):
                has_datetime = self.train_loader.dataset.datasets[0].has_datetime # Check if datetime is on for the first ForecastingDataset
                self.get_sklearn_data(
                    train_loader=self.train_loader,
                    val_loader=self.val_loader,
                    test_loader=self.test_loader,
                    has_datetime=has_datetime,
                )
                train_x = self.sklearn_data["train_x"]
                train_y = self.sklearn_data["train_y"]

            self.ctx.log(f"train_x shape: {train_x.shape}")
            self.ctx.log(f"train_y shape: {train_y.shape}")

            # Check if NaNs in train_x or train_y
            if np.isnan(train_x).any():
                # Compute how many NaNs there are
                num_nans = np.isnan(train_x).sum()
                self.ctx.log(f"train_x contains {num_nans} NaN values.")

            if np.isnan(train_y).any():
                num_nans = np.isnan(train_y).sum()
                self.ctx.log(f"train_y contains {num_nans} NaN values.")

            if np.isnan(train_x).any() or np.isnan(train_y).any():
                raise ValueError("train_x or train_y contains NaN values.")

        if target_type in ["multistep", "average"]:
            scoring = "neg_root_mean_squared_error"
        elif target_type == "average_direction":
            scoring = "accuracy"
        else:
            raise ValueError(f"Invalid target type: {target_type}")

        if tuning_method == "grid":
            clf = GridSearchCV(
                estimator=model,
                param_grid=param_dict,
                scoring=scoring,
                n_jobs=n_jobs,
                cv=cv,
                verbose=verbose,
            )
        elif tuning_method == "random":
            clf = RandomizedSearchCV(
                estimator=model,
                param_distributions=param_dict,
                scoring=scoring,
                n_jobs=n_jobs,
                cv=cv,
                verbose=verbose,
                n_iter=n_iter,
            )
        else:
            raise ValueError(f"Invalid tuning method: {tuning_method}")


        # Print out model param_dict with Rich table
        table = Table(title="Hyperparameter Grid")
        table.add_column("Parameter", style="bold cyan")
        table.add_column("Values", style="magenta")
        for k, v in param_dict.items():
            table.add_row(str(k), str(v))
        self.ctx.log(table)

        with self.ctx.status("Fitting sklearn model..."):
            search = clf.fit(X=train_x, y=train_y)

        # Save model
        best_model = search.best_estimator_
        if best_model_path is None:
            self.best_model_path = self.log_dir / "model.pkl"
        else:
            self.best_model_path = Path(best_model_path)

        # Create directory if it doesn't exist
        self.best_model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(best_model, self.best_model_path) # Save the best model

        # TODO: Add model.upload() for neptune logging
        self.print_master(f"Best model saved to {self.best_model_path}")

        # Log best model parameters
        self.logger["best_model/params"] = search.best_params_
        self.logger["best_model/score"] = search.best_score_

    def test_sklearn(
        self,
        metrics: List[str],
        target_type: str = "multistep",
        test_x: Optional[np.ndarray] = None,
        test_y: Optional[np.ndarray] = None,
        best_model: Optional[BaseEstimator] = None,
    ) -> None:

        if best_model is None:
            best_model = joblib.load(self.best_model_path)

        # Evaluate best model on the test set
        if all(v is None for v in [test_x, test_y]):
            test_x = self.sklearn_data["test_x"]
            test_y = self.sklearn_data["test_y"]

        test_preds = best_model.predict(X=test_x)
        test_metrics, metric_keys = get_metrics(
            target=test_y,
            output=test_preds,
            metrics=metrics,
            target_type=target_type,
        )
        train_preds = best_model.predict(X=self.sklearn_data["train_x"])
        train_metrics, _ = get_metrics(
            target=self.sklearn_data["train_y"],
            output=train_preds,
            metrics=metrics,
            target_type=target_type,
        )

        # Return metrics in dictionary format
        test_stats = dict()
        train_stats = dict()
        for i, metric in enumerate(metric_keys):
            test_stats[metric] = test_metrics[i]
            train_stats[metric] = train_metrics[i]

        self.log_stats(stats=test_stats, metrics=metrics, mode="test")
        self.log_stats(stats=train_stats, metrics=metrics, mode="train")

    def train_torch(
        self,
        model: Union[nn.Module, BaseEstimator],
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        num_epochs: int,
        device=None,
        train_loader: Optional[DataLoader] = None,
        val_loader: Optional[DataLoader] = None,
        metrics: List[str] = ["mse"],
        best_model_metric: str = "mse",
        best_model_path: Optional[str] = None,
        early_stopping: bool = False,
        patience: Optional[int] = None,
        scheduler: Optional[_LRScheduler] = None,
        target_type: str = "multistep",
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
        self.print_master(f"Training on {num_examples} examples")

        self.best_val_metric = float("inf")
        if early_stopping:
            assert (
                patience is not None
            ), "Patience must be specified for early stopping."
            self.init_earlystopping(patience=patience, path=self.best_model_path)
            self.print_master("Early stopping initialized")

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
                self.validate_torch(
                    model=model,
                    val_loader=val_loader,
                    criterion=criterion,
                    metrics=metrics,
                    epoch=epoch,
                    best_model_metric=best_model_metric,
                    best_model_path=self.best_model_path,
                    early_stopping=early_stopping,
                    device=device,
                    target_type=target_type,
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

    def validate_torch(
        self,
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
        device,
        best_model_metric: str = "mse",
        metrics: List[str] = ["mse"],
        epoch: Optional[int] = None,
        best_model_path: Optional[str] = None,
        early_stopping: Optional[bool] = False,
        target_type: str = "multistep",
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
        stats = self.evaluate_torch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            metrics=metrics,
            device=device,
            target_type=target_type,
        )
        self.log_stats(stats=stats, metrics=metrics, mode="val")

        if best_model_metric in {"mse", "mae", "rmse", "mape"}:
            val_metric = stats[best_model_metric] # Minimizing metrics
        elif best_model_metric in {"r2", "accuracy", "precision", "recall", "f1", "auc"}:
            val_metric = -stats[best_model_metric] # Maximizing metrics
        else:
            raise ValueError(f"Invalid best model metric: {best_model_metric}")

        # Save best model and apply early stopping
        if early_stopping:
            self.early_stopping(val_metric, model)
        else:
            if val_metric < self.best_val_metric:
                self.print_master(
                    f"Validation {best_model_metric} decreased ({self.best_val_metric:.6f} --> {val_metric:.6f})."
                )
                path_dir = Path(best_model_path).parent.absolute()
                if not path_dir.is_dir():
                    path_dir.mkdir(parents=True)
                torch.save(model.state_dict(), best_model_path)
                self.print_master(f"Saving Model Weights at: {best_model_path}...")
                self.best_val_metric = val_metric

        self.print_master("Validation complete")

    def test_torch(
        self,
        model: Union[nn.Module, BaseEstimator],
        criterion: nn.Module,
        device=None,
        test_loader: Optional[DataLoader] = None,
        metrics: List[str] = ["mse"],
        target_type: str = "multistep",
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
        stats = self.evaluate_torch(
            model=model,
            loader=test_loader,
            criterion=criterion,
            metrics=metrics,
            device=device,
            target_type=target_type,
        )

        self.log_stats(stats=stats, metrics=metrics, mode="test")

    def evaluate(
        self,
        model: Union[nn.Module, BaseEstimator],
        loader: DataLoader,
        criterion: nn.Module,
        device,
        metrics: List[str] = ["mse"],
        target_type: str = "multistep",
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
        num_examples = len(loader.dataset)  # Total number of examples across all ranks

        model.eval()
        with torch.no_grad():

            # Initialize Metrics
            total_metrics = np.zeros(len(metrics))

            for i, batch in enumerate(loader):
                x = batch[0].to(device)
                y = batch[1].to(device)
                output = model(x)

                # Compute metrics
                num_batch_examples = batch[0].shape[0]
                batch_metrics, metric_keys = get_metrics(
                    metrics=metrics,
                    output=output,
                    target=y,
                    target_type=target_type,
                )
                total_metrics +=  batch_metrics * num_batch_examples

        # Return metrics in dictionary format
        stats = dict()
        for i, metric in enumerate(metric_keys):
            stats[metric] = total_metrics[i] / num_examples

        return stats

    def log_universe(
        self,
        market_metrics,
        root: str,
        parent: str,
    ) -> None:
        """
        Logs the universe to the logger.

        Args:
            root: The root directory containing the data.
            universe: The Universe object.
            parent: The parent experiment ID.

        Returns:
            None
        """
        pass

        # Find the parent experiment file
        # market_metrics =

        # if self.logging == "neptune":
        #     self.logger["universe"].upload(universe)
        # elif self.logging == "offline":
        #     with open(self.log_file, "w") as f:
        #         json.dump(universe, f, indent=2)
        # else:
        #     raise ValueError(f"Invalid logging method: {self.logging}.")

    def log_stats(self, stats: dict, metrics: List[str], mode: str):
        modes = {"val": "Validation", "test": "Test", "train": "Train"}
        Mode = modes[mode]

        for metric in metrics:
            self.print_master(f"{Mode} {metric}: {stats[metric]:.6f}")
            self.epoch_logger(self.logger, f"{mode}/{metric}", stats[metric])

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
