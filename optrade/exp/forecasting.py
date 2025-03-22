import json
import warnings
import random
import time
import neptune
from typing import Optional, List
from pathlib import Path
from rich.console import Console

warnings.filterwarnings(
    "ignore", message="h5py not installed, hdf5 features will not be supported."
)

# Torch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler

# Custom Modules
from optrade.utils.train import EarlyStopping
from optrade.utils.misc import format_time_dynamic, generate_random_id


class Experiment:
    def __init__(
        self,
        logdir: str = "logs",
        logging: str = "offline",
        seed: int = 42,
        ablation_id: Optional[int] = None,
        exp_id: str = generate_random_id(length=6),
        neptune_project_name: Optional[str] = None,
        neptune_api_token: Optional[str] = None,
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

        if ablation_id:
            self.log_dir = Path("logs") / exp_id / str(ablation_id)
        else:
            self.log_dir = Path("logs") / exp_id

        self.log_dir = Path(self.log_dir) / self.log_dir

        self.init_logger(
            logdir=logdir,
            ablation_id=ablation_id,
            exp_id=exp_id,
        )

        self.set_seed()

    def run(self):
        # Stop Logger
        if self.logging == "neptune":
            self.logger.stop()
        elif self.logging == "offline":
            # Save offline logging to JSON file
            with open(self.log_file, "w") as f:
                json.dump(self.logger, f, indent=2)
        else:
            raise ValueError(f"Invalid logging method: {self.logging}.")

    def set_seed(self) -> None:
        """Fixes a seed for reproducibility purposes."""
        # Reproducibility
        torch.manual_seed(self.seed)  # CPU
        np.random.seed(self.seed)  # Numpy
        random.seed(self.seed)  # Python
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.seed)  # GPU
            torch.cuda.manual_seed_all(self.seed)  # multi-GPU

    def init_logger(
        self,
        logdir: str,
        ablation_id: int,
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

    def init_earlystopping(self, path: str, patience: int) -> None:
        self.early_stopping = EarlyStopping(
            patience=patience,
            path=path,
        )

    def train(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        train_loader: DataLoader,
        num_epochs: int,
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

        if best_model_path is None:
            best_model_path = self.log_dir / "model.pth"

        # Deep learning (PyTorch) pipeline
        num_examples = len(train_loader.dataset)
        self.print_master(f"Training on {num_examples} examples...")

        self.best_val_metric = float("inf")
        if early_stopping:
            assert (
                patience is not None
            ), "Patience must be specified for early stopping."
            self.init_earlystopping(patience=patience, path=best_model_path)
            self.print_master("Early stopping initialized.")

        # <--------------- Training --------------->
        for epoch in range(num_epochs):
            model.train()
            total_loss = torch.tensor(0.0, device=model.device)
            running_loss = torch.tensor(0.0, device=model.device)
            running_num_examples = torch.tensor(0.0, device=model.device)
            start_time = time.time()

            for i, batch in enumerate(train_loader):
                x = batch[0].to(model.device)
                y = batch[1].to(model.device)
                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)

                # Metrics
                num_batch_examples = torch.tensor(
                    batch[0].shape[0], device=model.device
                )
                total_loss += loss * num_batch_examples
                running_loss += loss * num_batch_examples
                running_num_examples += num_batch_examples

                # Update model parameters
                loss.backward()
                optimizer.step()

                # Periodic Logging
                if (i + 1) % 100 == 0:

                    # Loss metrics
                    loss_tensor = running_loss.to(model.device)
                    num_examples_tensor = running_num_examples.to(model.device)
                    end_time = time.time()

                    # Only rank 0 prints and logs details
                    average_loss = loss_tensor.item() / num_examples_tensor.item()
                    self.print_master(
                        f"[Epoch {epoch}, Batch ({i+1}/{len(train_loader)})]: {end_time - start_time:.3f}s. Loss: {average_loss:.6f}"
                    )

                    # Reset trackers
                    running_loss = torch.tensor(0.0, device=model.device)
                    running_num_examples = torch.tensor(0.0, device=model.device)
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
                    best_model_path=best_model_path,
                    early_stopping=early_stopping,
                )

            # Early stopping
            if early_stopping and self.early_stopping.early_stop:
                self.print_master("EarlyStopping activated, ending training.")
                break

            # Checkpoint (online)
            if self.logging=="neptune":
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
        model: nn.Module,
        val_loader: DataLoader,
        criterion: nn.Module,
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
            model=model, loader=val_loader, criterion=criterion, metrics=metrics
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
        model: nn.Module,
        test_loader: DataLoader,
        criterion: nn.Module,
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

        # Load best model
        model_weights = torch.load(self.best_model_path)
        model.load_state_dict(model_weights)

        # Test set evaluation
        stats = self.evaluate(
            model=model, loader=test_loader, criterion=criterion, metrics=metrics
        )

        self.log_stats(stats=stats, metrics=metrics, mode="test")

    def evaluate(
        self,
        model: nn.Module,
        loader: DataLoader,
        criterion: nn.Module,
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
            total_loss = torch.tensor(0.0, device=model.device)
            total_mae = torch.tensor(0.0, device=model.device)

            for i, batch in enumerate(loader):
                x = batch[0].to(model.device)
                y = batch[1].to(model.device)
                output = model(x)
                loss = criterion(output, y)
                num_batch_examples = torch.tensor(
                    batch[0].shape[0], device=model.device
                )
                total_loss += loss * num_batch_examples

                # MAE
                if "mae" in metrics:
                    total_mae += (
                        mae_loss(output, batch[1].to(model.device)) * num_batch_examples
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
