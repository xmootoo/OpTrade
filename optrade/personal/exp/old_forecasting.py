import os
import sys
import datetime
import gc
import json
import warnings
import shutil
import random
import time
import neptune
import yaml

warnings.filterwarnings(
    "ignore", message="h5py not installed, hdf5 features will not be supported."
)

# Rich console
from rich.console import Console
from rich.pretty import pprint

# Pydantic
from pydantic import BaseModel
from typing import Any

# Torch
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Custom Modules
from optrade.utils.train import EarlyStopping
from optrade.utils.misc import format_time_dynamic
from optrade.data.forecasting import get_forecasting_loaders
from optrade.utils.personal.models import (
    get_criterion,
    get_model,
    get_optim,
    get_scheduler,
    compute_loss,
    model_update,
    forward_pass,
)

from optrade.utils.personal.logger import log_pydantic, epoch_logger


class Experiment:
    def __init__(self, args: BaseModel) -> None:
        self.args = args
        self.start_time = time.time()

    def run(self):
        # Rich Console
        self.ctx = Console()

        # Reproducibility
        self.generator = torch.Generator().manual_seed(self.args.exp.seed)
        torch.manual_seed(self.args.exp.seed)  # CPU
        np.random.seed(self.args.exp.seed)  # Numpy
        random.seed(self.args.exp.seed)  # Python
        if torch.cuda.is_available():
            torch.cuda.manual_seed(self.args.exp.seed)  # GPU
            torch.cuda.manual_seed_all(self.args.exp.seed)  # multi-GPU

        # Logging
        self.init_logger()

        # Initialize Device
        self.init_device()

        # Learning Type
        self.supervised_train()

        # Stop Logger
        if self.args.exp.neptune:
            self.logger.stop()
        else:
            # Save offline logging to JSON file
            with open(self.log_file, "w") as f:
                json.dump(self.logger, f, indent=2)

    def init_device(self):
        """
        Initialize CUDA (or MPS) devices.
        """
        if self.args.exp.mps:
            self.device = torch.device(
                "mps" if torch.backends.mps.is_available() else "cpu"
            )
            self.print_master("MPS hardware acceleration activated.")
        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.args.exp.gpu_id}")
        else:
            self.device = torch.device("cpu")
            self.print_master("CUDA not available. Running on CPU.")

    def init_dataloaders(self):
        """
        Initialize the dataloaders depending on the learning type and loader type.
        """

        loaders = get_forecasting_loaders(
            root=self.args.data.root,
            start_date=self.args.data.start_date,
            end_date=self.args.data.end_date,
            contract_stride=self.args.data.contract_stride,
            interval_min=self.args.data.interval_min,
            right=self.args.data.right,
            target_tte=self.args.data.target_tte,
            tte_tolerance=self.args.data.tte_tolerance,
            moneyness=self.args.data.moneyness,
            strike_band=self.args.data.strike_band,
            volatility_type=self.args.data.volatility_type,
            volatility_scaled=self.args.data.volatility_scaled,
            volatility_scalar=self.args.data.volatility_scalar,
            train_split=self.args.data.train_split,
            val_split=self.args.data.val_split,
            core_feats=self.args.data.core_feats,
            tte_feats=self.args.data.tte_feats,
            datetime_feats=self.args.data.datetime_feats,
            batch_size=self.args.train.batch_size,
            shuffle=self.args.data.shuffle,
            drop_last=self.args.data.drop_last,
            num_workers=self.args.data.num_workers,
            prefetch_factor=self.args.data.prefetch_factor,
            pin_memory=self.args.data.pin_memory,
            clean_up=self.args.data.clean_up,
            offline=self.args.data.offline,
            verbose=self.args.data.verbose,
            scaling=self.args.data.scaling,
            intraday=self.args.data.intraday,
            target_channels=self.args.data.target_channels,
            seq_len=self.args.data.seq_len,
            pred_len=self.args.data.pred_len,
            dtype=self.args.data.dtype,
        )

        self.train_loader, self.val_loader, self.test_loader = loaders[:3]

        if self.args.data.scaling:
            self.scaler = loaders[3]

    def init_model(self):
        """
        Initialize the model.
        """
        self.model = get_model(
            self.args, self.generator, self.args.data.target_channels
        )
        self.model.to(self.device)
        num_params = self.count_parameters()
        self.logger["parameters/sl/num_params"] = num_params
        self.print_master(
            f"{self.args.exp.model_id} model initialized with {num_params:,} parameters."
        )

    def init_optimizer(self):
        """
        Initialize the optimizer
        """
        self.optimizer = get_optim(self.args, self.model, self.args.train.optimizer)
        self.print_master(f"{self.args.train.optimizer} optimizer initialized.")

    def init_logger(self):
        """
        Initialize the logger
        """

        self.log_dir = os.path.join(
            "logs",
            f"{self.args.exp.ablation_id}_{self.args.exp.model_id}_{self.args.exp.id}",
            str(self.args.exp.seed),
        )

        if self.args.exp.neptune:
            # Initialize Neptune run with the time-based ID
            self.logger = neptune.init_run(
                project=self.args.exp.project_name,
                api_token=self.args.exp.api_token,
                custom_run_id=self.args.exp.time,
            )
            self.print_master("Neptune logger initialized.")
        else:
            self.logger = dict()
            self.log_file = os.path.join(self.log_dir, "log.json")

            if os.path.exists(self.log_dir):
                self.print_master(f"Using existing log directory: {self.log_dir}")
            else:
                os.makedirs(self.log_dir, exist_ok=True)
                self.print_master(f"Created new log directory: {self.log_dir}")

            self.print_master("Offline logger initialized.")

        log_pydantic(self.logger, self.args, "parameters")

    def init_earlystopping(self, path: str):
        self.early_stopping = EarlyStopping(
            patience=self.args.early_stopping.patience,
            verbose=self.args.early_stopping.verbose,
            delta=self.args.early_stopping.delta,
            path=path,
        )

    def train(
        self,
        model,
        model_id,
        optimizer,
        train_loader,
        best_model_path,
        criterion,
        val_loader=None,
        scheduler=None,
        mae=False,
        early_stopping=False,
    ):
        """
            Trains a model.

        Args:
            model (nn.Module): The model to train.
            model_id (str): The model ID.
            optimizer (torch.optim): The optimizer to use.
            train_loader (torch.utils.data.DataLoader): The training data.
            best_model_path (str): The path to save the best model.
            criterion (torch.nn): The loss function.
            val_loader (torch.utils.data.DataLoader): The validation data.
            scheduler (torch.optim.lr_scheduler): The learning rate scheduler.
            mae (bool): Whether to log Mean Absolute Error (MAE).
            early_stopping (bool): Whether to use early stopping or not, relative to validation evaluation.
        """

        # Deep learning (PyTorch) pipeline
        num_examples = len(train_loader.dataset)
        self.print_master(f"Training on {num_examples} examples...")

        self.best_val_metric = float("inf")
        if early_stopping:
            self.init_earlystopping(best_model_path)
            self.print_master("Early stopping initialized.")

        # <--------------- Training --------------->
        for epoch in range(self.args.train.epochs):
            model.train()
            total_loss = torch.tensor(0.0, device=self.device)
            running_loss = torch.tensor(0.0, device=self.device)
            running_num_examples = torch.tensor(0.0, device=self.device)
            start_time = time.time()

            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                output = forward_pass(self.args, model, batch, model_id, self.device)
                loss = compute_loss(
                    output, batch, criterion, model_id, self.args, self.device
                )

                # Metrics
                num_batch_examples = torch.tensor(batch[0].shape[0], device=self.device)
                total_loss += loss * num_batch_examples
                running_loss += loss * num_batch_examples
                running_num_examples += num_batch_examples

                # Update model parameters
                model_update(model, loss, optimizer, model_id)

                # Periodic Logging
                if (i + 1) % 100 == 0:

                    # Loss metrics
                    loss_tensor = running_loss.to(self.device)
                    num_examples_tensor = running_num_examples.to(self.device)
                    end_time = time.time()

                    # Only rank 0 prints and logs details
                    average_loss = loss_tensor.item() / num_examples_tensor.item()
                    self.print_master(
                        f"[Epoch {epoch}, Batch ({i+1}/{len(train_loader)})]: {end_time - start_time:.3f}s. Loss: {average_loss:.6f}"
                    )

                    # Reset trackers
                    running_loss = torch.tensor(0.0, device=self.device)
                    running_num_examples = torch.tensor(0.0, device=self.device)
                    start_time = time.time()

                if scheduler:
                    scheduler.step()

            # Average Loss + Logging
            epoch_loss = total_loss.item() / num_examples
            self.print_master(f"Epoch {epoch}. Training loss: {epoch_loss:.6f}.")
            epoch_logger(self.args, self.logger, "train/loss", epoch_loss)

            # <--------------- Validation --------------->
            if val_loader:
                self.validate(
                    model,
                    val_loader,
                    model_id,
                    criterion,
                    mae,
                    epoch,
                    best_model_path,
                    early_stopping,
                )

            # Early stopping
            if early_stopping and self.early_stopping.early_stop:
                self.print_master("EarlyStopping activated, ending training.")
                break

            # Checkpoint (online)
            if self.args.exp.neptune:
                pass
            else:
                run_time = time.time() - self.start_time
                self.logger["parameters/running_time"] = format_time_dynamic(run_time)

    def validate(
        self,
        model,
        val_loader,
        model_id,
        criterion,
        mae,
        epoch,
        best_model_path,
        early_stopping,
    ):
        """
        Validate the model.
        """
        stats = self.evaluate(
            model=model,
            model_id=model_id,
            loader=val_loader,
            criterion=criterion,
            mae=mae,
        )

        val_loss = stats["loss"]
        self.log_stats(stats, mae, "val")

        if self.args.exp.best_model_metric == "loss":
            val_metric = val_loss
        elif self.args.exp.best_model_metric in {"acc"}:
            val_metric = -stats[self.args.exp.best_model_metric]
        else:
            raise ValueError(
                f"Invalid best model metric: {self.args.exp.best_model_metric}"
            )

        # Save best model and apply early stopping
        if early_stopping:
            self.early_stopping(val_metric, model)
        else:
            if val_metric < self.best_val_metric:
                if self.args.exp.best_model_metric == "loss":
                    self.print_master(
                        f"Validation loss decreased ({self.best_val_metric:.6f} --> {val_metric:.6f})."
                    )
                path_dir = os.path.abspath(os.path.dirname(best_model_path))
                if not os.path.isdir(path_dir):
                    os.makedirs(path_dir)
                torch.save(model.state_dict(), best_model_path)
                self.print_master(f"Saving Model Weights at: {best_model_path}...")
                self.best_val_metric = val_metric

        self.print_master("Validation complete")

    def supervised_train(self):
        """
        Train the model in supervised mode.
        """

        # Load train loaders
        self.init_dataloaders()

        # Initialize Model and Optimizer
        self.init_model()
        self.init_optimizer()

        # Get supervised criterions
        self.criterion = get_criterion(self.args, self.args.train.criterion)
        self.print_master(f"{self.args.train.criterion} initialized.")

        # Get supervised scheduler
        if self.args.train.scheduler != "None":
            self.sl_scheduler = get_scheduler(
                self.args,
                self.args.train.scheduler,
                "supervised",
                self.optimizer,
                len(self.train_loader),
            )
        else:
            self.sl_scheduler = None
        self.print_master("Starting Supervised Training...")

        # Supervised Training
        best_model_path = os.path.join(self.log_dir, f"supervised.pth")
        self.train(
            model=self.model,
            model_id=self.args.exp.model_id,
            optimizer=self.optimizer,
            train_loader=self.train_loader,
            best_model_path=best_model_path,
            criterion=self.criterion,
            val_loader=self.val_loader,
            scheduler=self.sl_scheduler,
            mae=self.args.exp.mae,
            early_stopping=self.args.train.early_stopping,
        )

        # Upload best model to Neptune
        if self.args.exp.neptune and not self.args.exp.sklearn:
            self.print_master("Uploading best model to Neptune.")
            self.logger[f"model_checkpoints/{self.args.exp.model_id}_sl"].upload(
                best_model_path
            )

        # Test model
        self.print_master("Starting Supervised Testing...")

        self.test(
            model=self.model,
            model_id=self.args.exp.model_id,
            best_model_path=best_model_path,
            criterion=self.criterion,
            mae=self.args.exp.mae,
        )

    def test(
        self,
        model: nn.Module,
        model_id: str,
        best_model_path: str,
        criterion: nn.Module,
        mae: bool = False,
    ) -> None:

        # <---Deep learning (PyTorch) pipeline--->
        # Load best model
        model_weights = torch.load(best_model_path)
        model.load_state_dict(model_weights)

        # Test set evaluation
        stats = self.evaluate(
            model=model,
            model_id=model_id,
            loader=self.test_loader,
            criterion=criterion,
            mae=mae,
        )

        self.log_stats(stats, mae, "test")

    def evaluate(
        self,
        model: nn.Module,
        model_id: str,
        loader: DataLoader,
        criterion: nn.Module,
        mae: bool = False,
    ) -> dict:
        """
        Evaluate the model return evaluation loss and/or evaluation accuracy.
        """
        self.print_master("Evaluating...")
        stats = dict()
        mae_loss = nn.L1Loss()
        num_examples = len(loader.dataset)  # Total number of examples across all ranks

        model.eval()
        with torch.no_grad():

            # Initialize Metrics
            total_loss = torch.tensor(0.0, device=self.device)
            total_mae = torch.tensor(0.0, device=self.device)

            for i, batch in enumerate(loader):
                output = forward_pass(self.args, model, batch, model_id, self.device)
                loss = compute_loss(
                    output, batch, criterion, model_id, self.args, self.device
                )
                num_batch_examples = torch.tensor(batch[0].shape[0], device=self.device)
                total_loss += loss * num_batch_examples

                # MAE
                if mae:
                    total_mae += (
                        mae_loss(output, batch[1].to(self.device)) * num_batch_examples
                    )

        # Loss
        stats["loss"] = total_loss.item() / num_examples

        # MAE
        if mae:
            stats["mae"] = total_mae.item() / num_examples

        return stats

    def log_stats(self, stats, mae, mode):
        modes = {"val": "Validation", "test": "Test"}
        Mode = modes[mode]

        loss = stats["loss"]
        self.print_master(f"Model {Mode} Loss: {loss:.6f}")
        epoch_logger(self.args, self.logger, f"{mode}/loss", loss)

        if mae:
            mae_value = stats["mae"]
            self.print_master(f"Model {Mode} MAE: {mae_value:.6f}")
            epoch_logger(self.args, self.logger, f"{mode}/mae", mae_value)

    def count_parameters(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)

    def print_master(self, message):
        """
        Prints statements to the rank 0 node.
        """
        self.ctx.log(message)
