from copy import deepcopy
from pathlib import Path
from typing import Tuple, List, Union, Optional
from rich.console import Console
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset
from datetime import datetime, timedelta
import pandas_market_calendars as mcal


# Datasets
from optrade.data.thetadata import find_optimal_strike, get_expirations
from optrade.data.contracts import Contract, ContractDataset
from optrade.utils.error_handlers import (
    DataValidationError,
    INCOMPATIBLE_START_DATE,
    INCOMPATIBLE_END_DATE,
    NAN_FEATURES,
)

# Features
from optrade.data.features import transform_features
from optrade.utils.misc import datetime_to_tensor, tensor_to_datetime

# Get absolute path for this script
SCRIPT_DIR = Path(__file__).resolve().parent


class ForecastingDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        target_channels: Optional[List[str]] = None,
        target_type: str = "multistep",
        dtype: str = "float32",
    ) -> None:
        """
        Initializes the ForecastingDataset class.

        Args:
            data (pd.DataFrame):
                Input DataFrame containing the time series data.
            seq_len (int):
                Length of the lookback window for each sample.
            pred_len (int):
                Length of the forecast window (number of steps ahead to predict).
            target_channels (Optional[List[str]]):
                List of column names to include as target channels. If None, all columns are used.
            target_type (str):
                Type of target to predict. Must be one of:
                - "multistep": Predicts the full future sequence (regression).
                - "average": Predicts the average value over the forecast window (regression).
                - "average_direction": Predicts the sign of the average change (binary classification).
            dtype (str):
                Data type for the internal PyTorch tensors (e.g., "float32", "float64"). Default is "float32".

        Returns:
            None
        """

        # Numeric data conversion (remove datetime strings from direct features)
        self.has_datetime = "datetime" in data.columns
        if self.has_datetime:
            self.datetime = data["datetime"].values  # Store as numpy array
            data_numeric = data.drop(columns=["datetime"]).to_numpy()
        else:
            data_numeric = data.to_numpy()

        # Check for NaNs using the original DataFrame (with column names)
        columns = (
            data.drop(columns=["datetime"]).columns
            if self.has_datetime
            else data.columns
        )
        nan_counts = data[columns].isna().sum()

        if nan_counts.any():
            bad_feats = nan_counts[nan_counts > 0]
            message = (
                f"[ForecastingDataset] Found NaNs in {len(bad_feats)} feature(s):\n"
                f"{bad_feats.to_string()}\n"
                "This is often caused by illiquid instruments (e.g., no volume or quote data) "
                "or unstable features like LOB imbalance or spread when interval_min is too low.\n"
                "Consider dropping these features or resampling to a longer interval by increasing interval_min parmaeter."
            )
            raise DataValidationError(
                message=message, error_code=NAN_FEATURES, verbose=True, warning=False
            )

        self.torch_dtype = eval("torch." + dtype)
        self.np_dtype = eval("np." + dtype)
        self.data = torch.tensor(data_numeric, dtype=self.torch_dtype)
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.target_type = target_type

        if target_channels is not None and len(target_channels) > 0:
            feature_names = data.columns.to_list()
            self.target_channels_idx = [
                feature_names.index(channel) for channel in target_channels
            ]

    def __len__(self) -> int:
        """
        Returns the number of input-target pairs in the dataset.
        """
        return self.data.shape[0] - self.seq_len - self.pred_len

    def __getitem__(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
    ]:
        """Get a sample from the dataset.

        This method retrieves an input-target pair at the specified index, with input being
        the lookback window and target being the forecast window based on the target_type.

        Args:
            idx: Index of the starting point of the lookback window.

        Returns:
            If datetime is available:
                tuple: A tuple containing (input_tensor, target_tensor, input_datetime, target_datetime)
                    - input_tensor: Lookback window of shape (num_features, seq_len).
                    - target_tensor: Target window with shape depending on target_type:
                      - "multistep": (num_target_features, pred_len)
                      - "average": (num_target_features, 1)
                      - "average_direction": (num_target_features, 1)
                    - input_datetime: Datetime values for input window of shape (seq_len,).
                    - target_datetime: Datetime values for target window of shape (pred_len,).
            Otherwise:
                tuple: A tuple containing (input_tensor, target_tensor)
                    - input_tensor: Lookback window of shape (num_features, seq_len).
                    - target_tensor: Target window with shape as described above depending on `target_type` arg.
        """
        input = self.data[idx : idx + self.seq_len]

        if hasattr(self, "target_channels_idx"):
            target = self.data[
                idx + self.seq_len : idx + self.seq_len + self.pred_len,
                self.target_channels_idx,
            ]
        else:
            target = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]

        input_tensor = input.transpose(0, 1)
        target_tensor = target.transpose(0, 1)

        if self.target_type == "average":
            target_tensor = target_tensor.mean(dim=0).unsqueeze(0)
        elif self.target_type == "average_direction":
            target_tensor = (target_tensor.mean(dim=0) > 0).unsqueeze(0).float()
        elif self.target_type == "multistep":
            pass
        else:
            raise ValueError(
                "Invalid target_type. Options: 'multistep', 'average', or 'average_direction'."
            )

        if self.has_datetime:
            input_datetime = self.datetime[idx : idx + self.seq_len]
            target_datetime = self.datetime[
                idx + self.seq_len : idx + self.seq_len + self.pred_len
            ]

            # Convert datetime arrays to tensors
            input_datetime_tensor = datetime_to_tensor(input_datetime)
            target_datetime_tensor = datetime_to_tensor(target_datetime)

            return (
                input_tensor,
                target_tensor,
                input_datetime_tensor,
                target_datetime_tensor,
            )
        else:
            return input_tensor, target_tensor

    def to_numpy(self) -> Union[
        Tuple[np.ndarray, np.ndarray],
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    ]:
        """Converts the dataset into a set of NumPy arrays for scikit-learn model training.
        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - inputs: NumPy array of shape (num_samples, seq_len, num_features).
                - targets: NumPy array of shape (num_samples, pred_len, num_target_features).
            If datetime is available:
                - input_datetimes: NumPy array of shape (num_samples, seq_len).
                - target_datetimes: NumPy array of shape (num_samples, pred_len).
        """
        if self.target_type == "multistep":
            num_target_features = self.data.shape[1]
            target_dtype = self.np_dtype
        elif self.target_type == "average":
            num_target_features = 1
            target_dtype = self.np_dtype
        elif self.target_type == "average_direction":
            num_target_features = 1
            target_dtype = np.float32
        else:
            raise ValueError(
                "Invalid target_type. Options: 'multistep', 'average', or 'average_direction'."
            )
        inputs = np.empty(
            (len(self), self.seq_len, self.data.shape[1]), dtype=self.np_dtype
        )
        targets = np.empty(
            (len(self), self.pred_len, num_target_features), dtype=target_dtype
        )

        if self.has_datetime:
            # Specify the unit here to match what tensor_to_datetime uses
            unit = "s"  # seconds
            # Create the arrays with explicit time unit
            input_datetimes = np.empty(
                (len(self), self.seq_len), dtype=f"datetime64[{unit}]"
            )
            target_datetimes = np.empty(
                (len(self), self.pred_len), dtype=f"datetime64[{unit}]"
            )

        for i in range(len(self)):
            item = self.__getitem__(i)
            input_tensor, target_tensor = item[0], item[1]
            inputs[i] = input_tensor.numpy().T
            targets[i] = target_tensor.numpy().T

            if self.has_datetime:
                input_datetime_tensor, target_datetime_tensor = item[2], item[3]
                input_datetimes[i] = tensor_to_datetime(
                    input_datetime_tensor, batch_mode=False
                )
                target_datetimes[i] = tensor_to_datetime(
                    target_datetime_tensor, batch_mode=False
                )

        if self.has_datetime:
            return inputs, targets, input_datetimes, target_datetimes
        else:
            return inputs, targets

    def get_item(self, idx: int) -> Union[
        Tuple[torch.Tensor, torch.Tensor],
        Tuple[torch.Tensor, torch.Tensor, np.ndarray, np.ndarray],
    ]:
        """Get a sample from the dataset.
        This method retrieves an input-target pair at the specified index, with input being
        the lookback window and target being the forecast window based on the target_type.
        Args:
            idx: Index of the starting point of the lookback window.
        Returns:
            If datetime is available:
                tuple: A tuple containing (input_tensor, target_tensor, input_datetime, target_datetime)
                    - input_tensor: Lookback window of shape (num_features, seq_len).
                    - target_tensor: Target window with shape depending on target_type:
                      - "multistep": (num_target_features, pred_len)
                      - "average": (num_target_features, 1)
                      - "average_direction": (num_target_features, 1)
                    - input_datetime: Datetime values for input window of shape (seq_len,).
                    - target_datetime: Datetime values for target window of shape (pred_len,).
            Otherwise:
                tuple: A tuple containing (input_tensor, target_tensor)
                    - input_tensor: Lookback window of shape (num_features, seq_len).
                    - target_tensor: Target window with shape as described above.
        """
        return self.__getitem__(idx)


def normalize_concat_dataset(
    concat_dataset: ConcatDataset,
    scaler: StandardScaler,
) -> None:
    """
    Modifies the data in a ConcatDataset in-place by normalizing it using a fitted StandardScaler.

    Args:
        concat_dataset: ConcatDataset object containing ForecastingDatasets
        scaler: Fitted StandardScaler from scikit-learn.

    Returns:
        None
    """
    for dataset in concat_dataset.datasets:
        data = dataset.data.numpy()

        # Normalize data
        normalized_data = scaler.transform(data)

        # Replace data with normalized version
        dataset.data = torch.tensor(normalized_data, dtype=dataset.torch_dtype)


def normalize_datasets(
    train_dataset: ConcatDataset,
    val_dataset: ConcatDataset,
    test_dataset: ConcatDataset,
) -> Tuple[ConcatDataset, ConcatDataset, ConcatDataset, StandardScaler]:
    """
    Normalizes financial time series datasets using StandardScaler.
    Fits scaler only on training data to prevent look-ahead bias.

    Args:
        train_dataset: Training dataset (ConcatDataset of ForecastingDatasets)
        val_dataset: Validation dataset
        test_dataset: Test dataset

    Returns:
        Tuple[ConcatDataset, ConcatDataset, ConcatDataset, StandardScaler]: Normalized training, validation, and test datasets, and the fitted Standard
    """
    # Extract all underlying data from train_dataset
    all_train_data = []

    # Iterate through the individual ForecastingDataset objects within the ConcatDataset
    for dataset_idx in range(len(train_dataset.datasets)):
        individual_dataset = train_dataset.datasets[dataset_idx]

        # Extract data tensor from the individual dataset
        data = individual_dataset.data.numpy()
        all_train_data.append(data)

    # Stack all the arrays together
    combined_train_data = np.vstack(all_train_data)

    # Fit scaler on training data only
    scaler = StandardScaler()
    scaler.fit(combined_train_data)

    # Apply normalization to all datasets
    normalize_concat_dataset(train_dataset, scaler)
    normalize_concat_dataset(val_dataset, scaler)
    normalize_concat_dataset(test_dataset, scaler)

    return train_dataset, val_dataset, test_dataset, scaler


def get_forecasting_dataset(
    contract_dataset: ContractDataset,
    tte_tolerance: Tuple[int, int],
    seq_len: Optional[int] = None,
    pred_len: Optional[int] = None,
    core_feats: List[str] = ["option_returns"],
    tte_feats: Optional[List[str]] = None,
    datetime_feats: Optional[List[str]] = None,
    vol_feats: Optional[List[str]] = None,
    rolling_volatility_range: Optional[List[int]] = None,
    keep_datetime: bool = False,
    target_type: str = "multistep",
    clean_up: bool = False,
    offline: bool = False,
    intraday: bool = False,
    target_channels: Optional[List[str]] = None,
    dtype: str = "float32",
    save_dir: Optional[str] = None,
    download_only: bool = False,
    validate_contracts: bool = False,
    modify_contracts: bool = False,
    verbose: bool = False,
    warning: bool = True,
    dev_mode: bool = False,
) -> Union[ContractDataset, Tuple[ConcatDataset, ContractDataset]]:
    """
    Creates a PyTorch dataset object composed of multiple ForecastingDatasets, each representing
    different option contracts.

    Args:
        contract_dataset: ContractDataset object containing option contract parameters
        tte_tolerance: Tuple of (min, max) time to expiration tolerance in days
        core_feats: List of core features to include
        tte_feats: List of time-to-expiration features to include
        datetime_feats: List of datetime features to include
        vol_feats: List of volatility features to include
        rolling_volatility_range: List of rolling volatility ranges to include
        keep_datetime: Whether to keep the datetime column in the dataset
        target_type: Type of forecasting target. Options: "multistep" (float), "average" (float), or "average_direction" (binary).
        clean_up: Whether to clean up the data after use
        offline: Whether to load saved contracts from disk
        intraday: Whether to use intraday data
        target_channels: List of target channels to include in the target tensor. If None, all channels will be included.
        seq_len: Sequence length of lookback window (input)
        pred_len: Prediction length of forecast window (target)
        dtype: Data type for the PyTorch tensors
        save_dir: Save directory
        download_only: Whether to download data only (used mainly for Universe class)
        validate_contracts: Whether to validate contracts by requesting data from ThetaData API and adjustintg start and end dates if necessary.
        modify_contracts: Whether to delete old contracts .pkl file and save the (new) validate contracts in the same path. Warning: This will overwrite the old contracts.
        verbose: Whether to print verbose output
        warning: Whether to print verbose DataValidationError statements as warnings or errors.
        dev_mode: Whether to run in development mode.

    Returns:
        ContractDataset: The updated ContractDataset object if `download_only`=True or `validate_contracts`=True.
        Tuple[ConcatDataset, ContractDataset]: A tuple containing the concatenated PyTorch dataset and the updated ContractDataset if download_only=False.
    """

    ctx = Console()
    dataset_list = []
    validated_contracts = []  # Track which contracts are actually valid
    updated_contracts = []  # Track contracts that were updated (not in initial list)
    expirations_exist = False
    tried_contracts = set()

    # First, validate that download_only and validate_contracts aren't both True
    if download_only and validate_contracts:
        raise ValueError(
            "Please use download_only and validate_contracts separately. Both cannot be True."
        )

    if download_only:
        clean_up = False
        offline = False
    elif validate_contracts:
        clean_up = True
        offline = False
    else:
        assert seq_len is not None, "seq_len must be provided for forecasting dataset"
        assert pred_len is not None, "pred_len must be provided for forecasting dataset"

    # Save initial contracts to compare adjusted contracts later
    initial_contracts = deepcopy(contract_dataset.contracts)

    # Iterate through each contract in the ContractDataset
    for (
        contract
    ) in (
        initial_contracts
    ):  # Use initial_contracts instead to avoid modifying during iteration
        # Flag to track if we should try the next contract
        move_to_next_contract = False
        original_contract = deepcopy(contract)  # Keep original for comparison

        # Continue trying the current contract until we succeed or decide to move on
        while not move_to_next_contract:
            try:
                df = contract.load_data(
                    save_dir=save_dir,
                    clean_up=clean_up,
                    offline=offline,
                    warning=warning,
                    dev_mode=dev_mode,
                )

                if not (download_only or validate_contracts):
                    # Select and add features
                    data = transform_features(
                        df=df,
                        core_feats=core_feats,
                        tte_feats=tte_feats,
                        datetime_feats=datetime_feats,
                        vol_feats=vol_feats,
                        rolling_volatility_range=rolling_volatility_range,
                        root=contract.root,
                        right=contract.right,
                        strike=contract.strike,
                        exp=contract.exp,
                        keep_datetime=keep_datetime,
                    )

                    # Convert to PyTorch dataset
                    dataset = ForecastingDataset(
                        data=data,
                        seq_len=seq_len,
                        pred_len=pred_len,
                        target_channels=target_channels,
                        target_type=target_type,
                        dtype=dtype,
                    )
                    dataset_list.append(dataset)

                # If contract was modified from its original state, add to updated contracts
                if contract != original_contract:
                    updated_contracts.append(contract)

                # Add valid contract to validated list
                validated_contracts.append(contract)
                move_to_next_contract = True

            except DataValidationError as e:
                # Handle date-related errors that might be recoverable
                if e.error_code in [INCOMPATIBLE_START_DATE, INCOMPATIBLE_END_DATE]:
                    candidate_start_date = e.real_start_date
                    candidate_exp = (
                        e.real_end_date
                        if e.error_code == INCOMPATIBLE_END_DATE
                        else contract.exp
                    )

                    move_to_next_contract, new_contract = calibrate_new_contract(
                        contract_dataset=contract_dataset,
                        original_contract=original_contract,
                        candidate_start_date=candidate_start_date,
                        candidate_exp=candidate_exp,
                        tte_tolerance=tte_tolerance,
                        expirations_exist=expirations_exist,
                        save_dir=save_dir,
                        verbose=verbose,
                        dev_mode=dev_mode,
                    )
                    expirations_exist = True


                    # If calibration suggests retrying with a new contract...
                    if not move_to_next_contract and new_contract is not None:
                        contract_key = (new_contract.start_date, new_contract.exp)

                        if contract_key in tried_contracts:
                            if verbose:
                                ctx.log(f"Previously tried contract {contract_key} failed. Skipping.")
                            move_to_next_contract = True  # Skip this contract
                            continue

                        tried_contracts.add(contract_key)
                        contract = new_contract  # Try the calibrated contract

                elif e.error_code == NAN_FEATURES:
                    raise e  # Raise the error to be handled outside
                else:
                    if verbose:
                        ctx.log(
                            f"DataValidationError for {contract}: {e}. Moving to next contract."
                        )
                    move_to_next_contract = True

            except Exception as e:
                if verbose:
                    ctx.log(
                        f"Unknown error for {contract}: {e}. Moving to next contract."
                    )
                raise e
                # move_to_next_contract = True

    # Find removed contracts (in initial but not in validated)
    removed_contracts = [c for c in initial_contracts if c not in validated_contracts]

    # Find truly new contracts (in updated but not equivalent to any in initial)
    # This is different from just comparing sets because contracts might be updated versions
    new_contracts = [c for c in updated_contracts]

    # Display changes if verbose
    if verbose and (removed_contracts or new_contracts):
        from rich.table import Table

        # Create a table for removed contracts
        if removed_contracts:
            removed_table = Table(title="Invalid Contracts", show_header=True)
            removed_table.add_column("Root", style="cyan")
            removed_table.add_column("Start Date", style="cyan")
            removed_table.add_column("Expiration", style="cyan")
            removed_table.add_column("Strike", style="cyan")
            removed_table.add_column("Right", style="cyan")

            for contract in removed_contracts:
                removed_table.add_row(
                    contract.root,
                    contract.start_date,
                    contract.exp,
                    str(contract.strike),
                    contract.right,
                    style="red",
                )
            ctx.print(removed_table)

        # Create a table for updated/new contracts
        if updated_contracts:
            updated_table = Table(title="Updated (New Contracts", show_header=True)
            updated_table.add_column("Root", style="cyan")
            updated_table.add_column("Start Date", style="cyan")
            updated_table.add_column("Expiration", style="cyan")
            updated_table.add_column("Strike", style="cyan")
            updated_table.add_column("Right", style="cyan")
            updated_table.add_column("Status", style="cyan")

            for contract in updated_contracts:
                # Check if this is a modified version of an original contract
                is_update = False
                for orig in initial_contracts:
                    if (
                        contract.root == orig.root
                        and contract.strike == orig.strike
                        and contract.interval_min == orig.interval_min
                        and contract.right == orig.right
                    ):
                        is_update = True
                        break

                status = "Updated" if is_update else "New"

                updated_table.add_row(
                    contract.root,
                    contract.start_date,
                    contract.exp,
                    str(contract.strike),
                    contract.right,
                    status,
                    style="green",
                )
            ctx.print(updated_table)

    # Replace the contract list with only validated contracts
    contract_dataset.contracts = validated_contracts

    # Save contract changes if needed
    contracts_changed = set(validated_contracts) != set(initial_contracts)
    if contracts_changed and modify_contracts:
        if verbose:
            ctx.log(
                f"Contract changes detected: {len(removed_contracts)} removed, {len(updated_contracts)} updated/added"
            )
            ctx.log("Saving and overwriting ContractDataset .pkl file")
        contract_dataset.save(clean_file=True)

    if download_only or validate_contracts:
        return contract_dataset
    else:
        if len(dataset_list) == 0:
            raise ValueError(
                "No valid contracts found. All contracts were invalid or contained errors."
            )
        return ConcatDataset(dataset_list), contract_dataset


def calibrate_new_contract(
    contract_dataset: ContractDataset,
    original_contract: Contract,
    candidate_start_date: str,
    candidate_exp: str,
    tte_tolerance: Tuple[int, int],
    expirations_exist: bool = False,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    dev_mode: bool = False,
) -> Tuple[bool, Optional[Contract]]:

    # Rich console
    ctx = Console()

    # <------Calibrate New Expiration------>
    if verbose:
        ctx.log(
            f"Calibrating new expiration date '{candidate_exp}' for original contract..."
        )

    # Query list of expirations to find closests expiration to real_end_date
    expirations_offline = (
        True if expirations_exist else False
    )  # Do not query API if expirations.csv exists
    expirations = get_expirations(
        root=original_contract.root,
        save_dir=save_dir,
        clean_up=False,
        offline=expirations_offline,
        dev_mode=dev_mode,
    )
    expirations["date"] = pd.to_datetime(
        expirations["date"].astype(str), format="%Y%m%d"
    )  # Ensure dates are in datetime format
    candidate_exp = pd.to_datetime(
        candidate_exp, format="%Y%m%d"
    )  # Ensure candidate expiration is in datetime format
    closest_date = (expirations["date"] - candidate_exp).abs().idxmin()
    new_exp = expirations.loc[closest_date, "date"].strftime("%Y%m%d")

    if verbose:
        ctx.log(f"Closest expiration date found: {new_exp}")

    # <------Calibrate New Start Date------>
    if verbose:
        ctx.log(
            f"Calibrating new start date '{candidate_start_date}' for original contract..."
        )
    # Validate that the new start date doesn't fall on a weekend or market holiday, otherwise move forward in time to next valid date
    new_start_date = get_valid_start_date(candidate_start_date)

    if verbose:
        ctx.log(f"New start date found: {new_start_date}")

    # <------TTE Tolerance Check------>
    # Check if the timespan from new_start_date to new_exp is within time-to-expiration tolerance
    start_date_dt = pd.to_datetime(new_start_date, format="%Y%m%d")
    exp_date_dt = pd.to_datetime(new_exp, format="%Y%m%d")
    time_span = exp_date_dt - start_date_dt

    if time_span >= pd.Timedelta(days=tte_tolerance[0]):
        # Update the contract if it meets the tolerance
        if verbose:
            ctx.log(
                f"TTE Tolerance = {time_span} check passed: updating contract with new start date: {new_start_date} and expiration: {new_exp}"
            )

        # Create a new contract and modify parameters
        contract = deepcopy(original_contract)
        contract.start_date = new_start_date
        contract.exp = new_exp

        # Update strike based on new parameters
        contract.strike = find_optimal_strike(
            root=contract.root,
            start_date=new_start_date,
            exp=new_exp,
            right=contract_dataset.right,
            interval_min=contract_dataset.interval_min,
            moneyness=contract_dataset.moneyness,
            strike_band=contract_dataset.strike_band,
            volatility_scaled=contract_dataset.volatility_scaled,
            volatility_scalar=contract_dataset.volatility_scalar,
            hist_vol=contract_dataset.hist_vol,
            clean_up=True,
        )

        return False, contract
    else:
        if verbose:
            ctx.log(
                f"Contract timespan ({time_span.days} days) is too short. Moving to next contract."
            )
        return True, None


def get_valid_start_date(candidate_start_date: str) -> str:
    """
    Return the next valid NYSE trading day given a candidate date in YYYYMMDD format.

    This function checks whether the provided date falls on a weekend or a NYSE holiday.
    If so, it advances the date forward to the next valid trading day.

    Args:
        candidate_start_date (str): The date to validate, in 'YYYYMMDD' format.

    Returns:
        str: The next valid NYSE trading day in 'YYYYMMDD' format.

    Raises:
        ValueError: If no valid trading day is found within the search buffer.
    """
    nyse = mcal.get_calendar("NYSE")
    date_obj = datetime.strptime(candidate_start_date, "%Y%m%d")

    # Get 10 future trading days (timezone-aware, but we’ll just use .date())
    trading_days = nyse.valid_days(
        start_date=candidate_start_date,
        end_date=(date_obj + timedelta(days=10)).strftime("%Y-%m-%d")
    )

    for d in trading_days:
        if d.date() >= date_obj.date():
            return d.strftime("%Y%m%d")

    raise ValueError(f"No valid trading day found after {candidate_start_date}")


def get_forecasting_loaders(
    train_contract_dataset: ContractDataset,
    val_contract_dataset: ContractDataset,
    test_contract_dataset: ContractDataset,
    seq_len: int,
    pred_len: int,
    tte_tolerance: Tuple[int, int],
    core_feats: List[str] = ["option_returns"],
    tte_feats: Optional[List[str]] = None,
    datetime_feats: Optional[List[str]] = None,
    vol_feats: Optional[List[str]] = None,
    rolling_volatility_range: Optional[List[int]] = None,
    keep_datetime: bool = False,
    target_channels: Optional[List[str]] = None,
    target_type: str = "multistep",
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 4,
    prefetch_factor: Optional[int] = None,
    pin_memory: bool = torch.cuda.is_available(),
    persistent_workers: bool = True,
    clean_up: bool = False,
    offline: bool = False,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    scaling: bool = False,
    intraday: bool = False,
    dtype: str = "float32",
    modify_contracts: bool = False,
    warning: bool = True,
    dev_mode: bool = False,
) -> Union[
    Tuple[DataLoader, DataLoader, DataLoader, None],
    Tuple[DataLoader, DataLoader, DataLoader, StandardScaler],
]:
    """
    Forms training, validation, and test dataloaders for option contract data.

    Args:
        train_contract_dataset: Contract dataset for training
        val_contract_dataset: Contract dataset for validation
        test_contract_dataset: Contract dataset for testing
        seq_len: Sequence length for input data
        pred_len: Prediction length for forecasting
        tte_tolerance: Tuple of (min, max) time to expiration tolerance in minutes
        core_feats: List of core features to include
        tte_feats: List of time-to-expiration features to include
        datetime_feats: List of datetime features to include
        keep_datetime: Whether to keep the datetime column in the dataset
        target_type: Type of forecasting target. Options: "multistep" (float), "average" (float), or "average_direction" (binary).
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of subprocesses to use for data loading
        prefetch_factor: Number of batches to prefetch
        pin_memory: Whether to pin memory for faster GPU transfer
        clean_up: Whether to clean up the data after use
        offline: Whether to load saved contracts from disk
        save_dir: Directory to save/load processed datasets
        modify_contracts: Whether to modify contracts if they are invalid in get_forecasting_dataset function calls.
        verbose: Whether to print verbose output
        scaling: Whether to normalize the datasets
        intraday: Whether to use intraday data
        target_channels: List of target channels for forecasting
        dtype: Data type for tensors
        warning: Whether to show warnings

    Returns:
        Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders if scaling=False.
        Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]: Train, validation, and test data loaders, and the scaler if scaling=True.
    """

    # Get the combined datasets of contract data for training, validation, and testing
    train_dataset, _ = get_forecasting_dataset(
        contract_dataset=train_contract_dataset,
        tte_tolerance=tte_tolerance,
        seq_len=seq_len,
        pred_len=pred_len,
        core_feats=core_feats,
        tte_feats=tte_feats,
        vol_feats=vol_feats,
        rolling_volatility_range=rolling_volatility_range,
        datetime_feats=datetime_feats,
        keep_datetime=keep_datetime,
        target_type=target_type,
        clean_up=clean_up,
        offline=offline,
        intraday=intraday,
        target_channels=target_channels,
        dtype=dtype,
        modify_contracts=modify_contracts,
        save_dir=save_dir,
        verbose=verbose,
        warning=warning,
        dev_mode=dev_mode,
    )

    val_dataset, _ = get_forecasting_dataset(
        contract_dataset=val_contract_dataset,
        tte_tolerance=tte_tolerance,
        seq_len=seq_len,
        pred_len=pred_len,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        vol_feats=vol_feats,
        rolling_volatility_range=rolling_volatility_range,
        keep_datetime=keep_datetime,
        target_type=target_type,
        clean_up=clean_up,
        offline=offline,
        intraday=intraday,
        target_channels=target_channels,
        dtype=dtype,
        modify_contracts=modify_contracts,
        save_dir=save_dir,
        verbose=verbose,
        warning=warning,
        dev_mode=dev_mode,
    )
    test_dataset, _ = get_forecasting_dataset(
        contract_dataset=test_contract_dataset,
        tte_tolerance=tte_tolerance,
        seq_len=seq_len,
        pred_len=pred_len,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        vol_feats=vol_feats,
        rolling_volatility_range=rolling_volatility_range,
        keep_datetime=keep_datetime,
        target_type=target_type,
        clean_up=clean_up,
        offline=offline,
        intraday=intraday,
        target_channels=target_channels,
        dtype=dtype,
        modify_contracts=modify_contracts,
        save_dir=save_dir,
        verbose=verbose,
        warning=warning,
        dev_mode=dev_mode,
    )

    if scaling:
        train_dataset, val_dataset, test_dataset, scaler = normalize_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset,
        )
    else:
        scaler = None

    # Create dataloaders for training, validation, and testing
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=persistent_workers,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, test_loader, scaler


def create_windows(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    window_stride: int,
    intraday: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates rolling windows of data for a given DataFrame. Should be used primarily for scikit-learn models and/or intraday modeling,
    otherwise default to optrade.data.forecasing.get_forecasting_loaders or optrade.data.forecasting.get_forecasting_datasets.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        seq_len (int): Length of the input sequence.
        pred_len (int): Length of the prediction sequence.
        window_stride (int): Number of steps to move the window forward.
        intraday (bool): Whether the data is intraday or not. If True, the function will first split the data
                         into separate trading days before creating individual windows that cannot crossover
                         between days. Otherwise, the function will create windows that can span multiple days.
    Returns:
        input (np.ndarray): Array of input windows of shape (num_windows, seq_len, num_features) where num_features
                            is the number of columns in the DataFrame (removing datetime but adding returns).
        target (np.ndarray): Array of target windows of shape (num_windows, pred_len, 1).
                             Target contains only returns for the 'option_mid_price'.
    """
    datetime = df["datetime"]
    df_copy = df.copy()

    # Define input features (all columns except datetime)
    feature_columns = [col for col in df_copy.columns if col != "datetime"]
    inputs, targets = [], []

    if intraday:
        # Get all unique days
        days = datetime.dt.date.unique()

        # Iterate through each day independently
        for day in days:
            day_mask = datetime.dt.date == day
            day_data = df_copy.loc[day_mask].copy()

            print(f"Length of data: {len(day_data)}")

            # Since returns will be part of the input features but don't exist for 9:30am
            # we remove the market open (9:30am) of each day
            first_time = day_data["datetime"].iloc[0].time()
            if first_time.hour == 9 and first_time.minute == 30:
                day_data = day_data.iloc[1:].reset_index(drop=True)
                print(f"Day data after removing 9:30am: {day_data.tail()}")

            # Raise an error if the length of the day is less than the sum of seq_len+pred_len
            if len(day_data) < seq_len + pred_len:
                raise ValueError(
                    f"seq_len + pred_len = {seq_len + pred_len} exceeds the length of the day. \
                    Either set intraday=False or reduce seq_len and/or pred_len."
                )

            # Get input features and targets, convert to NumPy arrays
            day_features = day_data[feature_columns].to_numpy()
            day_targets = day_data["option_returns"].to_numpy().reshape(-1, 1)

            # Apply the sliding window technique to obtain windows for inputs and targets
            for i in range(0, len(day_data) - seq_len - pred_len + 1, window_stride):
                inputs.append(day_features[i : i + seq_len])
                targets.append(day_targets[i + seq_len : i + seq_len + pred_len])
    else:
        # Since returns will be part of the input features but don't exist for first market open
        # i.e. 9:30am on the first day, we remove it
        first_time = datetime.iloc[0].time()
        if first_time.hour == 9 and first_time.minute == 30:
            df_copy = df_copy.iloc[1:].reset_index(drop=True)

        # Extract features and targets
        features = df_copy[feature_columns].to_numpy()
        targets_data = df_copy["option_returns"].to_numpy().reshape(-1, 1)

        # Create windows
        for i in range(0, len(df_copy) - seq_len - pred_len + 1, window_stride):
            inputs.append(features[i : i + seq_len])
            targets.append(targets_data[i + seq_len : i + seq_len + pred_len])

    # Convert to numpy arrays
    return np.array(inputs), np.array(targets)


if __name__ == "__main__":
    # Test: get_forecasting_loaders
    root = "AMZN"
    total_start_date = "20230101"
    total_end_date = "20230901"
    right = "C"
    interval_min = 60
    contract_stride = 3
    target_tte = 30
    tte_tolerance = (15, 45)
    moneyness = "ATM"
    volatility_scaled = True
    volatility_scalar = 0.01
    volatility_type = "period"
    strike_band = 0.05

    # TTE features
    tte_feats = ["sqrt", "exp_decay"]

    # Datetime features
    datetime_feats = [
        "sin_minute_of_day",
        "cos_minute_of_day",
        "sin_hour_of_week",
        "cos_hour_of_week",
    ]

    # Select features
    core_feats = [
        "log_option_returns",
        "log_stock_returns",
        "option_returns",
        "stock_returns",
        "distance_to_strike",
        "moneyness",
        "option_lob_imbalance",
        "option_quote_spread",
        "stock_lob_imbalance",
        "stock_quote_spread",
        "option_mid_price",
        "option_bid_size",
        "option_bid",
        "option_ask_size",
        "option_close",
        "option_volume",
        "option_count",
        "stock_mid_price",
        "stock_bid_size",
        "stock_bid",
        "stock_ask_size",
        "stock_ask",
        "stock_volume",
        "stock_count",
    ]

    # Testing: get_loaders
    from optrade.data.contracts import get_contract_datasets

    train_cd, val_cd, test_cd = get_contract_datasets(
        root=root,
        start_date=total_start_date,
        end_date=total_end_date,
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
        verbose=True,
        train_split=0.5,
        val_split=0.25,
        dev_mode=True,
    )

    output = get_forecasting_loaders(
        train_contract_dataset=train_cd,
        val_contract_dataset=val_cd,
        test_contract_dataset=test_cd,
        tte_tolerance=tte_tolerance,
        keep_datetime=True,
        target_type="average",
        seq_len=100,
        pred_len=10,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        batch_size=32,
        clean_up=False,
        offline=True,
        save_dir=None,
        verbose=True,
        scaling=True,
        dev_mode=True,
    )
    train_loader, val_loader, test_loader = output[0:3]

    print(f"Num train examples: {len(train_loader.dataset)}")
    print(f"Num val examples: {len(val_loader.dataset)}")
    print(f"Num test examples: {len(test_loader.dataset)}")

    for batch in train_loader:
        x, y, x_dt, y_dt = batch
        print(f"x ({x.dtype}) shape: {x.shape}. y ({y.dtype}) shape: {y.shape}")

        print(f"x_dt (before): {x_dt.shape}. y_dt (before): {y_dt.shape}")
        x_dt = tensor_to_datetime(timestamp_tensor=x_dt, batch_mode=True)
        y_dt = tensor_to_datetime(timestamp_tensor=y_dt, batch_mode=True)

        print(f"x_dt: {x_dt}. y_dt: {y_dt}")
        break

    # # Testing: create_windows
    # from optrade.data.features import transform_features
    # from optrade.data.contracts import Contract
    # from rich.console import Console

    # console = Console()

    # contract = Contract.find_optimal(
    #     root="AAPL",
    #     start_date="20241107",
    #     volatility_scaled=False,
    #     strike_band=0.05,
    #     moneyness="OTM",
    #     interval_min=1,
    #     right="C",
    #     target_tte=30,
    #     tte_tolerance=(25, 35),
    # )

    # df = contract.load_data(clean_up=True, offline=False, warning=True)

    # # TTE features
    # tte_feats = ["sqrt", "exp_decay"]

    # # Datetime features
    # datetime_feats = [
    #     "sin_minute_of_day",
    #     "cos_minute_of_day",
    #     "sin_hour_of_week",
    #     "cos_hour_of_week",
    # ]

    # # Select features
    # core_feats = [
    #     "option_returns",
    #     "stock_returns",
    #     "distance_to_strike",
    #     "moneyness",
    #     "option_lob_imbalance",
    #     "option_quote_spread",
    #     "stock_lob_imbalance",
    #     "stock_quote_spread",
    #     "option_mid_price",
    #     "option_bid_size",
    #     "option_bid",
    #     "option_ask_size",
    #     "option_close",
    #     "option_volume",
    #     "option_count",
    #     "stock_mid_price",
    #     "stock_bid_size",
    #     "stock_bid",
    #     "stock_ask_size",
    #     "stock_ask",
    #     "stock_volume",
    #     "stock_count",
    # ]

    # df = transform_features(
    #     df=df,
    #     core_feats=core_feats,
    #     tte_feats=tte_feats,
    #     datetime_feats=datetime_feats,
    #     strike=contract.strike,
    #     exp=contract.exp,
    #     keep_datetime=True,
    # )
    # print(df.columns)

    # x, y = create_windows(
    #     df=df, seq_len=30, pred_len=6, window_stride=1, intraday=False
    # )

    # print(x.shape, y.shape)
