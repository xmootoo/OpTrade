from pathlib import Path
from typing import Tuple, List, Union, Optional
from rich.console import Console
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset, ConcatDataset

# Data
from optrade.data.contracts import get_contract_datasets

# Datasets
from optrade.data.contracts import ContractDataset
from optrade.utils.error_handlers import (
    DataValidationError,
    INCOMPATIBLE_START_DATE,
    INCOMPATIBLE_END_DATE,
)

# Features
from optrade.data.features import transform_features

# Get absolute path for this script
SCRIPT_DIR = Path(__file__).resolve().parent


class ForecastingDataset(Dataset):
    """
    Args:
        data (torch.Tensor): Tensor of shape (num_time_steps, num_features)
        seq_len (int): Length of the lookback window
        pred_len (int): Length of the forecast window
        target_channels (list): Channels to forecast. By default target_channels=["option_returns"], which corresponds to the
                                option midprice returns. If None, all channels will be returned for the target.
        dtype (str): Desired data type of the tensor. Default is "float32".
    """

    def __init__(
        self,
        data: pd.DataFrame,
        seq_len: int,
        pred_len: int,
        target_channels: Optional[List[str]] = None,
        dtype: str = "float32",
    ) -> None:
        self.data = torch.tensor(data.to_numpy(), dtype=eval("torch." + dtype))
        self.seq_len = seq_len
        self.pred_len = pred_len

        if target_channels is not None and len(target_channels) > 0:
            feature_names = data.columns.to_list()
            self.target_channels_idx = [
                feature_names.index(channel) for channel in target_channels
            ]

    def __len__(self) -> int:
        return self.data.shape[0] - self.seq_len - self.pred_len

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            idx (int): Index of the starting point of the lookback window

        Returns:
            input (torch.Tensor): Lookback window of shape (num_features, seq_len)
            target (torch.Tensor): Target window of shape (len(target_channels), pred_len) if target_channels is not None, otherwise
                                   (num_features, pred_len).
        """
        input = self.data[idx : idx + self.seq_len]
        if self.target_channels:
            target = self.data[
                idx + self.seq_len : idx + self.seq_len + self.pred_len,
                self.target_channels_idx,
            ]
        else:
            target = self.data[idx + self.seq_len : idx + self.seq_len + self.pred_len]
        return input.transpose(0, 1), target.transpose(0, 1)


def normalize_concat_dataset(
    concat_dataset: ConcatDataset,
    scaler: StandardScaler,
) -> None:
    """
    Modifies the data in a ConcatDataset in-place by normalizing it using a fitted StandardScaler.

    Args:
        concat_dataset: ConcatDataset object containing ForecastingDatasets
        scaler: Fitted StandardScaler (scikit-learn)

    Returns:
        None
    """
    for dataset in concat_dataset.datasets:
        data = dataset.data.numpy()
        dtype = dataset.dtype

        # Normalize data
        normalized_data = scaler.transform(data)

        # Replace data with normalized version
        dataset.data = torch.tensor(normalized_data, dtype=dtype)


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
        save_scaler: Whether to save the scaler to disk
        scaler_path: Path to save the scaler if save_scaler is True

    Returns:
        Normalized train_dataset, val_dataset, test_dataset, and the fitted scaler
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
    contracts: ContractDataset,
    seq_len: int,
    pred_len: int,
    tte_tolerance: Tuple[int, int],
    core_feats: List[str] = ["option_returns"],
    tte_feats: Optional[List[str]] = None,
    datetime_feats: Optional[List[str]] = None,
    clean_up: bool = True,
    offline: bool = False,
    intraday: bool = False,
    target_channels: Optional[List[str]] = None,
    dtype: str = "float32",
    save_dir: Optional[str] = None,
    download_only: bool = False,
    verbose: bool = False,
) -> Dataset:
    """
    Creates a PyTorch dataset object composed of multiple ForecastingDatasets, each representing
    different option contracts.

    Args:
        contracts: ContractDataset object containing option contract parameters
        core_feats: List of core features to include
        tte_feats: List of time-to-expiration features to include
        datetime_feats: List of datetime features to include
        tte_tolerance: Tuple of (min, max) time to expiration tolerance in days
        clean_up: Whether to clean up the data after use
        offline: Whether to load saved contracts from disk
        intraday: Whether to use intraday data
        target_channels: List of target channels to include in the target tensor. If None, all channels will be included.
        seq_len: Sequence length of lookback window (input)
        pred_len: Prediction length of forecast window (target)
        dtype: Data type for the PyTorch tensors
        save_dir: Save directory
        download_only: Whether to download data only (used mainly for Universe class)
        verbose: Whether to print verbose output

    Returns:
        Concatenated PyTorch dataset object
    """

    ctx = Console()
    dataset_list = []

    if download_only:
        clean_up = offline = False

    for contract in contracts.contracts:
        # Flag to track if we should try the next contract
        move_to_next_contract = False

        # Continue trying the current contract until we succeed or decide to move on
        while not move_to_next_contract:
            try:
                df = contract.load_data(
                    save_dir=save_dir, clean_up=clean_up, offline=offline
                )

                if not download_only:
                    # Select and add features
                    data = transform_features(
                        df=df,
                        core_feats=core_feats,
                        tte_feats=tte_feats,
                        datetime_feats=datetime_feats,
                        strike=contract.strike,
                        exp=contract.exp,
                    )

                    # Convert to PyTorch dataset
                    dataset = ForecastingDataset(
                        data=data,
                        seq_len=seq_len,
                        pred_len=pred_len,
                        target_channels=target_channels,
                        dtype=dtype,
                    )
                    dataset_list.append(dataset)
                else:
                    pass

                # If we get here successfully, move to the next contract
                move_to_next_contract = True

            except DataValidationError as e:
                if e.error_code == INCOMPATIBLE_START_DATE:
                    new_start_date = e.real_start_date

                    # Check if (exp - new_start_date) is within tte_tolerance. If not, move to the next contract
                    if pd.to_datetime(contract.exp, format="%Y%m%d") - pd.to_datetime(
                        new_start_date, format="%Y%m%d"
                    ) < pd.Timedelta(days=tte_tolerance[0]):
                        if verbose:
                            ctx.log(
                                f"Option contract start date mismatch. New start date {new_start_date} is too close to expiration {contract.exp}. Moving to next contract."
                            )
                        move_to_next_contract = True
                    # If tte_tolerance is satisfied, update the contract with the new start date and try again
                    else:
                        if verbose:
                            ctx.log(
                                f"Option contract start date mismatch. Attempting to get data for {contract} with new start date: {new_start_date}"
                            )
                        contract.start_date = new_start_date
                elif e.error_code == INCOMPATIBLE_END_DATE:
                    new_start_date = e.real_start_date
                    new_exp = e.real_end_date

                    # Check if (new_exp - new_start_date) is within tte_tolerance. If not, move to the next contract
                    if pd.to_datetime(new_exp, format="%Y%m%d") - pd.to_datetime(
                        new_start_date, format="%Y%m%d"
                    ) < pd.Timedelta(days=tte_tolerance[0]):
                        if verbose:
                            ctx.log(
                                f"Option contract start date and end date mismatch. New start date {new_start_date} is too close to new expiration {new_exp}. Moving to next contract."
                            )
                        move_to_next_contract = True
                    # If tte_tolerance is satisfied, update contract expiration to the observed end date of option data
                    else:
                        # For other DataValidationError types, move to the next contract
                        if verbose:
                            ctx.log(
                                f"DataValidationError for {contract}: {e}. Moving to next contract."
                            )
                        move_to_next_contract = True
                        contract.exp = new_exp
                else:
                    # For other DataValidationError types, move to the next contract
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
                move_to_next_contract = True

    if download_only:
        return
    else:
        return ConcatDataset(dataset_list)


def get_forecasting_loaders(
    root: str = "AAPL",
    start_date: str = "20231107",
    end_date: str = "20241114",
    contract_stride: int = 5,
    interval_min: int = 1,
    right: str = "C",
    target_tte: int = 30,
    tte_tolerance: Tuple[int, int] = (25, 35),
    moneyness: str = "OTM",
    target_band: float = 0.05,
    volatility_type: str = "period",
    volatility_scaled: bool = True,
    volatility_scalar: float = 1.0,
    train_split: float = 0.7,
    val_split: float = 0.1,
    core_feats: List[str] = ["option_returns"],
    tte_feats: List[str] = ["sqrt"],
    datetime_feats: List[str] = ["sin_timeofday"],
    batch_size: int = 32,
    shuffle: bool = True,
    drop_last: bool = False,
    num_workers: int = 4,
    prefetch_factor: int = 2,
    pin_memory: bool = torch.cuda.is_available(),
    clean_up: bool = True,
    offline: bool = False,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    scaling: bool = True,
    intraday: bool = False,
    target_channels: List[int] = [0],
    seq_len: int = 100,
    pred_len: int = 10,
    dtype: str = "float64",
) -> Union[
    Tuple[DataLoader, DataLoader, DataLoader],
    Tuple[DataLoader, DataLoader, DataLoader, StandardScaler],
]:
    """

    Forms training, validation, and test dataloaders for option contract data.

    Args:
        root: Underlying stock symbol
        start_date: Start date for the total dataset in YYYYMMDD format
        end_date: End date for the total dataset in YYYYMMDD format
        contract_stride: Number of days between each contract (for sampling)
        interval_min: Interval in minutes for the underlying stock data and option data
        right: Option type (C for call, P for put)
        target_tte: Target time to expiration in days
        tte_tolerance: Tuple of (min, max) time to expiration tolerance in minutes
        moneyness: Moneyness of the option contract (OTM, ATM, ITM)
        target_band: Target band for moneyness selection
        volatility_type: Type of historical volatility to use
        volatility_scaled: Whether to scale strikes based on historical volatility
        volatility_scalar: Scalar to adjust historical volatility
        train_split: Proportion of total days to use for training
        val_split: Proportion of total days to use for validation
        core_feats: List of core features to include
        tte_feats: List of time-to-expiration features to include
        datetime_feats: List of datetime features to include
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        drop_last: Whether to drop the last incomplete batch
        num_workers: Number of subprocesses to use for data loading
        prefetch_factor: Number of batches to prefetch
        pin_memory: Whether to pin memory for faster GPU transfer
        clean_up: Whether to clean up the data after use
        offline: Whether to load saved contracts from disk
        contract_dir: Directory to save/load contracts
        verbose: Whether to print verbose output
        scaling: Whether to normalize the datasets

    Returns:
        Training, validation, and test PyTorch dataloaders.
    """

    train_contracts, val_contracts, test_contracts = get_contract_datasets(
        root=root,
        start_date=start_date,
        end_date=end_date,
        contract_stride=contract_stride,
        interval_min=interval_min,
        right=right,
        target_tte=target_tte,
        tte_tolerance=tte_tolerance,
        moneyness=moneyness,
        target_band=target_band,
        volatility_type=volatility_type,
        volatility_scaled=volatility_scaled,
        volatility_scalar=volatility_scalar,
        train_split=train_split,
        val_split=val_split,
        clean_up=clean_up,
        offline=offline,
        save_dir=save_dir,
        verbose=verbose,
    )

    # Get the combined datasets of contract data for training, validation, and testing
    train_dataset = get_forecasting_dataset(
        contracts=train_contracts,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        tte_tolerance=tte_tolerance,
        clean_up=clean_up,
        offline=offline,
        intraday=intraday,
        target_channels=target_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        dtype=dtype,
        verbose=verbose,
    )

    val_dataset = get_forecasting_dataset(
        contracts=val_contracts,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        tte_tolerance=tte_tolerance,
        clean_up=clean_up,
        offline=offline,
        intraday=intraday,
        target_channels=target_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        dtype=dtype,
        verbose=verbose,
    )
    test_dataset = get_forecasting_dataset(
        contracts=test_contracts,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        tte_tolerance=tte_tolerance,
        clean_up=clean_up,
        offline=offline,
        intraday=intraday,
        target_channels=target_channels,
        seq_len=seq_len,
        pred_len=pred_len,
        dtype=dtype,
        verbose=verbose,
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
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory,
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True,
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
    total_end_date = "20230601"
    right = "C"
    interval_min = 60
    contract_stride = 5
    target_tte = 30
    tte_tolerance = (15, 45)
    moneyness = "ATM"
    volatility_scaled = True
    volatility_scalar = 0.01
    volatility_type = "period"
    target_band = 0.05

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
    output = get_forecasting_loaders(
        root=root,
        start_date=total_start_date,
        end_date=total_end_date,
        contract_stride=contract_stride,
        interval_min=interval_min,
        right=right,
        target_tte=target_tte,
        tte_tolerance=tte_tolerance,
        moneyness=moneyness,
        target_band=target_band,
        volatility_type=volatility_type,
        volatility_scaled=volatility_scaled,
        volatility_scalar=volatility_scalar,
        train_split=0.4,
        val_split=0.3,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        batch_size=32,
        clean_up=False,
        offline=False,
        save_dir=None,
        verbose=True,
        scaling=True,
    )
    train_loader, val_loader, test_loader = output[0:3]

    print(f"Num train examples: {len(train_loader.dataset)}")
    print(f"Num val examples: {len(val_loader.dataset)}")
    print(f"Num test examples: {len(test_loader.dataset)}")

    # Testing: create_windows
    from optrade.data.features import transform_features
    from optrade.data.contracts import Contract
    from rich.console import Console

    console = Console()

    contract = Contract.find_optimal(
        root="AAPL",
        start_date="20241107",
        volatility_scaled=False,
        target_band=0.05,
        moneyness="OTM",
        interval_min=1,
        right="C",
    )

    df = contract.load_data(clean_up=True, offline=False)

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

    df = transform_features(
        df=df,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        strike=contract.strike,
        exp=contract.exp,
        keep_datetime=True,
    )
    print(df.columns)

    x, y = create_windows(
        df=df, seq_len=30, pred_len=6, window_stride=1, intraday=False
    )

    print(x.shape, y.shape)
