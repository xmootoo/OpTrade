import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, ConcatDataset
import pandas as pd
from typing import Tuple, Optional, List
from pathlib import Path
from rich.console import Console
from sklearn.preprocessing import StandardScaler

# Data
from optrade.data.thetadata.stocks import get_stock_data
from optrade.data.thetadata.get_data import get_data

# Datasets
from optrade.src.preprocessing.data.datasets import ContractDataset, ForecastingDataset

# Features
from optrade.src.preprocessing.features.get_features import get_features

# Historical Volatility
from optrade.src.preprocessing.data.volatility import get_historical_volatility

# Utils
from optrade.src.utils.data.error import DataValidationError, INCOMPATIBLE_START_DATE, INCOMPATIBLE_END_DATE, MISSING_DATES, UNKNOWN_ERROR

# Get absolute path for this script
SCRIPT_DIR = Path(__file__).resolve().parent



# <------------- Debugging Above ------------->

def get_contract_datasets(
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
    clean_up: bool = True,
    offline: bool = False,
    save_dir: Optional[Path] = None,
    verbose: bool=False,
) -> Tuple[ContractDataset, ContractDataset, ContractDataset]:
    """
    Returns the training, validation, and test datasets contract datasets. These contain mutually exclusive contracts
    at mutually exclusive time periods to prevent information leakage during training and evaluation.

    Args:
        root: Underlying stock symbol
        start_date: Start date for the total dataset in YYYYMMDD format
        end_date: End date for the total dataset in YYYYMMDD format
        contract_stride: Number of days between each contract
        interval_min: Interval in minutes for the underlying stock data
        right: Option type (C for call, P for put)
        target_tte: Target time to expiration in days
        tte_tolerance: Tuple of (min, max) time to expiration tolerance in days
        moneyness: Moneyness of the option contract (OTM, ATM, ITM)
        target_band: Target band for moneyness selection, proportion of current underlying price
        volatility_type: Type of historical volatility to use
        volatility_scaled: Whether to scale strikes based on historical volatility
        volatility_scalar: Scalar to adjust historical volatility
        train_split: Proportion of total days to use for training
        val_split: Proportion of total days to use for validation
        clean_up: Whether to clean up the data after use
        offline: Whether to load saved contracts from disk
        save_dir: Directory to save/load contracts
        verbose: Whether to print verbose output

    Returns:
        Training, validation, and test contract datasets.
    """

    # Set directories for saving or loading
    if save_dir is None:
        save_dir = SCRIPT_DIR.parents[2] / "data" / "historical_data" / "contracts"

    # Create a structured path based on key parameters
    contract_dir = (
        save_dir /
        root /
        f"{start_date}_{end_date}" /
        right /
        f"contract_stride_{contract_stride}" /
        f"interval_{interval_min}" /
        f"target_tte_{target_tte}" /
        f"moneyness_{moneyness}"
    )

    # Add volatility info to path if volatility_scaled is True
    if volatility_scaled:
        contract_dir = contract_dir / f"voltype_{volatility_type}_volscalar_{volatility_scalar}"
    else:
        contract_dir = contract_dir / f"target_band_{str(target_band).replace('.', 'p')}"

    # Offline loading (if already saved)
    if offline:
        if not all((
                (contract_dir / "train_contracts.pkl").exists(),
                (contract_dir / "val_contracts.pkl").exists(),
                (contract_dir / "test_contracts.pkl").exists()
            )):
                raise FileNotFoundError(f"Missing contract files in {contract_dir}")

        train_contracts = ContractDataset.load(contract_dir / "train_contracts.pkl")
        val_contracts = ContractDataset.load(contract_dir / "val_contracts.pkl")
        test_contracts = ContractDataset.load(contract_dir / "test_contracts.pkl")
        return train_contracts, val_contracts, test_contracts

    # Volatility-based selection of strikes (Optional)
    if volatility_scaled:
        hist_vol = get_hist_vol(
            root=root,
            start_date=start_date,
            end_date=end_date,
            interval_min=interval_min,
            volatility_window=train_split, # Use the ONLY training data to compute historical volatility (prevent data leakage)
            volatility_type=volatility_type
        )
    else:
        hist_vol = None

    # Get contiguous training, validation, and test (start_date, end_date) pairs in YYYYMMDD format
    total_days = (pd.to_datetime(end_date, format='%Y%m%d') - pd.to_datetime(start_date, format='%Y%m%d')).days
    num_train_days = int(train_split * total_days)
    num_val_days = int(val_split * total_days)

    train_end_date = (pd.to_datetime(start_date, format='%Y%m%d') + pd.Timedelta(days=num_train_days)).strftime('%Y%m%d')
    val_end_date = (pd.to_datetime(train_end_date, format='%Y%m%d') + pd.Timedelta(days=num_val_days)).strftime('%Y%m%d')
    test_start_date = (pd.to_datetime(val_end_date, format='%Y%m%d') + pd.Timedelta(days=1)).strftime('%Y%m%d')

    train_dates = (start_date, train_end_date)
    val_dates = (train_end_date, val_end_date)
    test_dates = (test_start_date, end_date)

    # Create the training, validation, and test contract datasets
    from rich.console import Console
    ctx = Console()
    ctx.log("------------CREATING TRAINING CONTRACTS------------") if verbose else None
    train_contracts = ContractDataset(
        root=root,
        total_start_date=train_dates[0],
        total_end_date=train_dates[1],
        contract_stride=contract_stride,
        interval_min=interval_min,
        right=right,
        target_tte=target_tte,
        tte_tolerance=tte_tolerance,
        moneyness=moneyness,
        target_band=target_band,
        hist_vol=hist_vol,
        volatility_scaled=volatility_scaled,
        volatility_scalar=volatility_scalar,
        verbose=verbose,
    ).generate_contracts()

    ctx.log("------------CREATING VALIDATION CONTRACTS------------") if verbose else None
    val_contracts = ContractDataset(
        root=root,
        total_start_date=val_dates[0],
        total_end_date=val_dates[1],
        contract_stride=contract_stride,
        interval_min=interval_min,
        right=right,
        target_tte=target_tte,
        tte_tolerance=tte_tolerance,
        moneyness=moneyness,
        target_band=target_band,
        hist_vol=hist_vol,
        volatility_scaled=volatility_scaled,
        volatility_scalar=volatility_scalar,
        verbose=verbose,
    ).generate_contracts()

    ctx.log("------------CREATING TEST CONTRACTS------------") if verbose else None
    test_contracts = ContractDataset(
        root=root,
        total_start_date=test_dates[0],
        total_end_date=test_dates[1],
        contract_stride=contract_stride,
        interval_min=interval_min,
        right=right,
        target_tte=target_tte,
        tte_tolerance=tte_tolerance,
        moneyness=moneyness,
        target_band=target_band,
        hist_vol=hist_vol,
        volatility_scaled=volatility_scaled,
        volatility_scalar=volatility_scalar,
        verbose=verbose,
    ).generate_contracts()

    if not clean_up:

        # Create all directories
        contract_dir.mkdir(parents=True, exist_ok=True)

        # Save with a simple name since the path contains all the parameter info
        train_contracts.save(contract_dir / "train_contracts.pkl")
        val_contracts.save(contract_dir / "val_contracts.pkl")
        test_contracts.save(contract_dir / "test_contracts.pkl")

    return train_contracts, val_contracts, test_contracts

def get_hist_vol(
    root: str,
    start_date: str,
    end_date: str,
    interval_min: int,
    volatility_window: float,
    volatility_type: str) -> pd.DataFrame:

    """
    Get historical volatility for a stock over a given time period.

    Args:
        root: Underlying stock symbol
        start_date: Start date for the total dataset in YYYYMMDD format
        end_date: End date for the total dataset in YYYYMMDD format
        interval_min: Interval in minutes for the underlying stock data
        volatility_window: Proportion of total days to use for historical volatility calculation
        volatility_type: Type of historical volatility to use. Options: "daily", "period", "annualized".

    Returns:
        Historical volatility value based on the specified type.
    """

    # Calculate number of days to use for historical volatility
    total_days = (pd.to_datetime(end_date, format='%Y%m%d') - pd.to_datetime(start_date, format='%Y%m%d')).days
    num_vol_days = int(volatility_window * total_days)
    vol_end_date = (pd.to_datetime(start_date, format='%Y%m%d') + pd.Timedelta(days=num_vol_days)).strftime('%Y%m%d')

    stock_data = get_stock_data(
        root=root,
        start_date=start_date,
        end_date=end_date,
        interval_min=interval_min,
        clean_up=True,
    )

    # Select only the first num_vol_days for calculating volatility
    stock_data = stock_data.loc[stock_data['datetime'] <= vol_end_date]

    # Calculate historical volatility
    return get_historical_volatility(stock_data, volatility_type)

def get_combined_dataset(
    contracts: ContractDataset,
    core_feats: list,
    tte_feats: list,
    datetime_feats: list,
    tte_tolerance: Tuple[int, int],
    clean_up: bool = True,
    offline: bool = False,
    intraday: bool = False,
    target_channels: list=[0],
    seq_len: int=100,
    pred_len: int=10,
    dtype: str="float64",
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
        target_channels: List of target channels to include
        seq_len: Sequence length
        pred_len: Prediction length
        dtype: Data type for the PyTorch tensors

    Returns:
        Concatenated PyTorch dataset object
    """

    ctx = Console()
    dataset_list = []

    for contract in contracts.contracts:
        # Flag to track if we should try the next contract
        move_to_next_contract = False

        # Continue trying the current contract until we succeed or decide to move on
        while not move_to_next_contract:
            try:
                df = get_data(
                    contract=contract,
                    clean_up=clean_up,
                    offline=offline,
                )

                # Select and add features
                data = get_features(
                    df=df,
                    core_feats=core_feats,
                    tte_feats=tte_feats,
                    datetime_feats=datetime_feats,
                    strike=contract.strike,
                ).to_numpy()

                # Convert to PyTorch dataset
                dataset = ForecastingDataset(data=data, seq_len=seq_len, pred_len=pred_len, target_channels=target_channels, dtype=dtype)
                dataset_list.append(dataset)

                # If we get here successfully, move to the next contract
                move_to_next_contract = True

            except DataValidationError as e:
                if e.error_code == INCOMPATIBLE_START_DATE:
                    new_start_date = e.real_start_date

                    # Check if (exp - new_start_date) is within tte_tolerance. If not, move to the next contract
                    if pd.to_datetime(contract.exp, format='%Y%m%d') - pd.to_datetime(new_start_date, format='%Y%m%d') < pd.Timedelta(days=tte_tolerance[0]):
                        ctx.log(f"Option contract start date mismatch. New start date {new_start_date} is too close to expiration {contract.exp}. Moving to next contract.")
                        move_to_next_contract = True
                    # If tte_tolerance is satisfied, update the contract with the new start date and try again
                    else:
                        ctx.log(f"Option contract start date mismatch. Attempting to get data for {contract} with new start date: {new_start_date}")
                        contract.start_date = new_start_date
                elif e.error_code == INCOMPATIBLE_END_DATE:
                    new_start_date = e.real_start_date
                    new_exp = e.real_end_date

                    # Check if (new_exp - new_start_date) is within tte_tolerance. If not, move to the next contract
                    if pd.to_datetime(new_exp, format='%Y%m%d') - pd.to_datetime(new_start_date, format='%Y%m%d') < pd.Timedelta(days=tte_tolerance[0]):
                        ctx.log(f"Option contract start date and end date mismatch. New start date {new_start_date} is too close to new expiration {new_exp}. Moving to next contract.")
                        move_to_next_contract = True
                    # If tte_tolerance is satisfied, update contract expiration to the observed end date of option data
                    else:
                        # For other DataValidationError types, move to the next contract
                        ctx.log(f"DataValidationError for {contract}: {e}. Moving to next contract.")
                        move_to_next_contract = True
                        contract.exp = new_exp
                else:
                    # For other DataValidationError types, move to the next contract
                    ctx.log(f"DataValidationError for {contract}: {e}. Moving to next contract.")
                    move_to_next_contract = True
            except Exception as e:
                ctx.log(f"Unknown error for {contract}: {e}. Moving to next contract.")
                move_to_next_contract = True

    return ConcatDataset(dataset_list)

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

def get_loaders(
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
    contract_dir: Optional[Path] = None,
    verbose: bool=False,
    scaling: bool=True,
    intraday: bool=False,
    target_channels: List[int]=[0],
    seq_len: int=100,
    pred_len: int=10,
    dtype: str="float64",
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]:
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
        save_dir=contract_dir,
        verbose=verbose,
    )

    # Get the combined datasets of contract data for training, validation, and testing
    train_dataset = get_combined_dataset(
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
    )

    val_dataset = get_combined_dataset(
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
    )
    test_dataset = get_combined_dataset(
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
    )

    if scaling:
        train_dataset, val_dataset, test_dataset, scaler = normalize_datasets(
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            test_dataset=test_dataset
        )

    # Create dataloaders for training, validation, and testing
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory
    )
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory
    )
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        pin_memory=pin_memory
    )

    if scaling:
        return train_loader, val_loader, test_loader, scaler
    else:
        return train_loader, val_loader, test_loader

if __name__ == "__main__":
    root="AMZN"
    total_start_date="20230101"
    total_end_date="20230601"
    right="C"
    interval_min=60
    contract_stride=5
    target_tte=30
    tte_tolerance=(15, 45)
    moneyness="ATM"
    volatility_scaled=True
    volatility_scalar=0.01
    volatility_type="period"
    target_band=0.05

    # TTE features
    tte_feats = ["sqrt", "exp_decay"]

    # Datetime features
    datetime_feats = ["sin_minute_of_day", "cos_minute_of_day", "sin_hour_of_week", "cos_hour_of_week"]

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
    output = get_loaders(
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
        contract_dir=None,
        verbose=True,
        scaling=True,
    )
    train_loader, val_loader, test_loader = output[0:3]

    print(f"Num train examples: {len(train_loader.dataset)}")
    print(f"Num val examples: {len(val_loader.dataset)}")
    print(f"Num test examples: {len(test_loader.dataset)}")
