import numpy as np
import torch
import torch.nn as nn
import pandas as pd
from typing import Tuple, Optional
from pathlib import Path

# Stock data
from optrade.data.thetadata.stocks import get_stock_data

# Datasets
from optrade.src.preprocessing.data.datasets import ContractDataset, ForecastingDataset, IntradayForecastingDataset

# Historical Volatility
from optrade.src.preprocessing.data.volatility import get_historical_volatility

# Get absolute path for this script
SCRIPT_DIR = Path(__file__).resolve().parent

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


if __name__ == "__main__":
    root="AMZN"
    total_start_date="20220101"
    total_end_date="20220301"
    right="C"
    interval_min=1
    contract_stride=15
    target_tte=10
    moneyness="ATM"
    volatility_scaled=True
    volatility_scalar=1.0
    volatility_type="period"
    target_band=0.05

    contracts = get_contract_datasets(root=root,
    start_date=total_start_date,
    end_date=total_end_date,
    train_split=0.7,
    val_split=0.1,
    contract_stride=contract_stride,
    interval_min=interval_min,
    right=right,
    target_tte=target_tte,
    tte_tolerance=(5, 15),
    moneyness=moneyness,
    target_band=target_band,
    volatility_scaled=volatility_scaled,
    volatility_scalar=volatility_scalar,
    volatility_type=volatility_type,
    clean_up=False,
    offline=False,
    )

    # TODO: Test loading each dataset

    save_dir = SCRIPT_DIR.parents[2] / "data" / "historical_data" / "contracts"
    contract_dir = (
        save_dir /
        root /
        f"{total_start_date}_{total_end_date}" /
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

    train_contracts = ContractDataset.load(contract_dir / "train_contracts.pkl")
    print(train_contracts.contracts)
