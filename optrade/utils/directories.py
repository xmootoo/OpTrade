import os
import shutil
from pathlib import Path
from typing import Tuple, Optional


def clean_up_file(file_path: str) -> None:
    """
    Removes a file from the filesystem.

    Args:
        file_path (str): Path to the file to be removed

    Returns:
        None
    """
    try:
        os.remove(file_path)
    except OSError as e:
        print(f"Warning: Could not delete file {file_path}: {e}")


def clean_up_dir(dir_path: str) -> None:
    """
    Remove a directory and all its contents (files and subdirectories).

    Args:
        dir_path (str): Path to the directory to be removed
    """
    try:
        shutil.rmtree(dir_path)
    except OSError as e:
        print(f"Warning: Could not delete directory {dir_path}: {e}")


def set_contract_dir(
    SCRIPT_DIR: Path,
    root: str = "AAPL",
    start_date: str = "20231107",
    end_date: str = "20241114",
    contract_stride: int = 5,
    interval_min: int = 1,
    right: str = "C",
    target_tte: int = 30,
    tte_tolerance: Tuple[int, int] = (25, 35),
    moneyness: str = "OTM",
    strike_band: float = 0.05,
    volatility_scaled: bool = True,
    volatility_scalar: float = 1.0,
    hist_vol: Optional[float] = None,
    save_dir: Optional[str] = None,
    dev_mode: bool = False,
) -> Path:
    """
    Sets up the directory structure for saving historical contract data.

    Args:
        SCRIPT_DIR (Path): Path to the script directory
        root (str): Root symbol of the contract
        start_date (str): Start date of the contract data
        end_date (str): End date of the contract data
        contract_stride (int): Number of days between contracts
        interval_min (int): Interval in minutes for the contract data
        right (str): Right of the contract (C for call, P for put)
        target_tte (int): Target time-to-expiration in days
        tte_tolerance (Tuple[int, int]): Lower and upper bounds for time-to-expiration
        moneyness (str): Moneyness of the contract (OTM, ATM, or ITM)
        strike_band (float): Band around the ATM strike price to define ITM and OTM
        volatility_scaled (bool): Whether to scale the strike price bands by historical volatility
        volatility_scalar (float): Scalar to multiply the historical volatility by
        hist_vol (Optional[float]): Historical volatility of the underlying
        save_dir (Optional[str]): Directory to save the contract data
        dev_mode (bool): Whether to use development mode

    Returns:
        Path: Path to the contract directory
    """

    working_dir = SCRIPT_DIR if dev_mode else Path.cwd()

    # Directory setup
    if save_dir is None:
        save_dir = working_dir / "historical_data" / "contracts"
    else:
        save_dir = Path(save_dir) / "contracts"

    # Create a structured path based on key parameters
    contract_dir = (
        save_dir
        / root
        / f"{start_date}_{end_date}"
        / right
        / f"contract_stride_{contract_stride}"
        / f"interval_{interval_min}"
        / f"target_tte_{target_tte}"
        / f"moneyness_{moneyness}"
    )

    # Add volatility info to path if volatility_scaled is True
    if volatility_scaled:
        contract_dir = (
            contract_dir / f"histvol_{hist_vol:.6f}_volscalar_{volatility_scalar}"
        )
    else:
        contract_dir = (
            contract_dir / f"strike_band_{str(strike_band).replace('.', 'p')}"
        )

    return contract_dir
