import pickle
from pathlib import Path
import pandas as pd
from typing import Tuple, Iterator, Dict, Any, Optional
from datetime import datetime, timedelta
from rich.console import Console
from pydantic import BaseModel, Field
import pandas_market_calendars as mcal

# Custom modules
from optrade.data.thetadata import load_all_data
from optrade.data.thetadata import find_optimal_exp
from optrade.data.thetadata import find_optimal_strike
from optrade.utils.directories import set_contract_dir
from optrade.utils.error_handlers import DataValidationError, MARKET_HOLIDAY, WEEKEND
from optrade.utils.volatility import get_train_historical_vol

SCRIPT_DIR = Path(__file__).resolve().parent


class Contract:
    """
    A class representing an options contract with methods for optimal contract selection.

    The Contract class defines the structure of an options contract including the underlying security,
    dates, strike price, and other key parameters.
    """

    def __init__(
        self,
        root: str = "AAPL",
        start_date: str = "20241107",
        exp: str = "20241206",
        strike: float = 225,
        interval_min: int = 1,
        right: str = "C",
    ):
        """Initialize a Contract instance.

        Args:
            root: Root symbol of the underlying security
            start_date: Start date in YYYYMMDD format
            exp: Expiration date in YYYYMMDD format
            strike: Strike price
            interval_min: Interval in minutes
            right: Option type ('C' for call, 'P' for put)
        """
        self.root = root
        self.start_date = start_date
        self.exp = exp
        self.strike = strike
        self.interval_min = interval_min
        self.right = right

    def __eq__(self, other):
        """Check if two contracts are equal."""
        if not isinstance(other, Contract):
            return False
        return (
            self.root == other.root
            and self.start_date == other.start_date
            and self.exp == other.exp
            and self.strike == other.strike
            and self.interval_min == other.interval_min
            and self.right == other.right
        )

    def __hash__(self):
        """Hash the contract for use in dictionaries and sets."""
        return hash(
            (
                self.root,
                self.start_date,
                self.exp,
                self.strike,
                self.interval_min,
                self.right,
            )
        )

    def __repr__(self):
        """Returns a string representation of the contract."""
        return f"Contract(root='{self.root}', start_date='{self.start_date}', exp='{self.exp}', strike={self.strike}, interval_min={self.interval_min}, right='{self.right}')"

    @classmethod
    def find_optimal(
        cls,
        root: str,
        start_date: str,
        interval_min: int,
        right: str,
        target_tte: int,
        tte_tolerance: Tuple[int, int],
        moneyness: str,
        strike_band: Optional[float] = 0.05,
        hist_vol: Optional[float] = None,
        volatility_scaled: bool = False,
        volatility_scalar: Optional[float] = 1.0,
        verbose: bool = True,
        warning: bool = False,
        dev_mode: bool = False,
    ) -> "Contract":
        """Find the optimal contract for a given security, start date, and approximate TTE.

        Args:
            root: Underlying stock symbol
            start_date: Start date for the contract in YYYYMMDD format
            interval_min: Interval in minutes
            right: Option type (C for call, P for put)
            target_tte: Target time to expiration in days
            tte_tolerance: Acceptable range for TTE as (min_days, max_days)
            moneyness: Contract moneyness (OTM, ATM, ITM)
            strike_band: Target percentage band for strike selection
            hist_vol: Historical volatility for dynamic strike selection
            volatility_scaled: Whether to select strike by volatility
            volatility_scalar: Scaling factor for volatiliy-based strike selection
            verbose: Whether to print verbose output
        """

        exp, _ = find_optimal_exp(
            root=root,
            start_date=start_date,
            target_tte=target_tte,
            tte_tolerance=tte_tolerance,
            clean_up=True,
            dev_mode=dev_mode,
        )

        ctx = Console()

        # Validate if start_date is a trading day
        date_obj = datetime.strptime(start_date, "%Y%m%d")

        # Check if it's a weekend
        if date_obj.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
            raise DataValidationError(
                message=f"Start date {start_date} falls on a weekend. Markets are closed.",
                error_code=WEEKEND,
                verbose=verbose,
                warning=warning,
            )

        # Check if it's a market holiday
        nyse = mcal.get_calendar("NYSE")
        trading_days = nyse.valid_days(start_date=start_date, end_date=start_date)

        if len(trading_days) == 0:
            raise DataValidationError(
                message=f"Start date {start_date} is a market holiday. Markets are closed.",
                error_code=MARKET_HOLIDAY,
                verbose=verbose,
                warning=warning,
            )

        strike = find_optimal_strike(
            root=root,
            start_date=start_date,
            exp=exp,
            right=right,
            interval_min=interval_min,
            moneyness=moneyness,
            strike_band=strike_band,
            hist_vol=hist_vol,
            volatility_scaled=volatility_scaled,
            volatility_scalar=volatility_scalar,
            clean_up=True,
            dev_mode=dev_mode,
        )

        if verbose:
            ctx.log(
                f"Identified optimal contract with strike price of ${strike} expiring on {exp}"
            )

        return cls(
            root=root,
            start_date=start_date,
            exp=exp,
            strike=strike,
            interval_min=interval_min,
            right=right,
        )

    def load_data(
        self,
        clean_up: bool = False,
        offline: bool = False,
        save_dir: Optional[str] = None,
        warning: bool = False,
        dev_mode: bool = False,
    ) -> pd.DataFrame:
        """Load data for the selected contract.

        Args:
            clean_up: Whether to clean up the data after use
            offline: Whether to load saved data from disk
            save_dir: Directory to save/load data
            warning: Whether to display warnings
            dev_mode: Whether to use development mode

        Returns:
            pd.DataFrame: The loaded data containing NBBO quotes and OHLCVC data for the contract and the underlying
        """
        return load_all_data(
            root=self.root,
            start_date=self.start_date,
            exp=self.exp,
            interval_min=self.interval_min,
            right=self.right,
            strike=self.strike,
            clean_up=clean_up,
            offline=offline,
            save_dir=save_dir,
            warning=warning,
            dev_mode=dev_mode,
        )


class ContractDataset:
    """
    A dataset containing options contracts generated with consistent parameters.
    """

    def __init__(
        self,
        root: str = "AAPL",
        total_start_date: str = "20231107",
        total_end_date: str = "20241114",
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
        verbose: bool = False,
        save_dir: Optional[str] = None,
        warning: bool = True,
        dev_mode: bool = False,
        contract_dir: Optional[Path] = None,
    ) -> None:
        """
        Initialize the ContractDataset with the specified parameters.

        Args:
            root: The security root symbol
            total_start_date: Start date for the dataset (YYYYMMDD)
            total_end_date: End date for the dataset (YYYYMMDD)
            contract_stride: Days between consecutive contracts
            interval_min: Data interval in minutes
            right: Option type (C/P)
            target_tte: Target time to expiration in days
            tte_tolerance: Acceptable range for TTE as (min_days, max_days)
            moneyness: Contract moneyness (OTM/ATM/ITM)
            strike_band: Target percentage band for strike selection
            volatility_scaled: Whether to scale by volatility
            volatility_scalar: Scaling factor for volatility
            hist_vol: Historical volatility for dynamic strike selection
            verbose: Whether to print verbose output
        """
        self.root = root
        self.total_start_date = total_start_date
        self.total_end_date = total_end_date
        self.contract_stride = contract_stride
        self.interval_min = interval_min
        self.right = right
        self.target_tte = target_tte
        self.tte_tolerance = tte_tolerance
        self.moneyness = moneyness
        self.strike_band = strike_band
        self.volatility_scaled = volatility_scaled
        self.volatility_scalar = volatility_scalar
        self.hist_vol = hist_vol
        self.verbose = verbose
        self.warning = warning
        self.dev_mode = dev_mode

        self.contracts = []

        if contract_dir is None:
            self.contract_dir = set_contract_dir(
                SCRIPT_DIR=SCRIPT_DIR,
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
                volatility_scaled=volatility_scaled,
                volatility_scalar=volatility_scalar,
                hist_vol=hist_vol,
                save_dir=save_dir,
                dev_mode=dev_mode,
            )
        else:
            self.contract_dir = Path(contract_dir)

    def generate(self) -> "ContractDataset":
        """
        Generate all contracts in the dataset based on configuration parameters. Contracts are generated by starting from total_start_date and
        advancing by contract_stride days until reaching the last valid date that allows for contracts within the specified time-to-expiration tolerance.

        Returns:
            ContractDataset: The dataset with all generated contracts
        """
        ctx = Console()
        # Parse dates
        start_date = datetime.strptime(self.total_start_date, "%Y%m%d")
        end_date = datetime.strptime(self.total_end_date, "%Y%m%d")
        max_tte = max(self.tte_tolerance)

        # Calculate the latest possible start date
        latest_start = end_date - timedelta(days=max_tte)

        # Generate contracts
        current_date = start_date

        while current_date <= latest_start:
            # Format initial date string
            date_str = current_date.strftime("%Y%m%d")
            attempt_date = current_date
            contract = None

            # Find a valid contract for the current date. Some dates may be ineligible due to holidays or weekends.
            while contract is None and attempt_date <= latest_start:
                attempt_date_str = attempt_date.strftime("%Y%m%d")
                try:
                    contract = Contract.find_optimal(
                        root=self.root,
                        start_date=attempt_date_str,
                        interval_min=self.interval_min,
                        right=self.right,
                        target_tte=self.target_tte,
                        tte_tolerance=self.tte_tolerance,
                        moneyness=self.moneyness,
                        strike_band=self.strike_band,
                        hist_vol=self.hist_vol,
                        volatility_scaled=self.volatility_scaled,
                        volatility_scalar=self.volatility_scalar,
                        verbose=self.verbose,
                        warning=self.warning,
                        dev_mode=self.dev_mode,
                    )

                    if attempt_date > current_date:
                        (
                            ctx.log(
                                f"Found valid contract at shifted date: {attempt_date_str}"
                            )
                            if self.verbose
                            else None
                        )

                except DataValidationError as e:
                    if e.error_code == WEEKEND:
                        (
                            ctx.log(f"Skipping weekend: {attempt_date_str}")
                            if self.verbose
                            else None
                        )
                        attempt_date += timedelta(days=1)
                    elif e.error_code == MARKET_HOLIDAY:
                        (
                            ctx.log(f"Skipping market holiday: {attempt_date_str}")
                            if self.verbose
                            else None
                        )
                        attempt_date += timedelta(days=1)
                    else:
                        (
                            ctx.log(
                                f"Unkown error: {str(e)}. Skipping date: {attempt_date_str}."
                            )
                            if self.verbose
                            else None
                        )

                    # Check if we've run out of valid dates
                    if attempt_date > latest_start:
                        (
                            ctx.log(
                                f"Unable to find valid contract starting from {date_str}"
                            )
                            if self.verbose
                            else None
                        )
                        break

                    continue

            # If we found a valid contract, add it and advance by stride
            if contract is not None:
                self.contracts.append(contract)
                ctx.log(f"Added contract: {contract}") if self.verbose else None
                current_date = attempt_date + timedelta(days=self.contract_stride)
            else:
                # If no contract was found, advance by one day to try the next period
                current_date += timedelta(days=1)

            (
                ctx.log(f"Next start date: {current_date.strftime('%Y%m%d')}")
                if self.verbose
                else None
            )

        return self

    def __len__(self) -> int:
        """Get the number of contracts in the dataset.

        Returns:
            int: Number of contracts
        """
        return len(self.contracts)

    def __getitem__(self, idx) -> Contract:
        """Get a contract by index.

        Returns:
            Contract: The contract at the specified index
        """
        return self.contracts[idx]

    def __iter__(self) -> Iterator:
        """Iterate through contracts.

        Returns:
            Iterator: Iterator for contracts
        """
        return iter(self.contracts)

    def save(self, filename: Optional[str] = None, clean_file: bool = False) -> None:
        """Save the dataset to a pickle file.

        Args:
            filepath: Optional custom filepath. If None, generates default name

        Returns:
            str: Path where the pickle file was saved
        """
        self.contract_dir.mkdir(parents=True, exist_ok=True)

        if not hasattr(self, "filepath"):
            self.filepath = (
                self.contract_dir / "contracts.pkl"
                if filename is None
                else self.contract_dir / filename
            )
        elif clean_file and self.filepath.exists():
            self.filepath.unlink()

        # Just pickle the whole dataset with contracts as-is
        with open(self.filepath, "wb") as f:
            pickle.dump(self, f)

        if self.verbose:
            ctx = Console()
            ctx.log(f'Contract dataset saved to "{self.filepath}"')

    @classmethod
    def load(cls, filepath: Path) -> "ContractDataset":
        """Load a dataset from a pickle file."""
        ctx = Console()

        with open(filepath, "rb") as f:
            instance = pickle.load(f)

        if instance.verbose:
            ctx.log(f"Contract dataset loaded from {filepath}")

        return instance


def get_contract_datasets(
    root: str,
    start_date: str,
    end_date: str,
    contract_stride: int,
    interval_min: int,
    right: str,
    target_tte: int,
    tte_tolerance: Tuple[int, int],
    moneyness: str,
    strike_band: Optional[float] = 0.05,
    volatility_type: Optional[str] = "period",
    volatility_scaled: Optional[bool] = False,
    volatility_scalar: Optional[float] = 1.0,
    train_split: float = 0.7,
    val_split: float = 0.1,
    clean_up: bool = True,
    offline: bool = False,
    save_dir: Optional[str] = None,
    verbose: bool = False,
    dev_mode: bool = False,
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
        strike_band: Target band for moneyness selection, proportion of current underlying price
        volatility_type: Type of historical volatility to use
        volatility_scaled: Whether to scale strikes based on historical volatility
        volatility_scalar: Scalar to adjust historical volatility-based strike selection
        train_split: Proportion of total days to use for training
        val_split: Proportion of total days to use for validation
        clean_up: Whether to clean up the data after use
        offline: Whether to load saved contracts from disk
        save_dir: Directory to save/load contracts
        verbose: Whether to print verbose output
        dev_mode: Whether to use development mode

    Returns:
        Training, validation, and test contract datasets.
    """

    # Volatility-based selection of strikes (Optional)
    if volatility_scaled:
        hist_vol = get_train_historical_vol(
            root=root,
            start_date=start_date,
            end_date=end_date,
            interval_min=interval_min,
            volatility_window=train_split,  # Use the ONLY training data to compute historical volatility (prevent data leakage)
            volatility_type=volatility_type,
        )
    else:
        hist_vol = None

    contract_dir = set_contract_dir(
        SCRIPT_DIR=SCRIPT_DIR,
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
        volatility_scaled=volatility_scaled,
        volatility_scalar=volatility_scalar,
        hist_vol=hist_vol,
        save_dir=save_dir,
        dev_mode=dev_mode,
    )

    # Offline loading (if already saved)
    if offline:
        if not all(
            (
                (contract_dir / "train_contracts.pkl").exists(),
                (contract_dir / "val_contracts.pkl").exists(),
                (contract_dir / "test_contracts.pkl").exists(),
            )
        ):
            raise FileNotFoundError(f"Missing contract files in {contract_dir}")

        train_contracts = ContractDataset.load(contract_dir / "train_contracts.pkl")
        val_contracts = ContractDataset.load(contract_dir / "val_contracts.pkl")
        test_contracts = ContractDataset.load(contract_dir / "test_contracts.pkl")
        return train_contracts, val_contracts, test_contracts

    # Get contiguous training, validation, and test (start_date, end_date) pairs in YYYYMMDD format
    total_days = (
        pd.to_datetime(end_date, format="%Y%m%d")
        - pd.to_datetime(start_date, format="%Y%m%d")
    ).days
    num_train_days = int(train_split * total_days)
    num_val_days = int(val_split * total_days)

    train_end_date = (
        pd.to_datetime(start_date, format="%Y%m%d") + pd.Timedelta(days=num_train_days)
    ).strftime("%Y%m%d")
    val_end_date = (
        pd.to_datetime(train_end_date, format="%Y%m%d")
        + pd.Timedelta(days=num_val_days)
    ).strftime("%Y%m%d")
    test_start_date = (
        pd.to_datetime(val_end_date, format="%Y%m%d") + pd.Timedelta(days=1)
    ).strftime("%Y%m%d")

    train_dates = (start_date, train_end_date)
    val_dates = (train_end_date, val_end_date)
    test_dates = (test_start_date, end_date)

    # Create the training, validation, and test contract datasets
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
        strike_band=strike_band,
        hist_vol=hist_vol,
        volatility_scaled=volatility_scaled,
        volatility_scalar=volatility_scalar,
        verbose=verbose,
        save_dir=save_dir,
        contract_dir=contract_dir,
        dev_mode=dev_mode,
    ).generate()

    (
        ctx.log("------------CREATING VALIDATION CONTRACTS------------")
        if verbose
        else None
    )
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
        strike_band=strike_band,
        hist_vol=hist_vol,
        volatility_scaled=volatility_scaled,
        volatility_scalar=volatility_scalar,
        verbose=verbose,
        save_dir=save_dir,
        contract_dir=contract_dir,
        dev_mode=dev_mode,
    ).generate()

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
        strike_band=strike_band,
        hist_vol=hist_vol,
        volatility_scaled=volatility_scaled,
        volatility_scalar=volatility_scalar,
        verbose=verbose,
        save_dir=save_dir,
        contract_dir=contract_dir,
        dev_mode=dev_mode,
    ).generate()

    # Check that train_contracts, val_contracts, and test_contracts are nonempty
    if not all(
        (len(train_contracts) > 0, len(val_contracts) > 0, len(test_contracts) > 0)
    ):
        raise ValueError(
            f"One or more contract datasets are empty. "
            f"Number of train contracts: {len(train_contracts)}, "
            f"Number of val contracts: {len(val_contracts)}, "
            f"Number of test contracts: {len(test_contracts)}. "
            f"Try adjusting contract_stride to sample more contracts or "
            f"train_split/val_split ratios for more equal distribution of contracts."
        )
    if not clean_up:
        train_contracts.save("train_contracts.pkl")
        val_contracts.save("val_contracts.pkl")
        test_contracts.save("test_contracts.pkl")

    return train_contracts, val_contracts, test_contracts


if __name__ == "__main__":

    # Test: Contract
    contract = Contract.find_optimal(
        root="AAPL",
        start_date="20241107",
        interval_min=1,
        right="C",
        target_tte=30,
        tte_tolerance=(25, 35),
        moneyness="OTM",
        strike_band=0.05,
        volatility_scaled=False,
        verbose=True,
    )

    df = contract.load_data(clean_up=True, offline=False, warning=True, dev_mode=True)

    print(df.head())

    # Test: get_contract_datasets()
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
    strike_band = 0.05

    train_contract_dataset, val_contract_dataset, test_contract_dataset = (
        get_contract_datasets(
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
            train_split=0.4,
            val_split=0.3,
            clean_up=True,
            offline=False,
            save_dir=None,
            verbose=True,
        )
    )

    train_contract_dataset.save("train_contracts.pkl")

    from rich.console import Console

    ctx = Console()
    ctx.log(f"Contract dir: {train_contract_dataset.contract_dir}")
    new_contract_dataset = ContractDataset.load(
        train_contract_dataset.contract_dir / "train_contracts.pkl"
    )
