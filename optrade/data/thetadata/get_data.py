import os
import pandas as pd
from typing import Optional
<<<<<<< HEAD
from datetime import datetime
=======
from pathlib import Path

>>>>>>> 666d177d1c5b6c7e237bcecb3092f7d32b4deb70
# Custom modules
from optrade.data.thetadata.contracts import Contract
from optrade.data.thetadata.options import get_option_data
from optrade.data.thetadata.stocks import get_stock_data
from optrade.src.utils.data.clean_up import clean_up_dir
from optrade.src.utils.data.error import DataValidationError, OPTION_DATE_MISMATCH, MISSING_DATES

SCRIPT_DIR = Path(__file__).resolve().parent

def get_data(
<<<<<<< HEAD
    start_date: datetime,
    end_date: datetime,
    /, 
    exp: Optional[str] = None,
    root: str="AAPL",
    strike: int=225,
    interval_min: int=1,
    right: str="C",
    save_dir: str="../historical_data/merged",
=======
    contract: Contract,
    save_dir: str="../historical_data/combined",
>>>>>>> 666d177d1c5b6c7e237bcecb3092f7d32b4deb70
    clean_up: bool=False,
    offline: bool=False,
) -> pd.DataFrame:
    """
    Gets historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    combined stocks and options across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).

    *NOTE: Data from OHLC ends at 15:59:00, while quote data ends at 16:00:00, so for simplicity we
    remove all rows with 16:00:00 in datetime from quote data, before merging quote and OHLC data.

    Args:
        contract (Contract): A Pydantic model representing an options contract. Attributes:
            - contract.root (str): The root symbol of the underlying security.
            - contract.start_date (str): The start date of the data in YYYYMMDD format.
            - contract.exp (Optional[str]): The expiration date of the option in YYYYMMDD format.
            - contract.strike (int): The strike price of the option in dollars.
            - contract.interval_min (int): The interval in minutes between data points.
            - contract.right (str): The type of option, either 'C' for call or 'P' for put.
        save_dir (str): The directory to save the data.
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                         saved in a temp folder and then subsequently deleted before returning the df.
        offline (bool): Whether to use offline (already saved) data instead of calling ThetaData API directly (default: False).

    Returns:
        DataFrame: The combined quote-level and OHLC data for an option and the underlying,
    """

    # Directory setup
    options_dir = SCRIPT_DIR.parent / "historical_data" / "options"
    stocks_dir = SCRIPT_DIR.parent / "historical_data" / "stocks"
    save_dir = Path(save_dir)

    if clean_up and not offline:
        temp_dir = SCRIPT_DIR.parent / "temp" / "combined"
        save_dir = Path(temp_dir)

    # Set up directory structure
    save_dir = (
        save_dir /
        contract.root /
        contract.right /
        f"{contract.start_date}_{contract.exp}" /
        f"{contract.strike}strike_{contract.exp}exp"
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    combined_file_path = save_dir / "combined.csv"

    # If offline mode is enabled, read and return the combined data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(combined_file_path, parse_dates=['datetime'])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {combined_file_path}."
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    option_df = get_option_data(
        root=contract.root,
        start_date=contract.start_date,
        end_date=contract.exp,
        exp=contract.exp,
        strike=contract.strike,
        interval_min=contract.interval_min,
        right=contract.right,
        save_dir=options_dir,
        clean_up=clean_up,
        offline=offline,
    )

    stock_df = get_stock_data(
        root=contract.root,
        start_date=contract.start_date,
        end_date=contract.exp,
        interval_min=contract.interval_min,
        save_dir=stocks_dir,
        clean_up=clean_up,
        offline=offline,
    )

    # Rename columns of options df according to the dictionary
    base_columns = [
        "datetime",
        "mid_price",
        "bid_size",
        "bid_exchange",
        "bid",
        "bid_condition",
        "ask_size",
        "ask_exchange",
        "ask",
        "ask_condition",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "count"
    ]

    option_columns = {col: "option_" + col for col in base_columns if col != "datetime"}
    option_df = option_df.rename(columns=option_columns)

    # Same for stocks
    stock_columns = {col: "stock_" + col for col in base_columns if col != "datetime"}
    stock_df = stock_df.rename(columns=stock_columns)

    # Check lengths of option_df and stock_df match exactly
    if len(option_df) != len(stock_df):

        # If lengths don't match, verify that option_df is a subset of stock_df
        # starting at a later date but with identical dates once options begin
        option_start_date = option_df["datetime"].iloc[0]
        stock_filtered = stock_df[stock_df["datetime"] >= option_start_date].reset_index(drop=True)

        if (option_df["datetime"]==stock_filtered["datetime"]).all():
            option_start_date = option_start_date.strftime("%Y%m%d")
            raise DataValidationError(
                f"Date mismatch between option and stock data. Option data starts on {option_start_date}, but stock data starts earlier.",
                OPTION_DATE_MISMATCH, option_start_date)
        else:
            missing_dates = option_df.loc[~option_df["datetime"].isin(stock_df["datetime"]), "datetime"].unique()
            raise DataValidationError(
                f"Date mismatch between option and stock data. The discrepancy is not simply that stock data starts earlier. \
                There are dates in options data that don't exist in stock data: {missing_dates}",
                MISSING_DATES
            )

    # Merge the two dataframes and save
    df = pd.merge(option_df, stock_df, on="datetime")

    # Clean up temp files
    if clean_up:
        clean_up_dir(temp_dir)
    else:
        # Save combined data
        df.to_csv(combined_file_path, index=False)

    return df

if __name__ == "__main__":
    from optrade.data.thetadata.contracts import Contract

    # contract = Contract().find_optimal(
    #     root="AAPL",
    #     start_date="20241107",
    #     interval_min=1,
    #     right="C",
    #     target_tte=30,
    #     tte_tolerance=(25, 35),
    #     moneyness="OTM",
    #     target_band=0.05,
    #     hist_vol=0.1,
    #     volatility_scaled=True,
    #     volatility_scalar=1.0
    # )

    contract = Contract(
        root="AMZN",
        start_date="20230118",
        exp="20230217",
        strike=97,
        interval_min=1,
        right="C")

    from rich.console import Console
    ctx = Console()

    ctx.log(contract)

    combined_df = get_data(contract=contract, clean_up=False, offline=False)
    print(combined_df.head())
