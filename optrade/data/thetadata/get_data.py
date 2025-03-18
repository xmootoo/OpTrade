import os
import pandas as pd
from pathlib import Path
from typing import Optional

# Custom modules
from optrade.data.thetadata.contracts import Contract
from optrade.data.thetadata.options import load_option_data
from optrade.data.thetadata.stocks import load_stock_data
from optrade.utils.data.clean_up import clean_up_dir
from optrade.utils.data.error import DataValidationError, INCOMPATIBLE_START_DATE, INCOMPATIBLE_END_DATE, MISSING_DATES, UNKNOWN_ERROR

SCRIPT_DIR = Path(__file__).resolve().parent

def load_all_data(
    contract: Contract,
    save_dir: Optional[str]=None,
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
    if save_dir is None:
        save_dir = SCRIPT_DIR.parent / "historical_data" / "combined"
        options_dir = SCRIPT_DIR.parent / "historical_data" / "options"
        stocks_dir = SCRIPT_DIR.parent / "historical_data" / "stocks"
    else:
        save_dir = Path(save_dir) / "combined"
        options_dir = Path(save_dir) / "options"
        stocks_dir = Path(save_dir) / "stocks"

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

    option_df = load_option_data(
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

    stock_df = load_stock_data(
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

        real_start_date = option_df["datetime"].iloc[0]
        stock_filtered_start = stock_df[stock_df["datetime"] >= real_start_date].reset_index(drop=True)

        real_end_date = option_df["datetime"].iloc[-1]
        stock_filtered = stock_filtered_start[stock_filtered_start["datetime"] <= real_end_date].reset_index(drop=True)

        # Verify if option_df starts earlier than stock_df
        if option_df["datetime"].to_list()==stock_filtered_start["datetime"].to_list():
            real_start_date = real_start_date.strftime("%Y%m%d")
            queried_start_date = stock_df["datetime"].iloc[0].strftime("%Y%m%d")
            raise DataValidationError(
                message=f"Option data queried on start_date={queried_start_date}, but the contract doesn't start until start_date={real_start_date}.",
                error_code=INCOMPATIBLE_START_DATE,
                real_start_date=real_start_date
            )

        # Otherwise verify if option_df starts earlier AND later than stock_df
        elif option_df["datetime"].to_list()==stock_filtered["datetime"].to_list():
            real_start_date = real_start_date.strftime("%Y%m%d")
            real_end_date = real_end_date.strftime("%Y%m%d")
            queried_start_date = stock_df["datetime"].iloc[0].strftime("%Y%m%d")
            queried_end_date = stock_df["datetime"].iloc[-1].strftime("%Y%m%d")
            raise DataValidationError(
                message=f"Option data queried on start_date={queried_end_date} with exp={queried_end_date}, but the contract starts on start_date={real_start_date} and ends on {real_end_date}.",
                error_code=INCOMPATIBLE_END_DATE,
                real_start_date=real_start_date,
                real_end_date=real_end_date
            )

        # Otherwise verify if option_df is a continuous subset of stock data (i.e. missing dates in between start_date and end_date)
        elif set(option_df["datetime"]).issubset(set(stock_filtered["datetime"])):
            option_dates_set = set(option_df["datetime"])
            stock_dates_set = set(stock_filtered["datetime"])

            # Find dates in option data that don't exist in stock data
            missing_in_stock = option_dates_set - stock_dates_set

            # Find dates in stock data that don't exist in option data
            missing_in_option = stock_dates_set - option_dates_set

            error_message = ""
            if missing_in_stock:
                error_message += f"\nDates in option data missing from stock data: {sorted(missing_in_stock)[:3]}"
                if len(missing_in_stock) > 3:
                    error_message += f" ...{sorted(missing_in_option)[-3:-1]}. Total number of missing dates: {len(missing_in_stock)}."

            if missing_in_option:
                error_message += f"\nDates in stock data missing from option data: {sorted(missing_in_option)[:3]}"
                if len(missing_in_option) > 3:
                    error_message += f" ...{sorted(missing_in_option)[-3:-1]}. Total number of missing dates: {len(missing_in_option)}."

            raise DataValidationError(
                message=error_message + "This discrepancy is likely due to anomalous market events.",
                error_code=MISSING_DATES)
        else:
            raise DataValidationError(
                message=f"Unknown error. Filtering stock data by start_date={real_start_date} and end_date={real_end_date} did not resolve the issue, and neither is option dates a subset of stock dates.",
                error_code=UNKNOWN_ERROR
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
        start_date="20230203",
        exp="20230217",
        strike=136,
        interval_min=1,
        right="C")

    from rich.console import Console
    ctx = Console()

    ctx.log(contract)

    combined_df = load_all_data(contract=contract, clean_up=False, offline=False)
    print(combined_df.head())
