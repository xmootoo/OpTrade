import os
import pandas as pd
from typing import Optional

# Custom modules
from optrade.data.thetadata.options import get_option_data
from optrade.data.thetadata.stocks import get_stock_data

from optrade.src.utils.data.clean_up import clean_up_dir

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def get_data(
    root: str="AAPL",
    start_date: str="20241107",
    end_date: str="20241107",
    exp: Optional[str]="20250117",
    strike: int=225,
    interval_min: int=1,
    right: str="C",
    save_dir: str="../historical_data/combined",
    clean_up: bool=False,
    offline: bool=False,
) -> pd.DataFrame:
    """
    Gets historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    combined stocks and options across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).

    *NOTE: Data from OHLC ends at 15:59:00, while quote data ends at 16:00:00, so for simplicity we
    remove all rows with 16:00:00 in datetime from quote data, before merging quote and OHLC data.

    Args:
        root (str): The root symbol of the underlying security.
        start_date (str): The start date of the data in YYYYMMDD format.
        end_date (str): The end date of the data in YYYYMMDD format.
        exp (Optional[str]): The expiration date of the option in YYYYMMDD format.
        strike (int): The strike price of the option in dollars.
        interval_min (int): The interval in minutes between data points.
        right (str): The type of option, either 'C' for call or 'P' for put.
        save_dir (str): The directory to save the data.
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                         saved in a temp folder and then subsequently deleted before returning the df.
        offline (bool): Whether to use offline (already saved) data instead of calling ThetaData API directly (default: False).

    Returns:
        DataFrame: The combined quote-level and OHLC data for an option and the underlying,
    """

    # Directory setup
    options_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "historical_data", "options")
    stocks_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "historical_data", "stocks")

    if clean_up and not offline:
        temp_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "temp", "combined")
        save_dir = temp_dir

    # Set up directory structure
    save_dir = os.path.join(save_dir, root, right, f"{start_date}_{end_date}", f"{strike}strike_{exp}exp")
    os.makedirs(save_dir, exist_ok=True)

    # Define file paths
    combined_file_path = os.path.join(save_dir, 'combined.csv')

    # If offline mode is enabled, read and return the combined data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(combined_file_path, parse_dates=['datetime'])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {combined_file_path}."
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    options_df = get_option_data(
        root=root,
        start_date=start_date,
        end_date=end_date,
        exp=exp,
        strike=strike,
        interval_min=interval_min,
        right=right,
        save_dir=options_dir,
        clean_up=clean_up,
        offline=offline,
    )

    stock_df = get_stock_data(
        root=root,
        start_date=start_date,
        end_date=end_date,
        interval_min=interval_min,
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
    options_df = options_df.rename(columns=option_columns)

    # Same for stocks
    stock_columns = {col: "stock_" + col for col in base_columns if col != "datetime"}
    stock_df = stock_df.rename(columns=stock_columns)

    # Assert that datetime columns are equal
    assert (options_df["datetime"] == stock_df["datetime"]).all(), "Datetime columns between option and stock data are not equal"

    # Merge the two dataframes and save
    df = pd.merge(options_df, stock_df, on="datetime")


    # Clean up temp files
    if clean_up:
        clean_up_dir(temp_dir)
    else:
        # Save combined data
        df.to_csv(combined_file_path, index=False)

    return df

if __name__ == "__main__":
    combined_df = get_data(clean_up=False, offline=False)
    print(combined_df.head())
