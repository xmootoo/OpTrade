import os
import pandas as pd
from typing import Optional
from pathlib import Path

# Custom modules
from optrade.data.thetadata.options import get_option_data
from optrade.data.thetadata.stocks import get_stock_data

def get_data(
    root: str="AAPL",
    start_date: str="20241107",
    end_date: str="20241107",
    tte: Optional[int]=-1,
    exp: Optional[str]="20250117",
    strike: int=225,
    interval_min: int=1,
    right: str="C",
    save_dir: str="../historical_data/merged"
) -> pd.DataFrame:
    """
    Gets historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    merged stocks and options across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).

    *NOTE: Data from OHLC ends at 15:59:00, while quote data ends at 16:00:00, so for simplicity we
    remove all rows with 16:00:00 in datetime from quote data, before merging quote and OHLC data.

    Args:
        root (str): The root symbol of the underlying security.
        start_date (str): The start date of the data in YYYYMMDD format.
        end_date (str): The end date of the data in YYYYMMDD format.
        exp (Optional[str]): The expiration date of the option in YYYYMMDD format.
        tte (Optional[int]): The time to expiration of the option in days.
        strike (int): The strike price of the option in dollars.
        interval_min (int): The interval in minutes between data points.
        right (str): The type of option, either 'C' for call or 'P' for put.

    Returns:
        DataFrame: The merged quote-level and OHLC data.
    """

    script_dir = Path(__file__).parent  # Get the directory containing the current script
    options_dir = script_dir.parent / "historical_data/options"  # Go up one level and add the subdirectories
    stocks_dir = script_dir.parent / "historical_data/stocks"

    if tte!=-1:
        exp = pd.to_datetime(start_date, format='%Y%m%d') + pd.DateOffset(days=tte)
        exp = exp.strftime('%Y%m%d')

    options_df = get_option_data(
        root=root,
        start_date=start_date,
        end_date=end_date,
        tte=tte,
        exp=exp,
        strike=strike,
        interval_min=interval_min,
        right=right,
        save_dir=options_dir,
    )

    stock_df = get_stock_data(
        root=root,
        start_date=start_date,
        end_date=end_date,
        interval_min=interval_min,
        save_dir=stocks_dir,
    )

    # Rename columns of options df according to the dictionary
    base_columns = [
        "datetime",
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
    merged_df = pd.merge(options_df, stock_df, on="datetime")
    base_dir = os.path.join(save_dir, root, right, f"{start_date}_{end_date}", f"{strike}strike_{exp}exp")
    os.makedirs(base_dir, exist_ok=True)
    file_path = os.path.join(base_dir, "merged.csv")
    merged_df.to_csv(file_path, index=False)

    return merged_df

if __name__ == "__main__":
    merged_df = get_data()
