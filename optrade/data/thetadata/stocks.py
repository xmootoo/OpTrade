import httpx
import csv
import pandas as pd
import os
from typing import Optional, Tuple
from pathlib import Path
from rich.console import Console

# Custom modules
from optrade.utils.data.clean_up import clean_up_dir
SCRIPT_DIR = Path(__file__).resolve().parent

def load_stock_data(
    root: str="AAPL",
    start_date: str="20231107",
    end_date: str="20231107",
    interval_min: int=1,
    save_dir: Optional[str]=None,
    clean_up: bool=False,
    offline: bool=False,
) -> pd.DataFrame:

    """
    Gets historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    stocks across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).

    *NOTE: Data from OHLC ends at 15:59:00, while quote data ends at 16:00:00, so for simplicity we
    remove all rows with 16:00:00 in datetime from quote data, before merging quote and OHLC data.

    Args:
        root (str): The root symbol of the underlying security.
        start_date (str): The start date of the data in YYYYMMDD format.
        end_date (str): The end date of the data in YYYYMMDD format.
        interval_min (int): The interval in minutes between data points.
        save_dir (str): The directory to save the data.
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                         saved in a temp folder and then subsequently deleted before returning the df.

    Returns:
        DataFrame: The merged quote-level and OHLC data.
    """

    BASE_URL = "http://127.0.0.1:25510/v2"  # all endpoints use this URL base

    intervals = str(interval_min * 60000) # Convert minutes to milliseconds

    ctx = Console()

    params = {
        'root': root,
        'start_date': start_date,
        'end_date': end_date,
        'use_csv': 'true',
        'ivl': intervals,
        'venue': "utp_cta", # Merged UTP & CTA data
    }

    # Set up directory structure
    if save_dir is None:
        save_dir = SCRIPT_DIR.parent / "historical_data" / "stocks"
    else:
        save_dir = Path(save_dir) / "stocks"

    if clean_up and not offline:
        temp_dir = SCRIPT_DIR.parent / "temp" / "stocks"
        save_dir = Path(temp_dir)

    save_dir = save_dir / root / f"{start_date}_{end_date}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    quote_file_path = save_dir / "quote.csv"
    ohlc_file_path = save_dir / "ohlc.csv"
    merged_file_path = save_dir / "merged.csv"

    # If offline mode is enabled, read and return the merged data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(merged_file_path, parse_dates=['datetime'])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {merged_file_path}. "
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    # <-- Quote data -->
    quote_url = BASE_URL + '/hist/stock/quote'

    while quote_url is not None:
        quote_response = httpx.get(quote_url, params=params, timeout=1000)  # make the request
        quote_response.raise_for_status()  # make sure the request worked

        # read the entire quote_response, and parse it as CSV
        csv_reader = csv.reader(quote_response.text.split("\n"))

        # Convert to pandas dataframe. The first row are the column names
        quote_df = pd.DataFrame(list(csv_reader)[1:], # Skip the first row and use it as columns
                        columns=next(csv.reader(quote_response.text.split("\n"))))  # Use first row as column names

        # Convert to float64
        numeric_columns = quote_df.columns.difference(['date'])
        quote_df[numeric_columns] = quote_df[numeric_columns].astype('float64')

        # Create a datetime column in standard format in 'YYYY-MM-DD HH:MM:SS' (e.g., 2022-09-27 18:00:00)
        # First get time of day in HH:MM:SS format from ms_of_day
        time_of_day = pd.to_datetime(pd.to_numeric(quote_df['ms_of_day']), unit='ms').dt.time

        # Then get the date in YYYY-MM-DD format
        date = pd.to_datetime(quote_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

        # Then concatenate the date and time_of_day
        quote_df['datetime'] = pd.to_datetime(date + ' ' + time_of_day.astype(str))

        # Remove date and ms_of_day (redundant information)
        quote_df = quote_df.drop(columns=['date', 'ms_of_day'])

        # Remove any rows with 16:00:00 in datetime, as OHLC data ends at 15:59:00
        quote_df = quote_df[quote_df['datetime'].dt.time != pd.to_datetime('16:00:00').time()]

        # Make datetime the first column
        quote_df = quote_df.reindex(columns=['datetime'] + list(quote_df.columns.drop('datetime')))

        # Save with mode='a' (append) after the first write
        if quote_url == BASE_URL + '/hist/stock/quote':
            quote_df.to_csv(quote_file_path, index=False) # Save first batch as .csv.
        else:
            quote_df.to_csv(quote_file_path, mode='a', header=False, index=False) # Update .csv for subsequent batches

        # check the Next-Page header to see if we have more data
        if 'Next-Page' in quote_response.headers and quote_response.headers['Next-Page'] != "null":
            quote_url = quote_response.headers['Next-Page']
            params = None
            ctx.log(f"Paginating to {quote_url}")
        else:
            quote_url = None

    # Redefine params for OHLC data (as it will set to None)
    params = {
        'root': root,
        'start_date': start_date,
        'end_date': end_date,
        'use_csv': 'true',
        'ivl': intervals,
        'venue': "utp_cta", # Merged UTP & CTA data
    }

    # <-- OHLC data -->
    ohlc_url = BASE_URL + '/hist/stock/ohlc'

    while ohlc_url is not None:
        ohlc_response = httpx.get(ohlc_url, params=params, timeout=1000)
        ohlc_response.raise_for_status()

        # read the entire response, and parse it as CSV
        csv_reader = csv.reader(ohlc_response.text.split("\n"))
        ohlc_df = pd.DataFrame(list(csv_reader)[1:],
                        columns=next(csv.reader(ohlc_response.text.split("\n"))))

        # Convert to float64
        numeric_columns = ohlc_df.columns.difference(['date'])
        ohlc_df[numeric_columns] = ohlc_df[numeric_columns].astype('float64')

        # Create datetime column
        time_of_day = pd.to_datetime(pd.to_numeric(ohlc_df['ms_of_day']), unit='ms').dt.time
        date = pd.to_datetime(ohlc_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')
        ohlc_df['datetime'] = pd.to_datetime(date + ' ' + time_of_day.astype(str))

        # Remove redundant columns
        ohlc_df = ohlc_df.drop(columns=['date', 'ms_of_day'])

        # Save with mode='a' (append) after the first write
        if ohlc_url == BASE_URL + '/hist/stock/ohlc':
            ohlc_df.to_csv(ohlc_file_path, index=False) # Save first batch as .csv
        else:
            ohlc_df.to_csv(ohlc_file_path, mode='a', header=False, index=False) # Update .csv for subsequent batches

        # Check for next page
        if 'Next-Page' in ohlc_response.headers and ohlc_response.headers['Next-Page'] != "null":
            ohlc_url = ohlc_response.headers['Next-Page']
            params = None
            ctx.log(f"Paginating to {ohlc_url}")
        else:
            ohlc_url = None

    # Read CSVs with datetime parsing
    quote_df = pd.read_csv(quote_file_path, parse_dates=['datetime'])
    ohlc_df = pd.read_csv(ohlc_file_path, parse_dates=['datetime'])

    # Merge on datetime to ensure proper alignment
    merged_df = pd.merge(quote_df, ohlc_df, on='datetime', how='inner')

    # Remove any duplicate columns that might exist in both dataframes
    duplicate_cols = merged_df.columns.duplicated()
    merged_df = merged_df.loc[:, ~duplicate_cols]

    # Remove last row (NaN)
    merged_df = merged_df.dropna()

    # Calculate regular mid prices
    merged_df["mid_price"] = (merged_df["bid"] + merged_df["ask"]) / 2

    # Clean up the entire temp_dir
    if clean_up:
        clean_up_dir(temp_dir)
    else:
        # Save merged data
        merged_df.to_csv(merged_file_path, index=False)


    return merged_df

def load_stock_data_eod(
    root: str="AAPL",
    start_date: str="20231107",
    end_date: str="20231107",
    save_dir: Optional[str]=None,
    clean_up: bool=False,
    offline: bool=False,
) -> pd.DataFrame:

    """
    Gets historical End of Day (EOD) report from ThetaData API for stocks across multiple exchanges.
    Each report is generated around 17:15:00 ET and contain NBBO and OHLCVC data.

    Args:
        root (str): The root symbol of the underlying security.
        start_date (str): The start date of the data in YYYYMMDD format.
        end_date (str): The end date of the data in YYYYMMDD format.
        save_dir (str): The directory to save the data.
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                         saved in a temp folder and then subsequently deleted before returning the df.
        offline (bool): Whether to work in offline mode, using previously saved data.

    Returns:
        DataFrame: The merged quote-level and OHLC data.
    """

    BASE_URL = "http://127.0.0.1:25510/v2"  # all endpoints use this URL base

    ctx = Console()

    params = {
        'root': root,
        'start_date': start_date,
        'end_date': end_date,
        'use_csv': 'true',
    }

    # Set up directory structure
    if save_dir is None:
        save_dir = SCRIPT_DIR.parent / "historical_data" / "stocks"
    else:
        save_dir = Path(save_dir) / "stocks"

    if clean_up and not offline:
        temp_dir = SCRIPT_DIR.parent / "temp" / "stocks"
        save_dir = Path(temp_dir)

    save_dir = save_dir / root / f"{start_date}_{end_date}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    eod_path = save_dir / "eod.csv"

    # If offline mode is enabled, read and return the merged data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(eod_path, parse_dates=['datetime'])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {eod_path}."
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    # <-- Quote data -->
    eod_url = BASE_URL + '/hist/stock/eod'

    while eod_url is not None:
        eod_response = httpx.get(eod_url, params=params, timeout=1000)  # make the request
        eod_response.raise_for_status()  # make sure the request worked

        # read the entire quote_response, and parse it as CSV
        csv_reader = csv.reader(eod_response.text.split("\n"))

        # Convert to pandas dataframe. The first row are the column names
        eod_df = pd.DataFrame(list(csv_reader)[1:], # Skip the first row and use it as columns
                        columns=next(csv.reader(eod_response.text.split("\n"))))  # Use first row as column names

        # Convert to float64
        numeric_columns = eod_df.columns.difference(['date'])
        eod_df[numeric_columns] = eod_df[numeric_columns].astype('float64')

        # Create a datetime column in standard format in 'YYYY-MM-DD HH:MM:SS' (e.g., 2022-09-27 18:00:00)
        # First get time of day in HH:MM:SS format from ms_of_day
        time_of_day = pd.to_datetime(pd.to_numeric(eod_df['ms_of_day']), unit='ms').dt.time

        # Then get the date in YYYY-MM-DD format
        date = pd.to_datetime(eod_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

        # Create the date_time_str column first
        eod_df['date_time_str'] = date + ' ' + time_of_day.astype(str)

        # Then parse it with format='mixed' to handle different time formats
        eod_df['datetime'] = pd.to_datetime(eod_df['date_time_str'], format='mixed')

        # Remove the temporary column and other redundant columns
        eod_df = eod_df.drop(columns=['date_time_str', 'date', 'ms_of_day', 'ms_of_day2'])

        # Make datetime the first column
        eod_df = eod_df.reindex(columns=['datetime'] + list(eod_df.columns.drop('datetime')))

        # Save with mode='a' (append) after the first write
        if eod_url == BASE_URL + '/hist/stock/eod':
            eod_df.to_csv(eod_path, index=False) # Save first batch as .csv.
        else:
            eod_df.to_csv(eod_path, mode='a', header=False, index=False) # Update .csv for subsequent batches

        # check the Next-Page header to see if we have more data
        if 'Next-Page' in eod_response.headers and eod_response.headers['Next-Page'] != "null":
            eod_url = eod_response.headers['Next-Page']
            params = None
            ctx.log(f"Paginating to {eod_url}")
        else:
            eod_url = None

    # Remove last row (NaN)
    eod_df = pd.read_csv(eod_path, parse_dates=['datetime']).dropna()

    # Calculate regular mid prices
    eod_df["mid_price"] = (eod_df["bid"] + eod_df["ask"]) / 2

    # Clean up the entire temp_dir
    if clean_up:
        clean_up_dir(temp_dir)
    else:
        # Save merged data
        eod_df.to_csv(eod_path, index=False)

    return eod_df

if __name__ == "__main__":
    df = load_stock_data(clean_up=False, offline=False)
    print(df.head())
    df = get_stock_data_eod(root="BALL", start_date="20230101", end_date="20231231", clean_up=False, offline=False)
    print(df)
