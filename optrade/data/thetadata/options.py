import httpx
import csv
import pandas as pd
import os
from typing import Optional
from rich.console import Console
from pathlib import Path

# Custom modules
from optrade.utils.data.clean_up import clean_up_dir

SCRIPT_DIR = Path(__file__).resolve().parent

def load_option_data(
    root: str="AAPL",
    start_date: str="20241107",
    end_date: str="20241107",
    exp: Optional[str]="20250117",
    strike: int=225,
    interval_min: int=1,
    right: str="C",
    save_dir: Optional[str]=None,
    clean_up: bool=False,
    offline: bool=False,
    count_ohlc_zeros: bool=False,
) -> pd.DataFrame:

    """
    Gets historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    options across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).

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

    Saves a 3 CSV files into the save_dir: quote-level data, OHLC data, and merged data (concatenation of both
    quote-level and OHLC data). Data is stored by default in data/historical_data/ directory.
    """

    ctx = Console()

    BASE_URL = "http://127.0.0.1:25510/v2"  # all endpoints use this URL base

    intervals = str(interval_min * 60000) # Convert minutes to milliseconds
    strike = str(strike * 1000) # Converts dollars to 1/10th cents

    params = {
        'root': root,
        'exp': exp,
        'strike': strike,
        'right': right,
        'start_date': start_date,
        'end_date': end_date,
        'use_csv': 'true',
        'ivl': intervals,
    }

    if save_dir is None:
            save_dir = SCRIPT_DIR.parent / "historical_data" / "options"
    else:
        save_dir = Path(save_dir) / "options"

    if clean_up and not offline:
        temp_dir = SCRIPT_DIR.parent / "temp" / "options"
        save_dir = Path(temp_dir)

    # Set up directory structure
    save_dir = (
        save_dir /
        root /
        right /
        f"{start_date}_{end_date}" /
        f"{strike}strike_{exp}exp"
    )
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
    quote_url = BASE_URL + '/hist/option/quote'

    while quote_url is not None:
        quote_response = httpx.get(quote_url, params=params, timeout=1000)  # make the request
        quote_response.raise_for_status()  # make sure the request worked

        # read the entire quote_response, and parse it as CSV
        csv_reader = csv.reader(quote_response.text.split("\n"))

        # Convert to pandas dataframe (eliminate [0,1,2,...] row of indices from ThetaData)
        quote_df = pd.DataFrame(list(csv_reader)[1:],  # Skip the first row and use it as columns
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
        quote_df = quote_df[['datetime'] + [col for col in quote_df.columns if col != 'datetime']]

        # Save with mode='a' (append) after the first write
        if quote_url == BASE_URL + '/hist/option/quote':
            quote_df.to_csv(quote_file_path, index=False) # Save first batch as .csv
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
        'exp': exp,
        'strike': strike,
        'right': right,
        'start_date': start_date,
        'end_date': end_date,
        'use_csv': 'true',
        'ivl': intervals,
    }

    # <-- OHLC data -->
    ohlc_url = BASE_URL + '/hist/option/ohlc'

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
        if ohlc_url == BASE_URL + '/hist/option/ohlc':
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

    # Convert datetime to pandas datetime if it isn't already
    if not pd.api.types.is_datetime64_ns_dtype(merged_df['datetime']):
        merged_df['datetime'] = pd.to_datetime(merged_df['datetime'])

    # Handle market open midprices (9:30 AM) separately using OHLC data
    market_open_mask = ((merged_df['datetime'].dt.hour == 9) &
                       (merged_df['datetime'].dt.minute == 30) &
                       (merged_df['datetime'].dt.second == 0))

    merged_df.loc[market_open_mask, 'mid_price'] = (
        merged_df.loc[market_open_mask, ['open', 'close']]
        .mean(axis=1)
    )

    # Catch any remaining zeroes at market open if open and close are also zero
    # by backfilling with the next non-zero value
    zero_mask = merged_df['mid_price'] == 0
    zero_indices = zero_mask[zero_mask].index

    # For each zero, take the next non-zero value
    for idx in zero_indices:
        # Get next index
        next_idx = idx + 1
        # Use the next value to fill the zero
        merged_df.loc[idx, 'mid_price'] = merged_df.loc[next_idx, 'mid_price']

    # Verify fix
    remaining_zeros = merged_df[merged_df['mid_price'] == 0]
    if not remaining_zeros.empty:
        ctx.log("Still have zeros at:", remaining_zeros.index)

    if count_ohlc_zeros:
        # Check proportion of zeros for open and close (do not backfill/interpolate these)
        zero_mask_open = merged_df['open'] == 0
        ctx.log(f"Proportion of zeros in the open: {zero_mask_open.sum() / len(merged_df):.2f}")

        zero_mask_close = merged_df['close'] == 0
        ctx.log(f"Proportion of zeros in the close: {zero_mask_close.sum() / len(merged_df):.2f}")

        # ctx.log proportion of zeros to total dates
        zero_mask_high = merged_df['high'] == 0
        ctx.log(f"Proportion of zeros in the high: {zero_mask_high.sum() / len(merged_df):.2f}")

        zero_mask_low = merged_df['low'] == 0
        ctx.log(f"Proportion of zeros in the low: {zero_mask_low.sum() / len(merged_df):.2f}")

    # Clean up the entire temp_dir
    if clean_up:
        clean_up_dir(temp_dir)
    else:
        # Save merged data
        merged_df.to_csv(merged_file_path, index=False)

    return merged_df

def fill_open_zeros(group):
    if group.iloc[0]['mid_price'] == 0:
        first_nonzero = group[
            (group['mid_price'] != 0) &
            (group['datetime'].dt.time <= pd.Timestamp('09:35').time())
        ]['mid_price'].iloc[0]
        group.loc[group['mid_price'] == 0, 'mid_price'] = first_nonzero
    return group

if __name__ == "__main__":

    df = load_option_data(
        root="MSFT",
        start_date="20240105",
        end_date="20240205",
        right="C",
        exp="20240419",
        strike=400,
        interval_min=1,
        clean_up=False,
        offline=False
    )
    print(df.head())
