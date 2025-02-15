import httpx
import csv
import pandas as pd
import os
from typing import Optional, Tuple

def get_option_data(
    root: str="AAPL",
    start_date: str="20241107",
    end_date: str="20241107",
    tte: Optional[int]=-1,
    exp: Optional[str]="20250117",
    strike: int=225,
    interval_min: int=1,
    right: str="C",
    save_dir: str="../historical_data/options",
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
        tte (Optional[int]): The time to expiration of the option in days.
        strike (int): The strike price of the option in dollars.
        interval_min (int): The interval in minutes between data points.
        right (str): The type of option, either 'C' for call or 'P' for put.

    Saves a 3 CSV files into the save_dir: quote-level data, OHLC data, and merged data (concatenation of both
    quote-level and OHLC data). Data is stored by default in data/historical_data/ directory.
    """

    BASE_URL = "http://127.0.0.1:25510/v2"  # all endpoints use this URL base

    intervals = str(interval_min * 60000) # Convert minutes to milliseconds
    strike = str(strike * 1000) # Converts dollars to 1/10th cents

    # If Time to Expiration is provided, calculate expiration date by adding tte days to start_date
    if tte != -1:
        exp = pd.to_datetime(start_date, format='%Y%m%d') + pd.DateOffset(days=tte)
        exp = exp.strftime('%Y%m%d')

        # Set end_date = start_date + tte - 1 (in days)
        end_date = (pd.to_datetime(start_date, format='%Y%m%d') +
                    pd.DateOffset(days=tte-1)).strftime('%Y%m%d')

    print(f"Start date: {start_date}, End date: {end_date}, Expiration date: {exp}")

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

    # Create subfolder in save_dir with root symbol
    base_dir = os.path.join(save_dir, root, right, f"{start_date}_{end_date}", f"{strike}strike_{exp}exp")
    os.makedirs(base_dir, exist_ok=True)

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
        quote_file_path = os.path.join(base_dir, 'quote.csv')
        if quote_url == BASE_URL + '/hist/option/quote':
            quote_df.to_csv(quote_file_path, index=False) # Save first batch as .csv
        else:
            quote_df.to_csv(quote_file_path, mode='a', header=False, index=False) # Update .csv for subsequent batches

        # check the Next-Page header to see if we have more data
        if 'Next-Page' in quote_response.headers and quote_response.headers['Next-Page'] != "null":
            quote_url = quote_response.headers['Next-Page']
            params = None
            print(f"Paginating to {quote_url}")
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
        ohlc_file_path = os.path.join(base_dir, 'ohlc.csv')
        if ohlc_url == BASE_URL + '/hist/option/ohlc':
            ohlc_df.to_csv(ohlc_file_path, index=False) # Save first batch as .csv
        else:
            ohlc_df.to_csv(ohlc_file_path, mode='a', header=False, index=False) # Update .csv for subsequent batches

        # Check for next page
        if 'Next-Page' in ohlc_response.headers and ohlc_response.headers['Next-Page'] != "null":
            ohlc_url = ohlc_response.headers['Next-Page']
            params = None
            print(f"Paginating to {ohlc_url}")
        else:
            ohlc_url = None

    # After both loops are complete, merge the data
    # Read CSVs with datetime parsing
    quote_df = pd.read_csv(os.path.join(base_dir, 'quote.csv'), parse_dates=['datetime'])
    ohlc_df = pd.read_csv(os.path.join(base_dir, 'ohlc.csv'), parse_dates=['datetime'])

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

    # Save merged data
    merged_df.to_csv(os.path.join(base_dir, 'merged.csv'), index=False)

    return merged_df


if __name__ == "__main__":

    for i in range(15, 35):
        try:
            df = get_option_data(tte=i)
            print(f"Worked for TTE: {i}")
            print(df.head())
        except:
            pass
