import httpx  # install via pip install httpx
import csv
import pandas as pd
import os

def get_historical_data(
    root: str="AAPL",
    start_date: str="20241107",
    end_date: str="20241108",
    exp: str="20250117",
    strike: int=225,
    interval_min: int=1,
    right: str="C",
    save_dir: str="../historical_data",
) -> None:

    """
    Gets historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    options across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).

    *NOTE: Data from OHLC ends at 15:59:00, while quote data ends at 16:00:00, so for simplicity we
    remove all rows with 16:00:00 in datetime from quote data, before merging quote and OHLC data.

    Args:
        root (str): The root symbol of the underlying security.
        start_date (str): The start date of the data in YYYYMMDD format.
        end_date (str): The end date of the data in YYYYMMDD format.
        exp (str): The expiration date of the option in YYYYMMDD format.
        strike (int): The strike price of the option in dollars.
        interval_min (int): The interval in minutes between data points.
        right (str): The type of option, either 'C' for call or 'P' for put.

    Saves a 3 CSV files into the save_dir: quote-level data, OHLC data, and merged data (concatenation of both
    quote-level and OHLC data). Data is stored by default in data/historical_data/ directory.
    """

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

    # Create subfolder in save_dir with root symbol
    save_dir = os.path.join(save_dir, root)
    os.makedirs(save_dir, exist_ok=True)

    # <-- Quote data -->
    quote_url = BASE_URL + '/hist/option/quote'

    while quote_url is not None:
        quote_response = httpx.get(quote_url, params=params, timeout=10)  # make the request
        quote_response.raise_for_status()  # make sure the request worked

        # read the entire quote_response, and parse it as CSV
        csv_reader = csv.reader(quote_response.text.split("\n"))

        # Convert to pandas dataframe (eliminate [0,1,2,...] row of indices from ThetaData)
        quote_df = pd.DataFrame(list(csv_reader)[1:],  # Skip the first row and use it as columns
                        columns=next(csv.reader(quote_response.text.split("\n"))))  # Use first row as column names

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

        # Save to csv
        quote_df.to_csv(os.path.join(save_dir,f'{root}_{start_date}_{end_date}_quote.csv'), index=False)

        # check the Next-Page header to see if we have more data
        if 'Next-Page' in quote_response.headers and quote_response.headers['Next-Page'] != "null":
            quote_url = quote_response.headers['Next-Page']
            params = None
        else:
            quote_url = None


    # <-- OHLC data -->
    ohlc_url = BASE_URL + '/hist/option/ohlc'

    while ohlc_url is not None:
        ohlc_response = httpx.get(ohlc_url, params=params, timeout=10)  # make the request
        ohlc_response.raise_for_status()  # make sure the request worked

        # read the entire response, and parse it as CSV
        csv_reader = csv.reader(ohlc_response.text.split("\n"))

        ohlc_df = pd.DataFrame(list(csv_reader)[1:],  # Skip the first row and use it as columns
                        columns=next(csv.reader(ohlc_response.text.split("\n"))))  # Use first row as column names

        # Save to csv
        ohlc_df.to_csv(os.path.join(save_dir,f'{root}_{start_date}_{end_date}_ohlc.csv'), index=False)

        # Create a datetime column in standard format in 'YYYY-MM-DD HH:MM:SS' (e.g., 2022-09-27 18:00:00)
        # First get time of day in HH:MM:SS format from ms_of_day
        time_of_day = pd.to_datetime(pd.to_numeric(ohlc_df['ms_of_day']), unit='ms').dt.time

        # Then get the date in YYYY-MM-DD format
        date = pd.to_datetime(ohlc_df['date'], format='%Y%m%d').dt.strftime('%Y-%m-%d')

        # Then concatenate the date and time_of_day
        ohlc_df['datetime'] = pd.to_datetime(date + ' ' + time_of_day.astype(str))

        # Remove duplicate columns already in quote_df (e.g., date, ms_of_day)
        ohlc_df = ohlc_df.drop(columns=['date', 'ms_of_day'])

        # Save to csv
        ohlc_df.to_csv(os.path.join(save_dir,f'{root}_{start_date}_{end_date}_ohlc.csv'), index=False)

        # Drop datetime from ohlc_df
        ohlc_df = ohlc_df.drop(columns=['datetime'])

        # Load the quote_df from location and concatenate with ohlc_df
        quote_df = pd.read_csv(os.path.join(save_dir,f'{root}_{start_date}_{end_date}_quote.csv'))

        # Merge quote_df and ohlc_df on datetime
        merged_df = pd.concat([quote_df, ohlc_df], axis=1)

        # Save to csv (override the previous file)
        merged_df.to_csv(os.path.join(save_dir,f'{root}_{start_date}_{end_date}_total.csv'), index=False)

        # check the Next-Page header to see if we have more data
        if 'Next-Page' in ohlc_response.headers and ohlc_response.headers['Next-Page'] != "null":
            ohlc_url = ohlc_response.headers['Next-Page']
            params = None
        else:
            ohlc_url = None


if __name__ == "__main__":
    # TODO: Add pydantic args here
    get_historical_data()
