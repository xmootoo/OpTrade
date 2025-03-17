import httpx
import csv
import pandas as pd
import os

# Custom modules
from optrade.utils.data.clean_up import clean_up_dir
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_URL = "http://127.0.0.1:25510/v2"

def get_roots(
    sec: str="option",
    save_dir: str='../historical_data/roots',
    clean_up: bool=False,
    offline: bool=False,
) -> pd.DataFrame:
    """
    Fetches all root symbols for a given security type.

    Args:
        sec (str): The security type. Options: 'option', 'stock', 'index'.
    """

    url = BASE_URL + f'/list/roots/{sec}'
    params = {
      'use_csv': 'true',
    }

    # If clean_up is True, save the CSVs in a temp folder, which will be deleted later
    if clean_up and not offline:
        temp_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "temp", "roots")
        save_dir = temp_dir

    save_dir = os.path.join(save_dir, sec)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'roots.csv')

    # If offline mode is enabled, read and return the merged data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {file_path}. "
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    while url is not None:
        response = httpx.get(url, params=params, timeout=10)  # make the request
        response.raise_for_status()  # make sure the request worked

        # Parse CSV and create DataFrame, skipping empty rows
        csv_reader = csv.reader(response.text.split("\n"))
        rows = [row for row in csv_reader if row]  # Skip empty rows

        # Create DataFrame with column name
        df = pd.DataFrame(rows[1:], columns=['root'])  # Skip header row and name column
        print(df.head())

        # Save to CSV with appropriate mode
        if url == BASE_URL + f'/list/roots/{sec}':
            df.to_csv(file_path, index=False)  # Save first batch as .csv
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)  # Append subsequent batches

        # check the Next-Page header to see if we have more data
        if 'Next-Page' in response.headers and response.headers['Next-Page'] != "null":
            url = response.headers['Next-Page']
            params = None
        else:
            url = None

    # Load the CSV into pandas dataframe
    df = pd.read_csv(file_path)

    # Clean up the file if requested
    if clean_up:
        clean_up_dir(temp_dir)

    return df

def get_expirations(
    root: str='AAPL',
    save_dir: str='../historical_data/expirations',
    clean_up: bool=False,
    offline: bool=False,
) -> pd.DataFrame:
    """
    Fetch option expiration dates for a given root symbol and save to CSV.

    Args:
        root (str): The root symbol to get expirations for (default: 'AAPL')
        save_dir (str): Directory to save the CSV file (default: current directory)
    """
    url = BASE_URL + '/list/expirations'
    params = {'root': root}

    # If clean_up is True, save the CSVs in a temp folder, which will be deleted later
    if clean_up and not offline:
        temp_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "temp", "expirations")
        save_dir = temp_dir

    save_dir = os.path.join(save_dir, root)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'expirations.csv')

    # If offline mode is enabled, read and return the merged data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {file_path}. "
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    while url is not None:
        # Make the request
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()

        # Parse the response as CSV
        csv_reader = csv.reader(response.text.split('\n'))
        rows = list(csv_reader)

        # Extract the data rows (skipping header rows) and format as a list of dictionaries
        data_rows = [{'date': row[0]} for row in rows if len(row) > 0 and row[0].strip().isdigit()]

        # Create DataFrame for this batch
        df = pd.DataFrame.from_records(data_rows)

        # Save to CSV with appropriate mode
        if url == BASE_URL + '/list/expirations':
            df.to_csv(file_path, index=False)  # Save first batch as .csv
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)  # Append subsequent batches

        # Check for next page
        if 'Next-Page' in response.headers and response.headers['Next-Page'] != "null":
            url = response.headers['Next-Page']
            params = None
            print(f"Paginating to {url}")
        else:
            url = None

    # Load the CSV into pandas dataframe
    df = pd.read_csv(file_path)

    # Clean up the file if requested
    if clean_up:
        clean_up_dir(temp_dir)

    return df

def get_strikes(
    root: str='AAPL',
    exp: str="20250117",
    save_dir: str='../historical_data/strikes',
    clean_up: bool=False,
    offline: bool=False,
) -> pd.DataFrame:
    """
    Fetch option strike prices for a given root symbol and expiration, saving to CSV.

    Args:
        root (str): The root symbol to get expirations for (default: 'AAPL')
        exp (str): The expiration date to get strikes for (default: '20250117')
        save_dir (str): Directory to save the CSV file (default: current directory)
    """
    url = BASE_URL + '/list/strikes'
    params = {'root': root, 'exp': exp}

    # If clean_up is True, save the CSVs in a temp folder, which will be deleted later
    if clean_up and not offline:
        temp_dir = os.path.join(os.path.dirname(SCRIPT_DIR), "temp", "strikes")
        save_dir = temp_dir

    save_dir = os.path.join(save_dir, root, exp)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'strikes.csv')

    # If offline mode is enabled, read and return the merged data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {file_path}. "
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    while url is not None:
        # Make the request
        response = httpx.get(url, params=params, timeout=10)
        response.raise_for_status()

        # Parse the response as CSV
        csv_reader = csv.reader(response.text.split('\n'))
        rows = list(csv_reader)

        # Extract the data rows (skipping header rows) and format as a list of dictionaries
        data_rows = [{'strike': row[0]} for row in rows if len(row) > 0 and row[0].strip().isdigit()]

        # Create DataFrame for this batch
        df = pd.DataFrame.from_records(data_rows)

        # Divide values by 1000 to convert to dollars and convert to float
        df['strike'] = df['strike'].astype(float) / 1000

        # Save to CSV with appropriate mode
        if url == BASE_URL + '/list/strikes':
            df.to_csv(file_path, index=False)  # Save first batch as .csv
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)  # Append subsequent batches

        # Check for next page
        if 'Next-Page' in response.headers and response.headers['Next-Page'] != "null":
            url = response.headers['Next-Page']
            params = None
            print(f"Paginating to {url}")
        else:
            url = None

    # Load the CSV into pandas dataframe
    df = pd.read_csv(file_path)

    # Clean up the file if requested
    if clean_up:
        clean_up_dir(temp_dir)

    return df

if __name__ == '__main__':
    clean_up = False
    offline = True

    df1 = get_strikes(exp="20240419", root="MSFT", clean_up=clean_up, offline=offline)
    df2 = get_expirations(root="MSFT", clean_up=clean_up, offline=offline)
    df3 = get_roots(clean_up=clean_up, offline=offline)

    print(df1.head())
    print(df2.head())
    print(df3.head())