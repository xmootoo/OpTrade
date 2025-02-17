import httpx
import csv
import pandas as pd
import os

def get_roots(
    sec: str="option",
    save_dir: str='../historical_data/roots',
    clean_up: bool=False,
) -> pd.DataFrame:
    """
    Fetches all root symbols for a given security type.

    Args:
        sec (str): The security type. Options: 'option', 'stock', 'index'.
    """

    BASE_URL = "http://127.0.0.1:25510/v2"
    url = BASE_URL + f'/list/roots/{sec}'
    params = {
      'use_csv': 'true',
    }

    save_dir = os.path.join(save_dir, sec)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'roots.csv')

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
            print(f"Saving to {file_path}")
        else:
            df.to_csv(file_path, mode='a', header=False, index=False)  # Append subsequent batches
            print(f"Appending to {file_path}")

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
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Warning: Could not delete file {file_path}: {e}")

    return df

def get_expirations(
    root: str='AAPL',
    save_dir: str='../historical_data/expirations',
    clean_up: bool=False,
) -> pd.DataFrame:
    """
    Fetch option expiration dates for a given root symbol and save to CSV.

    Args:
        root (str): The root symbol to get expirations for (default: 'AAPL')
        save_dir (str): Directory to save the CSV file (default: current directory)
    """
    BASE_URL = "http://127.0.0.1:25510/v2"
    url = BASE_URL + '/list/expirations'
    params = {'root': root}

    save_dir = os.path.join(save_dir, root)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'expirations.csv')

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
        try:
            os.remove(file_path)
        except OSError as e:
            print(f"Warning: Could not delete file {file_path}: {e}")

    return df

def get_strikes(
    root: str='AAPL',
    exp: str="20250117",
    save_dir: str='../historical_data/strikes',
    clean_up: bool=False,
) -> pd.DataFrame:
    """
    Fetch option strike prices for a given root symbol and expiration, saving to CSV.

    Args:
        root (str): The root symbol to get expirations for (default: 'AAPL')
        exp (str): The expiration date to get strikes for (default: '20250117')
        save_dir (str): Directory to save the CSV file (default: current directory)
    """
    BASE_URL = "http://127.0.0.1:25510/v2"
    url = BASE_URL + '/list/strikes'
    params = {'root': root, 'exp': exp}

    save_dir = os.path.join(save_dir, root, exp)
    os.makedirs(save_dir, exist_ok=True)
    file_path = os.path.join(save_dir, 'strikes.csv')

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
        try:
            os.remove(file_path)
            print(f"Deleted file {file_path}")
        except OSError as e:
            print(f"Warning: Could not delete file {file_path}: {e}")

    return df

if __name__ == '__main__':
    get_strikes(exp="20240419", root="MSFT")
    get_expirations(root="MSFT", clean_up=True)
    get_roots(clean_up=False)
