import httpx
import csv
import pandas as pd
import os

def get_expirations(
    root: str = 'AAPL',
    base_dir: str = '../historical_data/listings'
) -> None:
    """
    Fetch option expiration dates for a given root symbol and save to CSV.

    Args:
        root (str): The root symbol to get expirations for (default: 'AAPL')
        base_dir (str): Directory to save the CSV file (default: current directory)
    """
    BASE_URL = "http://127.0.0.1:25510/v2"
    url = BASE_URL + '/list/expirations'
    params = {'root': root}

    base_dir = os.path.join(base_dir, root)

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
        expiry_file_path = os.path.join(base_dir, 'expirations.csv')
        if url == BASE_URL + '/list/expirations':
            df.to_csv(expiry_file_path, index=False)  # Save first batch as .csv
        else:
            df.to_csv(expiry_file_path, mode='a', header=False, index=False)  # Append subsequent batches

        # Check for next page
        if 'Next-Page' in response.headers and response.headers['Next-Page'] != "null":
            url = response.headers['Next-Page']
            params = None
            print(f"Paginating to {url}")
        else:
            url = None

    # Load the CSV into pandas dataframe
    df = pd.read_csv(os.path.join(base_dir, 'expirations.csv'))

    # Display the results
    print(df.head())

    return df





if __name__ == '__main__':
    get_expirations()
