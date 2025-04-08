import os
from pathlib import Path
import httpx
import csv
import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime
from rich.console import Console

# Custom modules
from optrade.utils.directories import clean_up_dir
from optrade.utils.error_handlers import (
    DataValidationError,
    INCOMPATIBLE_START_DATE,
    INCOMPATIBLE_END_DATE,
    MISSING_DATES,
    UNKNOWN_ERROR,
)

SCRIPT_DIR = Path(__file__).resolve().parent
BASE_URL = "http://127.0.0.1:25510/v2"


def get_roots(
    sec: str = "option",
    save_dir: Optional[str] = None,
    clean_up: bool = False,
    offline: bool = False,
    dev_mode: bool = False,
) -> pd.DataFrame:
    """
    Fetches all root symbols for a given security type.

    Args:
        sec (str): The security type. Options: 'option', 'stock', 'index'.
        save_dir (str): Directory to save the CSV file (default: current directory)
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                            saved in a temp folder and then subsequently deleted before returning the df.
        offline (bool): Whether to work in offline mode, using previously saved data.
        dev_mode (bool): Whether to run in development mode.

    Returns:
        pd.DataFrame: The DataFrame containing the root symbols for the given security type.
    """

    url = BASE_URL + f"/list/roots/{sec}"
    params = {
        "use_csv": "true",
    }

    # Set up directory structure
    working_dir = SCRIPT_DIR if dev_mode else Path.cwd()
    if save_dir is None:
        save_dir = working_dir / "roots"
    else:
        save_dir = Path(save_dir) / "roots"

    # If clean_up is True, save the CSVs in a temp folder, which will be deleted later
    if clean_up and not offline:
        temp_dir = working_dir / "temp" / "roots"
        save_dir = temp_dir

    save_dir = Path(save_dir) / sec
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / "roots.csv"

    # If offline mode is enabled, read and return the merged data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {file_path}."
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    while url is not None:
        response = httpx.get(url, params=params, timeout=10)  # make the request
        response.raise_for_status()  # make sure the request worked

        # Parse CSV and create DataFrame, skipping empty rows
        csv_reader = csv.reader(response.text.split("\n"))
        rows = [row for row in csv_reader if row]  # Skip empty rows

        # Create DataFrame with column name
        df = pd.DataFrame(rows[1:], columns=["root"])  # Skip header row and name column
        print(df.head())

        # Save to CSV with appropriate mode
        if url == BASE_URL + f"/list/roots/{sec}":
            df.to_csv(file_path, index=False)  # Save first batch as .csv
        else:
            df.to_csv(
                file_path, mode="a", header=False, index=False
            )  # Append subsequent batches

        # check the Next-Page header to see if we have more data
        if "Next-Page" in response.headers and response.headers["Next-Page"] != "null":
            url = response.headers["Next-Page"]
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
    root: str,
    save_dir: str = ".",
    clean_up: bool = False,
    offline: bool = False,
    dev_mode: bool = False,
) -> pd.DataFrame:
    """
    Fetch option expiration dates for a given root symbol and save to CSV.

    Args:
        root (str): The root symbol to get expirations for.
        save_dir (str): Directory to save the CSV file (default: current directory)
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                            saved in a temp folder and then subsequently deleted before returning the df.
        offline (bool): Whether to work in offline mode, using previously saved data.
        dev_mode (bool): Whether to run in development mode.

    Returns:
        pd.DataFrame: The DataFrame containing the expiration dates for the given root symbol.
    """
    url = BASE_URL + "/list/expirations"
    params = {"root": root}

    # Set up directory structure
    working_dir = SCRIPT_DIR if dev_mode else Path.cwd()
    if save_dir is None:
        save_dir = working_dir / "roots"
    else:
        save_dir = Path(save_dir) / "roots"

    # If clean_up is True, save the CSVs in a temp folder, which will be deleted later
    if clean_up and not offline:
        temp_dir = working_dir / "temp" / "expirations"
        save_dir = temp_dir

    save_dir = Path(save_dir) / root
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / "expirations.csv"

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
        csv_reader = csv.reader(response.text.split("\n"))
        rows = list(csv_reader)

        # Extract the data rows (skipping header rows) and format as a list of dictionaries
        data_rows = [
            {"date": row[0]}
            for row in rows
            if len(row) > 0 and row[0].strip().isdigit()
        ]

        # Create DataFrame for this batch
        df = pd.DataFrame.from_records(data_rows)

        # Save to CSV with appropriate mode
        if url == BASE_URL + "/list/expirations":
            df.to_csv(file_path, index=False)  # Save first batch as .csv
        else:
            df.to_csv(
                file_path, mode="a", header=False, index=False
            )  # Append subsequent batches

        # Check for next page
        if "Next-Page" in response.headers and response.headers["Next-Page"] != "null":
            url = response.headers["Next-Page"]
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
    root: str,
    exp: str,
    save_dir: str = ".",
    clean_up: bool = False,
    offline: bool = False,
    dev_mode: bool = False,
) -> pd.DataFrame:
    """
    Fetch option strike prices for a given root symbol and expiration, saving to CSV.

    Args:
        root (str): The root symbol to get expirations for.
        exp (str): The expiration date to get strikes for.
        save_dir (str): Directory to save the CSV file (default: current directory)
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                            saved in a temp folder and then subsequently deleted before returning the df.
        offline (bool): Whether to work in offline mode, using previously saved data.
        dev_mode (bool): Whether to run in development mode.

    Returns:
        pd.DataFrame: The DataFrame containing the strike prices for the given root and expiration.
    """
    url = BASE_URL + "/list/strikes"
    params = {"root": root, "exp": exp}

    # Set up directory structure
    working_dir = SCRIPT_DIR if dev_mode else Path.cwd()
    if save_dir is None:
        save_dir = working_dir / "roots"
    else:
        save_dir = Path(save_dir) / "roots"

    # If clean_up is True, save the CSVs in a temp folder, which will be deleted later
    if clean_up and not offline:
        temp_dir = working_dir / "temp" / "strikes"
        save_dir = temp_dir

    save_dir = Path(save_dir) / root / exp
    save_dir.mkdir(parents=True, exist_ok=True)
    file_path = save_dir / "strikes.csv"

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
        csv_reader = csv.reader(response.text.split("\n"))
        rows = list(csv_reader)

        # Extract the data rows (skipping header rows) and format as a list of dictionaries
        data_rows = [
            {"strike": row[0]}
            for row in rows
            if len(row) > 0 and row[0].strip().isdigit()
        ]

        # Create DataFrame for this batch
        df = pd.DataFrame.from_records(data_rows)

        # Divide values by 1000 to convert to dollars and convert to float
        df["strike"] = df["strike"].astype(float) / 1000

        # Save to CSV with appropriate mode
        if url == BASE_URL + "/list/strikes":
            df.to_csv(file_path, index=False)  # Save first batch as .csv
        else:
            df.to_csv(
                file_path, mode="a", header=False, index=False
            )  # Append subsequent batches

        # Check for next page
        if "Next-Page" in response.headers and response.headers["Next-Page"] != "null":
            url = response.headers["Next-Page"]
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


def find_optimal_exp(
    root: str,
    start_date: str,
    target_tte: int,
    tte_tolerance: Tuple[int, int],
    clean_up: bool = True,
    dev_mode: bool = False,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Returns the closest valid TTE to target_tte within tolerance range and its expiration date.

    Args:
        root: The root symbol of the underlying security
        start_date: The start date in YYYYMMDD format
        target_tte: Desired days to expiry (e.g., 30)
        tte_tolerance: (min_tte, max_tte) acceptable range
        save_dir: Directory to save the data files.
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                            saved in a temp folder and then subsequently deleted before returning the df.

    Returns:
        Tuple[str, int]: A tuple containing the optimal expiration date (in YYYYMMDD format) and
                         the corresponding time-to-expiration in days.
    """
    min_tte, max_tte = tte_tolerance

    # Get expirations and convert to list of strings
    expirations = get_expirations(
        root=root,
        clean_up=clean_up,
        dev_mode=dev_mode,
    ).values.squeeze()

    # Convert start_date to datetime
    start_dt = datetime.strptime(start_date, "%Y%m%d")

    # Calculate TTEs for each expiration
    valid_pairs = []
    for exp in expirations:
        # Convert expiration to datetime
        exp_dt = datetime.strptime(str(exp), "%Y%m%d")

        # Calculate days to expiry
        tte = (exp_dt - start_dt).days

        # Check if TTE is within tolerance range
        if min_tte <= tte <= max_tte:
            valid_pairs.append((tte, exp))

    # If we found valid TTEs, return the one closest to target
    if valid_pairs:
        # Sort by distance to target TTE
        optimal_tte, optimal_exp = min(
            valid_pairs, key=lambda x: abs(x[0] - target_tte)
        )
        return str(optimal_exp), optimal_tte
    else:
        raise ValueError(
            f"No valid TTE found within tolerance range {tte_tolerance}. "
            "Please try a wider tolerance band."
        )


def load_stock_data(
    root: str,
    start_date: str,
    end_date: str,
    interval_min: int = 1,
    save_dir: Optional[str] = None,
    clean_up: bool = False,
    offline: bool = False,
    dev_mode: bool = False,
) -> pd.DataFrame:
    """
    Gets historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    stocks across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).

    .. note::
       Data from OHLC ends at 15:59:00, while quote data ends at 16:00:00, so for simplicity we
       remove all rows with 16:00:00 in datetime from quote data, before merging quote and OHLC data.

    Args:
        root (str): The root symbol of the underlying security.
        start_date (str): The start date of the data in YYYYMMDD format.
        end_date (str): The end date of the data in YYYYMMDD format.
        interval_min (int): The interval in minutes between data points.
        save_dir (str): The directory to save the data.
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                            saved in a temp folder and then subsequently deleted before returning the df.
        offline (bool): Whether to work in offline mode, using previously saved data.
        dev_mode (bool): Whether to run in development mode.

    Returns:
        pd.DataFrame: The merged NBBO quote and OHLCVC data.
    """

    BASE_URL = "http://127.0.0.1:25510/v2"  # all endpoints use this URL base

    intervals = str(interval_min * 60000)  # Convert minutes to milliseconds

    ctx = Console()

    params = {
        "root": root,
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true",
        "ivl": intervals,
        "venue": "utp_cta",  # Merged UTP & CTA data
    }

    # Set up directory structure
    working_dir = SCRIPT_DIR if dev_mode else Path.cwd()

    if save_dir is None:
        save_dir = working_dir / "historical_data" / "stocks"
    else:
        save_dir = Path(save_dir) / "stocks"

    if clean_up and not offline:
        temp_dir = working_dir / "temp" / "stocks"
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
            return pd.read_csv(merged_file_path, parse_dates=["datetime"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {merged_file_path}. "
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    # <-- Quote data -->
    quote_url = BASE_URL + "/hist/stock/quote"

    while quote_url is not None:
        quote_response = httpx.get(
            quote_url, params=params, timeout=1000
        )  # make the request
        quote_response.raise_for_status()  # make sure the request worked

        # read the entire quote_response, and parse it as CSV
        csv_reader = csv.reader(quote_response.text.split("\n"))

        # Convert to pandas dataframe. The first row are the column names
        quote_df = pd.DataFrame(
            list(csv_reader)[1:],  # Skip the first row and use it as columns
            columns=next(csv.reader(quote_response.text.split("\n"))),
        )  # Use first row as column names

        # Convert to float64
        numeric_columns = quote_df.columns.difference(["date"])
        quote_df[numeric_columns] = quote_df[numeric_columns].astype("float64")

        # Create a datetime column in standard format in 'YYYY-MM-DD HH:MM:SS' (e.g., 2022-09-27 18:00:00)
        # First get time of day in HH:MM:SS format from ms_of_day
        time_of_day = pd.to_datetime(
            pd.to_numeric(quote_df["ms_of_day"]), unit="ms"
        ).dt.time

        # Then get the date in YYYY-MM-DD format
        date = pd.to_datetime(quote_df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

        # Then concatenate the date and time_of_day
        quote_df["datetime"] = pd.to_datetime(date + " " + time_of_day.astype(str))

        # Remove date and ms_of_day (redundant information)
        quote_df = quote_df.drop(columns=["date", "ms_of_day"])

        # Remove any rows with 16:00:00 in datetime, as OHLC data ends at 15:59:00
        quote_df = quote_df[
            quote_df["datetime"].dt.time != pd.to_datetime("16:00:00").time()
        ]

        # Make datetime the first column
        quote_df = quote_df.reindex(
            columns=["datetime"] + list(quote_df.columns.drop("datetime"))
        )

        # Save with mode='a' (append) after the first write
        if quote_url == BASE_URL + "/hist/stock/quote":
            quote_df.to_csv(quote_file_path, index=False)  # Save first batch as .csv.
        else:
            quote_df.to_csv(
                quote_file_path, mode="a", header=False, index=False
            )  # Update .csv for subsequent batches

        # check the Next-Page header to see if we have more data
        if (
            "Next-Page" in quote_response.headers
            and quote_response.headers["Next-Page"] != "null"
        ):
            quote_url = quote_response.headers["Next-Page"]
            params = None
            ctx.log(f"Paginating to {quote_url}")
        else:
            quote_url = None

    # Redefine params for OHLC data (as it will set to None)
    params = {
        "root": root,
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true",
        "ivl": intervals,
        "venue": "utp_cta",  # Merged UTP & CTA data
    }

    # <-- OHLC data -->
    ohlc_url = BASE_URL + "/hist/stock/ohlc"

    while ohlc_url is not None:
        ohlc_response = httpx.get(ohlc_url, params=params, timeout=1000)
        ohlc_response.raise_for_status()

        # read the entire response, and parse it as CSV
        csv_reader = csv.reader(ohlc_response.text.split("\n"))
        ohlc_df = pd.DataFrame(
            list(csv_reader)[1:],
            columns=next(csv.reader(ohlc_response.text.split("\n"))),
        )

        # Convert to float64
        numeric_columns = ohlc_df.columns.difference(["date"])
        ohlc_df[numeric_columns] = ohlc_df[numeric_columns].astype("float64")

        # Create datetime column
        time_of_day = pd.to_datetime(
            pd.to_numeric(ohlc_df["ms_of_day"]), unit="ms"
        ).dt.time
        date = pd.to_datetime(ohlc_df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        ohlc_df["datetime"] = pd.to_datetime(date + " " + time_of_day.astype(str))

        # Remove redundant columns
        ohlc_df = ohlc_df.drop(columns=["date", "ms_of_day"])

        # Save with mode='a' (append) after the first write
        if ohlc_url == BASE_URL + "/hist/stock/ohlc":
            ohlc_df.to_csv(ohlc_file_path, index=False)  # Save first batch as .csv
        else:
            ohlc_df.to_csv(
                ohlc_file_path, mode="a", header=False, index=False
            )  # Update .csv for subsequent batches

        # Check for next page
        if (
            "Next-Page" in ohlc_response.headers
            and ohlc_response.headers["Next-Page"] != "null"
        ):
            ohlc_url = ohlc_response.headers["Next-Page"]
            params = None
            ctx.log(f"Paginating to {ohlc_url}")
        else:
            ohlc_url = None

    # Read CSVs with datetime parsing
    quote_df = pd.read_csv(quote_file_path, parse_dates=["datetime"])
    ohlc_df = pd.read_csv(ohlc_file_path, parse_dates=["datetime"])

    # Merge on datetime to ensure proper alignment
    merged_df = pd.merge(quote_df, ohlc_df, on="datetime", how="inner")

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
    root: str,
    start_date: str,
    end_date: str,
    save_dir: Optional[str] = None,
    clean_up: bool = False,
    offline: bool = False,
    dev_mode: bool = False,
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
        dev_mode (bool): Whether to run in development mode.

    Returns:
        pd.DataFrame: The merged quote-level and OHLC data.
    """

    BASE_URL = "http://127.0.0.1:25510/v2"  # all endpoints use this URL base

    ctx = Console()

    params = {
        "root": root,
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true",
    }

    # Set up directory structure
    working_dir = SCRIPT_DIR if dev_mode else Path.cwd()
    if save_dir is None:
        save_dir = working_dir / "historical_data" / "stocks"
    else:
        save_dir = Path(save_dir) / "stocks"

    if clean_up and not offline:
        temp_dir = working_dir / "temp" / "stocks"
        save_dir = Path(temp_dir)

    save_dir = save_dir / root / f"{start_date}_{end_date}"
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    eod_path = save_dir / "eod.csv"

    # If offline mode is enabled, read and return the merged data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(eod_path, parse_dates=["datetime"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {eod_path}."
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    # <-- Quote data -->
    eod_url = BASE_URL + "/hist/stock/eod"

    while eod_url is not None:
        eod_response = httpx.get(
            eod_url, params=params, timeout=1000
        )  # make the request
        eod_response.raise_for_status()  # make sure the request worked

        # read the entire quote_response, and parse it as CSV
        csv_reader = csv.reader(eod_response.text.split("\n"))

        # Convert to pandas dataframe. The first row are the column names
        eod_df = pd.DataFrame(
            list(csv_reader)[1:],  # Skip the first row and use it as columns
            columns=next(csv.reader(eod_response.text.split("\n"))),
        )  # Use first row as column names

        # Convert to float64
        numeric_columns = eod_df.columns.difference(["date"])
        eod_df[numeric_columns] = eod_df[numeric_columns].astype("float64")

        # Create a datetime column in standard format in 'YYYY-MM-DD HH:MM:SS' (e.g., 2022-09-27 18:00:00)
        # First get time of day in HH:MM:SS format from ms_of_day
        time_of_day = pd.to_datetime(
            pd.to_numeric(eod_df["ms_of_day"]), unit="ms"
        ).dt.time

        # Then get the date in YYYY-MM-DD format
        date = pd.to_datetime(eod_df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

        # Create the date_time_str column first
        eod_df["date_time_str"] = date + " " + time_of_day.astype(str)

        # Then parse it with format='mixed' to handle different time formats
        eod_df["datetime"] = pd.to_datetime(eod_df["date_time_str"], format="mixed")

        # Remove the temporary column and other redundant columns
        eod_df = eod_df.drop(
            columns=["date_time_str", "date", "ms_of_day", "ms_of_day2"]
        )

        # Make datetime the first column
        eod_df = eod_df.reindex(
            columns=["datetime"] + list(eod_df.columns.drop("datetime"))
        )

        # Save with mode='a' (append) after the first write
        if eod_url == BASE_URL + "/hist/stock/eod":
            eod_df.to_csv(eod_path, index=False)  # Save first batch as .csv.
        else:
            eod_df.to_csv(
                eod_path, mode="a", header=False, index=False
            )  # Update .csv for subsequent batches

        # check the Next-Page header to see if we have more data
        if (
            "Next-Page" in eod_response.headers
            and eod_response.headers["Next-Page"] != "null"
        ):
            eod_url = eod_response.headers["Next-Page"]
            params = None
            ctx.log(f"Paginating to {eod_url}")
        else:
            eod_url = None

    # Remove last row (NaN)
    eod_df = pd.read_csv(eod_path, parse_dates=["datetime"]).dropna()

    # Calculate regular mid prices
    eod_df["mid_price"] = (eod_df["bid"] + eod_df["ask"]) / 2

    # Clean up the entire temp_dir
    if clean_up:
        clean_up_dir(temp_dir)
    else:
        # Save merged data
        eod_df.to_csv(eod_path, index=False)

    return eod_df


def find_optimal_strike(
    root: str,
    start_date: str,
    exp: str,
    right: str,
    interval_min: int,
    moneyness: str,
    strike_band: Optional[float] = 0.05,
    volatility_scaled: bool = False,
    hist_vol: Optional[float] = None,
    volatility_scalar: Optional[float] = 1.0,
    clean_up: bool = False,
    offline: bool = False,
    deterministic: Optional[
        bool
    ] = True,  # TODO: Implement deterministic algorithm or random selection
    dev_mode: bool = False,
) -> Tuple[float, str]:
    """
    Finds the optimal strike price for option return forecasting, prioritizing strikes
    that are likely to provide meaningful price movement data.

    Args:
        root: The root symbol of the option
        start_date: The start date in YYYYMMDD format
        exp: The expiration date in YYYYMMDD format
        right: Option type - "C" for call or "P" for put
        interval_min: The interval in minutes between data points (the resolution of the data).
        moneyness: Desired moneyness - "OTM", "ITM", or "ATM"
        strike_band: Base percentage distance from current price for strike selection
        volatility_scaled: Whether to adjust strike_band based on historical volatility
        hist_vol: Historical volatility to use for scaling strike_band (required if volatility_scaled=True).
        volatility_scalar: The number of standard deviations to scale the strike_band by.
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                            saved in a temp folder and then subsequently deleted before returning the df.
        offline (bool): Whether to work in offline mode, using previously saved data.
        deterministic: Use deterministic algorithm for strike selection (True by default, stochastic mode not yet implemented).
        dev_mode: Whether to run in development mode (True) or production mode (False).

    Returns:
        float: The optimal strike price for option return forecasting based on the specified criteria.
    """

    # Get current price and available strikes
    try:
        stock_data = load_stock_data(
            root=root,
            start_date=start_date,
            end_date=start_date,
            interval_min=interval_min,
            clean_up=clean_up,
            offline=offline,
            dev_mode=dev_mode,
        )
    except:
        # Shift start_date by 1 day if no data is found
        new_start_date = (
            pd.to_datetime(start_date, format="%Y%m%d") + pd.Timedelta(days=1)
        ).strftime("%Y%m%d")
        stock_data = load_stock_data(
            root=root,
            start_date=new_start_date,
            end_date=new_start_date,
            interval_min=interval_min,
            clean_up=clean_up,
            offline=offline,
            dev_mode=dev_mode,
        )
        print(f"Stock data not found for {start_date}, shifting to {new_start_date}")

    # Get the average midprice for the day to use as the current price
    current_price = stock_data["mid_price"].mean()

    # Get all available strikes for a given expiration
    strikes = get_strikes(
        root=root, exp=exp, clean_up=clean_up, offline=offline
    ).values.squeeze()

    # Calculate the target strike band
    if moneyness in ["ITM", "OTM"]:

        # print(f"Volatility scalar: {volatility_scalar}. Historical volatility: {hist_vol}")

        # Get historical prices and calculate volatility
        if volatility_scaled:
            assert hist_vol is not None, "Historical volatility must be provided"
            assert volatility_scalar is not None, "Volatility scalar must be provided"

            # Scale target band based on volatility
            scaled_vol = volatility_scalar * hist_vol  # (SD) * (num_SDs)
            strike_band = np.array(
                [
                    current_price - current_price * scaled_vol,
                    current_price + current_price * scaled_vol,
                ]
            )
        else:
            strike_band = np.array(
                [
                    current_price - strike_band * current_price,
                    current_price + strike_band * current_price,
                ]
            )

    # Calculate target strike based on moneyness. Find closest strike to target band
    if right == "C":
        if moneyness == "OTM":
            optimal_strike = strikes[np.argmin(np.abs(strikes - strike_band[0]))]
        elif moneyness == "ITM":
            optimal_strike = strikes[np.argmin(np.abs(strikes - strike_band[1]))]
        elif moneyness == "ATM":
            optimal_strike = strikes[np.argmin(np.abs(strikes - current_price))]
        else:
            raise ValueError(f"Invalid moneyness: {moneyness}")
    else:  # Put options
        if moneyness == "OTM":
            optimal_strike = strikes[np.argmin(np.abs(strikes - strike_band[1]))]
        elif moneyness == "ITM":
            optimal_strike = strikes[np.argmin(np.abs(strikes - strike_band[0]))]
        elif moneyness == "ATM":
            optimal_strike = strikes[np.argmin(np.abs(strikes - current_price))]
        else:
            raise ValueError(f"Invalid moneyness: {moneyness}")

    return optimal_strike


def load_option_data(
    root: str,
    start_date: str,
    end_date: str,
    exp: str,
    strike: float,
    interval_min: int,
    right: str,
    save_dir: Optional[str] = None,
    clean_up: bool = False,
    offline: bool = False,
    count_ohlc_zeros: bool = False,
    dev_mode: bool = False,
) -> pd.DataFrame:
    """
    Gets historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    options across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).

    .. note::
       Data from OHLC ends at 15:59:00, while quote data ends at 16:00:00, so for simplicity we
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
        offline (bool): Whether to work in offline mode, using previously saved data.
        count_ohlc_zeros (bool): Whether to count the proportion of zero values in OHLC transactions data.
        dev_mode (bool): Whether to run in development mode.

    Returns:
        pd.DataFrame: Merged DataFrame containing quote-level (NBBO) and OHLC data for the specified option.
    """

    ctx = Console()

    BASE_URL = "http://127.0.0.1:25510/v2"  # all endpoints use this URL base

    intervals = str(interval_min * 60000)  # Convert minutes to milliseconds
    strike = str(int(strike * 1000))  # Converts dollars to 1/10th cents

    params = {
        "root": root,
        "exp": exp,
        "strike": strike,
        "right": right,
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true",
        "ivl": intervals,
    }

    # Set up directory structure
    working_dir = SCRIPT_DIR if dev_mode else Path.cwd()
    if save_dir is None:
        save_dir = working_dir / "historical_data" / "options"
    else:
        save_dir = Path(save_dir) / "options"

    if clean_up and not offline:
        temp_dir = working_dir / "temp" / "options"
        save_dir = Path(temp_dir)

    # Set up directory structure
    save_dir = (
        save_dir
        / root
        / right
        / f"{start_date}_{end_date}"
        / f"{strike}strike_{exp}exp"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    quote_file_path = save_dir / "quote.csv"
    ohlc_file_path = save_dir / "ohlc.csv"
    merged_file_path = save_dir / "merged.csv"

    # If offline mode is enabled, read and return the merged data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(merged_file_path, parse_dates=["datetime"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {merged_file_path}. "
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    # <-- Quote data -->
    quote_url = BASE_URL + "/hist/option/quote"

    while quote_url is not None:
        quote_response = httpx.get(
            quote_url, params=params, timeout=1000
        )  # make the request
        quote_response.raise_for_status()  # make sure the request worked

        # read the entire quote_response, and parse it as CSV
        csv_reader = csv.reader(quote_response.text.split("\n"))

        # Convert to pandas dataframe (eliminate [0,1,2,...] row of indices from ThetaData)
        quote_df = pd.DataFrame(
            list(csv_reader)[1:],  # Skip the first row and use it as columns
            columns=next(csv.reader(quote_response.text.split("\n"))),
        )  # Use first row as column names

        # Convert to float64
        numeric_columns = quote_df.columns.difference(["date"])
        quote_df[numeric_columns] = quote_df[numeric_columns].astype("float64")

        # Create a datetime column in standard format in 'YYYY-MM-DD HH:MM:SS' (e.g., 2022-09-27 18:00:00)
        # First get time of day in HH:MM:SS format from ms_of_day
        time_of_day = pd.to_datetime(
            pd.to_numeric(quote_df["ms_of_day"]), unit="ms"
        ).dt.time

        # Then get the date in YYYY-MM-DD format
        date = pd.to_datetime(quote_df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")

        # Then concatenate the date and time_of_day
        quote_df["datetime"] = pd.to_datetime(date + " " + time_of_day.astype(str))

        # Remove date and ms_of_day (redundant information)
        quote_df = quote_df.drop(columns=["date", "ms_of_day"])

        # Remove any rows with 16:00:00 in datetime, as OHLC data ends at 15:59:00
        quote_df = quote_df[
            quote_df["datetime"].dt.time != pd.to_datetime("16:00:00").time()
        ]

        # Make datetime the first column
        quote_df = quote_df[
            ["datetime"] + [col for col in quote_df.columns if col != "datetime"]
        ]

        # Save with mode='a' (append) after the first write
        if quote_url == BASE_URL + "/hist/option/quote":
            quote_df.to_csv(quote_file_path, index=False)  # Save first batch as .csv
        else:
            quote_df.to_csv(
                quote_file_path, mode="a", header=False, index=False
            )  # Update .csv for subsequent batches

        # check the Next-Page header to see if we have more data
        if (
            "Next-Page" in quote_response.headers
            and quote_response.headers["Next-Page"] != "null"
        ):
            quote_url = quote_response.headers["Next-Page"]
            params = None
            ctx.log(f"Paginating to {quote_url}")
        else:
            quote_url = None

    # Redefine params for OHLC data (as it will set to None)
    params = {
        "root": root,
        "exp": exp,
        "strike": strike,
        "right": right,
        "start_date": start_date,
        "end_date": end_date,
        "use_csv": "true",
        "ivl": intervals,
    }

    # <-- OHLC data -->
    ohlc_url = BASE_URL + "/hist/option/ohlc"

    while ohlc_url is not None:
        ohlc_response = httpx.get(ohlc_url, params=params, timeout=1000)
        ohlc_response.raise_for_status()

        # read the entire response, and parse it as CSV
        csv_reader = csv.reader(ohlc_response.text.split("\n"))
        ohlc_df = pd.DataFrame(
            list(csv_reader)[1:],
            columns=next(csv.reader(ohlc_response.text.split("\n"))),
        )

        # Convert to float64
        numeric_columns = ohlc_df.columns.difference(["date"])
        ohlc_df[numeric_columns] = ohlc_df[numeric_columns].astype("float64")

        # Create datetime column
        time_of_day = pd.to_datetime(
            pd.to_numeric(ohlc_df["ms_of_day"]), unit="ms"
        ).dt.time
        date = pd.to_datetime(ohlc_df["date"], format="%Y%m%d").dt.strftime("%Y-%m-%d")
        ohlc_df["datetime"] = pd.to_datetime(date + " " + time_of_day.astype(str))

        # Remove redundant columns
        ohlc_df = ohlc_df.drop(columns=["date", "ms_of_day"])

        # Save with mode='a' (append) after the first write
        if ohlc_url == BASE_URL + "/hist/option/ohlc":
            ohlc_df.to_csv(ohlc_file_path, index=False)  # Save first batch as .csv
        else:
            ohlc_df.to_csv(
                ohlc_file_path, mode="a", header=False, index=False
            )  # Update .csv for subsequent batches

        # Check for next page
        if (
            "Next-Page" in ohlc_response.headers
            and ohlc_response.headers["Next-Page"] != "null"
        ):
            ohlc_url = ohlc_response.headers["Next-Page"]
            params = None
            ctx.log(f"Paginating to {ohlc_url}")
        else:
            ohlc_url = None

    # Read CSVs with datetime parsing
    quote_df = pd.read_csv(quote_file_path, parse_dates=["datetime"])
    ohlc_df = pd.read_csv(ohlc_file_path, parse_dates=["datetime"])

    # Merge on datetime to ensure proper alignment
    merged_df = pd.merge(quote_df, ohlc_df, on="datetime", how="inner")

    # Remove any duplicate columns that might exist in both dataframes
    duplicate_cols = merged_df.columns.duplicated()
    merged_df = merged_df.loc[:, ~duplicate_cols]

    # Remove last row (NaN)
    merged_df = merged_df.dropna()

    # Calculate regular mid prices
    merged_df["mid_price"] = (merged_df["bid"] + merged_df["ask"]) / 2

    # Convert datetime to pandas datetime if it isn't already
    if not pd.api.types.is_datetime64_ns_dtype(merged_df["datetime"]):
        merged_df["datetime"] = pd.to_datetime(merged_df["datetime"])

    # Handle market open midprices (9:30 AM) separately using OHLC data
    market_open_mask = (
        (merged_df["datetime"].dt.hour == 9)
        & (merged_df["datetime"].dt.minute == 30)
        & (merged_df["datetime"].dt.second == 0)
    )

    merged_df.loc[market_open_mask, "mid_price"] = merged_df.loc[
        market_open_mask, ["open", "close"]
    ].mean(axis=1)

    # Catch any remaining zeroes at market open if open and close are also zero
    # by backfilling with the next non-zero value
    zero_mask = merged_df["mid_price"] == 0
    zero_indices = zero_mask[zero_mask].index

    # For each zero, take the next non-zero value
    for idx in zero_indices:
        # Get next index
        next_idx = idx + 1
        # Use the next value to fill the zero
        merged_df.loc[idx, "mid_price"] = merged_df.loc[next_idx, "mid_price"]

    # Verify fix
    remaining_zeros = merged_df[merged_df["mid_price"] == 0]
    if not remaining_zeros.empty:
        ctx.log("Still have zeros at:", remaining_zeros.index)

    if count_ohlc_zeros:
        # Check proportion of zeros for open and close (do not backfill/interpolate these)
        zero_mask_open = merged_df["open"] == 0
        ctx.log(
            f"Proportion of zeros in the open: {zero_mask_open.sum() / len(merged_df):.2f}"
        )

        zero_mask_close = merged_df["close"] == 0
        ctx.log(
            f"Proportion of zeros in the close: {zero_mask_close.sum() / len(merged_df):.2f}"
        )

        # ctx.log proportion of zeros to total dates
        zero_mask_high = merged_df["high"] == 0
        ctx.log(
            f"Proportion of zeros in the high: {zero_mask_high.sum() / len(merged_df):.2f}"
        )

        zero_mask_low = merged_df["low"] == 0
        ctx.log(
            f"Proportion of zeros in the low: {zero_mask_low.sum() / len(merged_df):.2f}"
        )

    # Clean up the entire temp_dir
    if clean_up:
        clean_up_dir(temp_dir)
    else:
        # Save merged data
        merged_df.to_csv(merged_file_path, index=False)

    return merged_df


def load_all_data(
    root: str,
    start_date: str,
    exp: str,
    interval_min: int,
    right: str,
    strike: float,
    save_dir: Optional[str] = None,
    clean_up: bool = False,
    offline: bool = False,
    warning: bool = False,
    dev_mode: bool = False,
) -> pd.DataFrame:
    """
    Gets historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    combined stocks and options across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).

    .. note::
       Data from OHLC ends at 15:59:00, while quote data ends at 16:00:00, so for simplicity we
       remove all rows with 16:00:00 in datetime from quote data, before merging quote and OHLC data.

    Args:
        root (str): The root symbol of the underlying security.
        start_date (str): The start date of the data in YYYYMMDD format.
        exp (str): The expiration date of the option in YYYYMMDD format.
        interval_min (int): The interval in minutes between data points.
        right (str): The type of option, either 'C' for call or 'P' for put.
        strike (float): The strike price of the option in dollars.
        save_dir (str): The directory to save the data.
        clean_up (bool): Whether to clean up the CSV files after merging. If True, the CSV files are
                         saved in a temp folder and then subsequently deleted before returning the df.
        offline (bool): Whether to use offline (already saved) data instead of calling ThetaData API directly (default: False).
        dev_mode (bool): Whether to run in development mode.

    Returns:
        DataFrame: The combined quote-level and OHLC data for an option and the underlying,
    """

    # Set up directory structure
    working_dir = SCRIPT_DIR if dev_mode else Path.cwd()
    if save_dir is None:
        save_dir = working_dir / "historical_data" / "combined"
        options_dir = working_dir / "historical_data" / "options"
        stocks_dir = working_dir / "historical_data" / "stocks"
    else:
        save_dir = Path(save_dir) / "combined"
        options_dir = Path(save_dir) / "options"
        stocks_dir = Path(save_dir) / "stocks"

    if clean_up and not offline:
        temp_dir = working_dir / "temp" / "combined"
        save_dir = Path(temp_dir)

    # Set up directory structure
    save_dir = (
        save_dir / root / right / f"{start_date}_{exp}" / f"{strike}strike_{exp}exp"
    )

    save_dir.mkdir(parents=True, exist_ok=True)

    # Define file paths
    combined_file_path = save_dir / "combined.csv"

    # If offline mode is enabled, read and return the combined data. This assumes data is already saved.
    if offline:
        try:
            return pd.read_csv(combined_file_path, parse_dates=["datetime"])
        except FileNotFoundError:
            raise FileNotFoundError(
                f"No offline data found at {combined_file_path}."
                "Please run with offline=False and clean_up=False first to download the required data."
            )

    option_df = load_option_data(
        root=root,
        start_date=start_date,
        end_date=exp,
        exp=exp,
        strike=strike,
        interval_min=interval_min,
        right=right,
        save_dir=options_dir,
        clean_up=clean_up,
        offline=offline,
        dev_mode=dev_mode,
    )

    stock_df = load_stock_data(
        root=root,
        start_date=start_date,
        end_date=exp,
        interval_min=interval_min,
        save_dir=stocks_dir,
        clean_up=clean_up,
        offline=offline,
        dev_mode=dev_mode,
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
        "count",
    ]

    option_columns = {col: "option_" + col for col in base_columns if col != "datetime"}
    option_df = option_df.rename(columns=option_columns)

    # Same for stocks
    stock_columns = {col: "stock_" + col for col in base_columns if col != "datetime"}
    stock_df = stock_df.rename(columns=stock_columns)

    # Check lengths of option_df and stock_df match exactly
    if len(option_df) != len(stock_df):

        real_start_date = option_df["datetime"].iloc[0]
        stock_filtered_start = stock_df[
            stock_df["datetime"] >= real_start_date
        ].reset_index(drop=True)

        real_end_date = option_df["datetime"].iloc[-1]
        stock_filtered = stock_filtered_start[
            stock_filtered_start["datetime"] <= real_end_date
        ].reset_index(drop=True)

        # Verify if option_df starts earlier than stock_df
        if (
            option_df["datetime"].to_list()
            == stock_filtered_start["datetime"].to_list()
        ):
            real_start_date = real_start_date.strftime("%Y%m%d")
            queried_start_date = stock_df["datetime"].iloc[0].strftime("%Y%m%d")
            raise DataValidationError(
                message=f"Option data queried on start_date={queried_start_date}, but the contract doesn't start until start_date={real_start_date}.",
                error_code=INCOMPATIBLE_START_DATE,
                real_start_date=real_start_date,
                warning=warning,
            )

        # Otherwise verify if option_df starts earlier AND later than stock_df
        elif option_df["datetime"].to_list() == stock_filtered["datetime"].to_list():
            real_start_date = real_start_date.strftime("%Y%m%d")
            real_end_date = real_end_date.strftime("%Y%m%d")
            queried_start_date = stock_df["datetime"].iloc[0].strftime("%Y%m%d")
            queried_end_date = stock_df["datetime"].iloc[-1].strftime("%Y%m%d")
            raise DataValidationError(
                message=f"Option data queried on start_date={queried_end_date} with exp={queried_end_date}, but the contract starts on start_date={real_start_date} and ends on {real_end_date}.",
                error_code=INCOMPATIBLE_END_DATE,
                real_start_date=real_start_date,
                real_end_date=real_end_date,
                warning=warning,
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
                message=error_message
                + "This discrepancy is likely due to anomalous market events.",
                error_code=MISSING_DATES,
                warning=warning,
            )
        else:
            raise DataValidationError(
                message=f"Unknown error. Filtering stock data by start_date={real_start_date} and end_date={real_end_date} did not resolve the issue, and neither is option dates a subset of stock dates.",
                error_code=UNKNOWN_ERROR,
                warning=warning,
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
    # Test: get_strikes, get_expirations, get_roots
    clean_up = True
    offline = False

    df1 = get_strikes(
        exp="20240419", root="MSFT", clean_up=clean_up, offline=offline, dev_mode=True
    )
    df2 = get_expirations(
        root="MSFT", clean_up=clean_up, offline=offline, dev_mode=True
    )
    df3 = get_roots(clean_up=clean_up, offline=offline, dev_mode=True)

    print(df1.head())
    print(df2.head())
    print(df3.head())

    # Test: find_optimal_exp
    try:
        optimal_exp, optimal_tte = find_optimal_exp(
            root="AAPL", start_date="20241107", tte_tolerance=(20, 40), target_tte=37
        )
        print(f"Found valid TTE: {optimal_tte}")
        print(f"Expiration date: {optimal_exp}")
    except ValueError as e:
        print(f"Error: {e}")

    # Test: find_optimal_strike
    optimal_strike = find_optimal_strike(
        root="AAPL",
        start_date="20241107",
        exp="20241206",
        right="C",
        interval_min=1,
        moneyness="ITM",
        strike_band=0.10,
        hist_vol=0.20,
        volatility_scaled=True,
        volatility_scalar=2.0,
        clean_up=True,
        offline=False,
        dev_mode=True,
    )

    from rich.console import Console

    console = Console()
    console.log(
        f"Optimal strike of {optimal_strike} found successfully!", style="bold green"
    )

    # Test: load_stock_data
    df = load_stock_data(
        root="AAPL",
        start_date="20230101",
        end_date="20231231",
        clean_up=False,
        offline=False,
        dev_mode=True,
    )
    print(df.head())
    df = load_stock_data_eod(
        root="BALL",
        start_date="20230101",
        end_date="20231231",
        clean_up=False,
        offline=False,
        dev_mode=True,
    )
    print(df)

    # Test: load_option_data
    df = load_option_data(
        root="MSFT",
        start_date="20240105",
        end_date="20240205",
        right="C",
        exp="20240419",
        strike=400,
        interval_min=1,
        clean_up=False,
        offline=False,
        dev_mode=True,
    )
    print(df.head())

    # Test: load_all_data
    df = load_all_data(
        root="AAPL",
        start_date="20241107",
        exp="20250117",
        interval_min=1,
        right="C",
        strike=225,
        clean_up=False,
        offline=False,
        dev_mode=True,
    )
    print(df.head())
