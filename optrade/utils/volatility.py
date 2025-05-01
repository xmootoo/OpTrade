import pandas as pd
import pandas_market_calendars as mcal
import numpy as np
import warnings
from rich.console import Console

warnings.filterwarnings("ignore", message="The argument 'date_parser' is deprecated")

# Custom modules
from optrade.data.thetadata import load_stock_data


def get_historical_vol(
    stock_data: pd.DataFrame,
    volatility_type: str = "period",
    verbose: bool = False,
) -> float:
    """
    Calculate historical volatility using intraday data from regular trading hours (9:30AM-3:59PM).
    Uses mid prices ((bid+ask)/2) for return calculations to avoid bid-ask bounce. Properly accounts
    for overnight return removal between trading days.

    Args:
        stock_data: DataFrame with datetime column in format "YYYY-MM-DD HH:MM:SS"
                   Must be sorted and contain regular intervals during trading hours
        volatility_type: Type of volatility to calculate. Options: "daily", "period", "annualized".

    Returns:
        Volatility value based on the specified type
    """
    ctx = Console()

    # Convert datetime strings to datetime objects
    datetimes = pd.to_datetime(stock_data["datetime"])

    # Verify data is sorted
    assert (
        datetimes.diff().dropna() > pd.Timedelta(0)
    ).all(), "DataFrame must be sorted by datetime in ascending order"

    # Calculate mid prices
    mid_prices = (stock_data["bid"] + stock_data["ask"]) / 2
    unique_dates = stock_data["datetime"].dt.date.unique()

    # Drop NaN values
    unique_dates = unique_dates[~pd.isna(unique_dates)]
    num_trading_days = len(unique_dates)

    # Get intervals per day by counting observations in first full day
    first_day = stock_data[stock_data["datetime"].dt.date == unique_dates[0]]
    intervals_per_day = len(first_day)
    returns_per_day = intervals_per_day - 1

    if verbose:
        ctx.log(
            f"Intervals per day are {intervals_per_day} and returns per day are {returns_per_day}"
        )

    # Calculate log returns using mid prices
    log_returns = np.log(mid_prices.values[1:] / mid_prices.values[:-1])

    # Remove overnight returns (last interval of day to first interval of next day)
    valid_return_days = (np.arange(len(log_returns)) + 1) % intervals_per_day != 0

    # Validate the valid_return_days
    n = np.random.randint(0, num_trading_days - 1)
    idx = (returns_per_day) + (intervals_per_day * n)
    assert (
        valid_return_days[idx] == False
    ), f"Did not remove correct overnight returns, expected False at index {idx}, got {valid_return_days[idx]}"

    # Get valid log returns values
    valid_returns = log_returns[valid_return_days]

    # Expected valid returns = (returns per day * number of days)
    expected_valid_returns = returns_per_day * num_trading_days

    assert (
        valid_returns.shape[0] == expected_valid_returns
    ), f"Number of valid returns ({valid_returns.shape[0]}) does not match expected ({expected_valid_returns})"

    # Calculate the standard deviation of interval returns
    interval_vol = np.std(valid_returns)

    # Scale up to daily volatility (sqrt of number of returns per day)
    daily_vol = interval_vol * np.sqrt(returns_per_day)

    # Scale to the number of trading days in the dataset
    period_vol = daily_vol * np.sqrt(num_trading_days)

    if volatility_type == "daily":
        return daily_vol
    elif volatility_type == "period":
        return period_vol
    elif volatility_type == "annualized":
        return period_vol * np.sqrt(252 / num_trading_days)
    else:
        raise ValueError(f"Invalid volatility type: {type}")


def get_train_historical_vol(
    root: str,
    start_date: str,
    end_date: str,
    interval_min: int,
    volatility_window: float,
    volatility_type: str,
) -> float:
    """
    Get historical volatility for a stock over a given time period.

    Args:
        root: Underlying stock symbol
        start_date: Start date for the total dataset in YYYYMMDD format
        end_date: End date for the total dataset in YYYYMMDD format
        interval_min: Interval in minutes for the underlying stock data
        volatility_window: Proportion of total days to use for historical volatility calculation
        volatility_type: Type of historical volatility to use. Options: "daily", "period", "annualized".

    Returns:
        Historical volatility value based on the specified type.
    """

    # Calculate number of days to use for historical volatility
    total_days = (
        pd.to_datetime(end_date, format="%Y%m%d")
        - pd.to_datetime(start_date, format="%Y%m%d")
    ).days
    num_vol_days = int(volatility_window * total_days)
    vol_end_date = (
        pd.to_datetime(start_date, format="%Y%m%d") + pd.Timedelta(days=num_vol_days)
    ).strftime("%Y%m%d")

    stock_data = load_stock_data(
        root=root,
        start_date=start_date,
        end_date=end_date,
        interval_min=interval_min,
        clean_up=True,
    )

    # Select only the first num_vol_days for calculating volatility
    stock_data = stock_data.loc[stock_data["datetime"] <= vol_end_date]

    # Calculate historical volatility
    return get_historical_vol(stock_data, volatility_type)



def get_previous_trading_day(date: pd.Timestamp, n_days: int = 1) -> pd.Timestamp:
    """
    Returns the timestamp of the n-th previous NYSE trading day before `date`.

    Args:
        date: A pd.Timestamp
        n_days: How many trading days to go back

    Returns:
        pd.Timestamp of the previous trading day
    """
    nyse = mcal.get_calendar("NYSE")

    # Start with a wide enough window to guarantee coverage
    window_days = n_days + 15  # pad in case of holidays
    start = date - pd.Timedelta(days=window_days)
    end = date

    schedule = nyse.valid_days(
        start_date=start.strftime("%Y-%m-%d"),
        end_date=end.strftime("%Y-%m-%d")
    )

    if len(schedule) < n_days + 1:
        raise ValueError(f"Not enough trading days available to go back {n_days} from {date}")

    return pd.Timestamp(schedule[-(n_days + 1)])

def get_rolling_volatility(
    reference_df: pd.DataFrame,
    root: str,
    interval_min: int = 20,
    return_type: str = "log",
    time_col: str = "datetime",
    dev_mode: bool = False,
) -> pd.Series:
    """
    Computes realized volatility over a lookback window ending at each timestamp in reference_df[time_col],
    with diagnostics for missing data.
    """
    from optrade.data.thetadata import load_stock_data

    reference_df = reference_df.copy()
    reference_df[time_col] = pd.to_datetime(reference_df[time_col])
    reference_times = reference_df[time_col]

    # Get stock data covering enough history
    start_dt = get_previous_trading_day(reference_times.min(), n_days=10)
    end_dt = reference_times.max()
    start_date_str = start_dt.strftime("%Y%m%d")
    end_date_str = end_dt.strftime("%Y%m%d")

    stock_data = load_stock_data(
        root=root,
        start_date=start_date_str,
        end_date=end_date_str,
        interval_min=1,
        dev_mode=dev_mode,
    )

    stock_data["datetime"] = pd.to_datetime(stock_data["datetime"])
    stock_data["mid_price"] = (stock_data["bid"] + stock_data["ask"]) / 2
    stock_data.set_index("datetime", inplace=True)

    out_vols = []
    skipped_timestamps = []
    short_windows = []
    misaligned_timestamps = []

    for t in reference_times:
        end_idx = stock_data.index.searchsorted(t)

        start_idx = end_idx - interval_min
        if start_idx < 0:
            skipped_timestamps.append(t)
            out_vols.append(np.nan)
            continue

        if end_idx > len(stock_data):
            misaligned_timestamps.append(t)
            out_vols.append(np.nan)
            continue

        price_window = stock_data.iloc[start_idx:end_idx]["mid_price"]

        if len(price_window) < 2:
            short_windows.append(t)
            out_vols.append(np.nan)
            continue

        returns = (
            np.log(price_window).diff().dropna()
            if return_type == "log"
            else price_window.pct_change().dropna()
        )

        out_vols.append(np.std(returns))

    rolling_vol = pd.Series(out_vols, index=reference_df.index, name=f"realized_vol_{interval_min}min")
    rolling_vol = rolling_vol.interpolate(method="linear", limit_direction="both")

    # === Diagnostics
    total_nans = rolling_vol.isna().sum()
    if total_nans > 0:
        print(f"[VOL WARNING] {total_nans} NaNs in rolling_vol_{interval_min}min")

    if skipped_timestamps:
        print(f"[VOL DIAG] Skipped {len(skipped_timestamps)} timestamps due to insufficient lookback")
    if short_windows:
        print(f"[VOL DIAG] {len(short_windows)} windows had <2 prices")
    if misaligned_timestamps:
        print(f"[VOL DIAG] {len(misaligned_timestamps)} timestamps were outside stock_data index")

    return rolling_vol



# def get_rolling_volatility(
#     reference_df: pd.DataFrame,
#     root: str,
#     interval_min: int = 20,
#     return_type: str = "log",
#     time_col: str = "datetime",
#     dev_mode: bool = False,
# ) -> pd.Series:
#     """
#     Computes realized volatility over a lookback window ending at each timestamp in reference_df[time_col].

#     Args:
#         reference_df: DataFrame containing timestamps (e.g., 15-min intervals for prediction)
#         root: Stock symbol (e.g., "AAPL")
#         interval_min: Length of lookback window in minutes
#         return_type: "log" or "simple" returns
#         time_col: Name of datetime column in reference_df
#         dev_mode: Pass through to load_stock_data for data loading control

#     Returns:
#         pd.Series of realized volatility values, aligned with reference_df index
#     """
#     from optrade.data.thetadata import load_stock_data

#     reference_df = reference_df.copy()
#     reference_df[time_col] = pd.to_datetime(reference_df[time_col])
#     reference_times = reference_df[time_col]

#     # Determine data range needed for computing volatility
#     start_dt = get_previous_trading_day(reference_times.min(), n_days=10)
#     end_dt = reference_times.max()
#     start_date_str = start_dt.strftime("%Y%m%d")
#     end_date_str = end_dt.strftime("%Y%m%d")

#     # Load high-frequency stock data (1-minute)
#     stock_data = load_stock_data(
#         root=root,
#         start_date=start_date_str,
#         end_date=end_date_str,
#         interval_min=1,
#         dev_mode=dev_mode,
#     )

#     stock_data["datetime"] = pd.to_datetime(stock_data["datetime"])
#     stock_data["mid_price"] = (stock_data["bid"] + stock_data["ask"]) / 2
#     stock_data.set_index("datetime", inplace=True)

#     out_vols = []


#     for t in reference_times:
#         end_idx = stock_data.index.searchsorted(t)
#         start_idx = end_idx - interval_min

#         if start_idx < 0:
#             out_vols.append(np.nan)
#             continue

#         price_window = stock_data.iloc[start_idx:end_idx]["mid_price"]

#         if len(price_window) < 2:
#             out_vols.append(np.nan)
#             continue

#         returns = (
#             np.log(price_window).diff().dropna()
#             if return_type == "log"
#             else price_window.pct_change().dropna()
#         )

#         out_vols.append(np.std(returns))

#     rolling_vol = pd.Series(out_vols, index=reference_df.index, name=f"realized_vol_{interval_min}min")

#     return rolling_vol


if __name__ == "__main__":
    # Example usage
    root = "AAPL"
    start_date = "20230101"
    end_date = "20230131"
    interval_min = 20

    reference_df = load_stock_data(
        root=root,
        start_date=start_date,
        end_date=end_date,
        interval_min=interval_min,
        dev_mode=True
    )

    vol_df = get_rolling_volatility(
        root=root,
        reference_df=reference_df,
        time_col="datetime",
        interval_min=600,
        return_type="log",
    )

    print(vol_df.head())
    print(vol_df.tail())

    # Check if NaNs in vol_df
    if vol_df.isna().any():
        print("NaN values found in vol_df")
    else:
        print("No NaN values in vol_df")

    print(reference_df.head())
    print(reference_df.tail())
