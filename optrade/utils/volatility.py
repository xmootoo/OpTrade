import pandas as pd
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
