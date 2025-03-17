import pandas as pd
import numpy as np

def get_historical_volatility(
    stock_data: pd.DataFrame,
    volatility_type: str="period"
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
    # Convert datetime strings to datetime objects
    datetimes = pd.to_datetime(stock_data['datetime'])

    # Verify data is sorted
    assert (datetimes.diff().dropna() > pd.Timedelta(0)).all(), \
        "DataFrame must be sorted by datetime in ascending order"

    # Calculate mid prices
    mid_prices = (stock_data['bid'] + stock_data['ask']) / 2
    unique_dates = stock_data['datetime'].dt.date.unique()

    # Drop NaN values
    unique_dates = unique_dates[~pd.isna(unique_dates)]
    num_trading_days = len(unique_dates)

    # Get intervals per day by counting observations in first full day
    first_day = stock_data[stock_data['datetime'].dt.date == unique_dates[0]]
    intervals_per_day = len(first_day)
    returns_per_day = intervals_per_day - 1

    print(f"Intervals per day are {intervals_per_day} and returns per day are {returns_per_day}")

    # Calculate log returns using mid prices
    log_returns = np.log(mid_prices.values[1:] / mid_prices.values[:-1])

    # Remove overnight returns (last interval of day to first interval of next day)
    valid_return_days = (np.arange(len(log_returns)) + 1) % intervals_per_day != 0

    # Validate the valid_return_days
    n = np.random.randint(0, num_trading_days)
    idx = (returns_per_day)+(intervals_per_day*n)
    assert valid_return_days[idx] == False, f"Did not remove correct overnight returns, expected False at index {idx}, got {valid_return_days[idx]}"

    # Get valid log returns values
    valid_returns = log_returns[valid_return_days]

    # Expected valid returns = (returns per day * number of days)
    expected_valid_returns = (returns_per_day * num_trading_days)

    assert valid_returns.shape[0] == expected_valid_returns, \
        f"Number of valid returns ({valid_returns.shape[0]}) does not match expected ({expected_valid_returns})"

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
