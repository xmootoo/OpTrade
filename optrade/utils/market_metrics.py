import pandas as pd
import numpy as np
import pandas_datareader.data as web
import statsmodels.api as sm
from datetime import datetime
import warnings
from typing import Dict, Any

warnings.filterwarnings("ignore", message="The argument 'date_parser' is deprecated")

from optrade.data.thetadata import load_stock_data_eod
from optrade.data.thetadata import load_stock_data


def get_historical_vol(
    stock_data: pd.DataFrame, volatility_type: str = "period"
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

    print(
        f"Intervals per day are {intervals_per_day} and returns per day are {returns_per_day}"
    )

    # Calculate log returns using mid prices
    log_returns = np.log(mid_prices.values[1:] / mid_prices.values[:-1])

    # Remove overnight returns (last interval of day to first interval of next day)
    valid_return_days = (np.arange(len(log_returns)) + 1) % intervals_per_day != 0

    # Validate the valid_return_days
    n = np.random.randint(0, num_trading_days)
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
) -> pd.DataFrame:
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


def get_fama_french_factors(
    root: str,
    start_date: str,
    end_date: str,
    mode: str = "3_factor",
) -> Dict[str, Any]:
    """
    Calculate Fama-French factor model exposures for a stock over the specified period.

    Args:
        root (str): Root symbol of the underlying security
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format
        mode (str): Mode for the Fama-French model ("3_factor" or "5_factor")

    Returns:
        Dictionary containing the factor betas:
        - market_beta: Market excess return sensitivity
        - size_beta: Small Minus Big (SMB) factor exposure
        - value_beta: High Minus Low (HML) book-to-market factor exposure
        - profitability_beta: Robust Minus Weak (RMW) profitability factor (5-factor only)
        - investment_beta: Conservative Minus Aggressive (CMA) investment factor (5-factor only)
        - r_squared: Proportion of return variation explained by the factors
    """

    # Suppress the date_parser deprecation warning
    warnings.filterwarnings(
        "ignore", message="The argument 'date_parser' is deprecated"
    )

    # Shift the start_date by -1 day to get returns for the current day
    ff_start_date = datetime.strptime(start_date, "%Y%m%d")
    ff_end_date = datetime.strptime(end_date, "%Y%m%d")

    # Get stock data
    stock_data = load_stock_data_eod(
        root=root,
        start_date=start_date,
        end_date=end_date,
        clean_up=True,
        offline=False,
    )

    # Calculate daily returns
    stock_data["returns"] = stock_data["close"].pct_change().dropna()
    stock_data["Date"] = stock_data["datetime"].dt.date

    # Drop NaN
    stock_data = stock_data.dropna()

    # Drop all other columns besides Date and returns
    stock_data = stock_data[["Date", "returns"]]

    # Get Fama-French factor data based on mode
    if mode == "3_factor":
        ff_data = web.DataReader(
            "F-F_Research_Data_Factors_daily",
            "famafrench",
            start=ff_start_date,
            end=ff_end_date,
        )[0]
        factor_columns = ["Mkt-RF", "SMB", "HML"]
    elif mode == "5_factor":  # 5_factor
        ff_data = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily",
            "famafrench",
            start=ff_start_date,
            end=ff_end_date,
        )[0]
        factor_columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose '3_factor' or '5_factor'.")

    # Convert percentages to decimals
    ff_data = ff_data / 100

    # Truncate ff_data to the same date range as stock_data["Date"]
    valid_dates = pd.DatetimeIndex(stock_data["Date"])
    ff_data = ff_data.loc[ff_data.index.intersection(valid_dates)]

    # Reset index to make Date a column
    ff_data_reset = ff_data.reset_index()
    ff_data_reset["Date"] = ff_data_reset["Date"].dt.date

    # Merge stock_data with ff_data on Date
    aligned_data = pd.merge(stock_data, ff_data_reset, on="Date", how="inner")

    # Linear regression
    X = aligned_data[factor_columns]
    X = sm.add_constant(X)
    y = aligned_data["returns"] - aligned_data["RF"]  # Excess return

    # Ensure y is 1-dimensional
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Run regression
    model = sm.OLS(y, X).fit()

    # Prepare results
    result = {
        "market_beta": model.params.get("Mkt-RF", None),
        "size_beta": model.params.get("SMB", None),
        "value_beta": model.params.get("HML", None),
        "r_squared": model.rsquared,
    }

    # Add additional factors for 5-factor model
    if mode == "5_factor":
        result["profitability_beta"] = model.params.get("RMW", None)
        result["investment_beta"] = model.params.get("CMA", None)

    return result


def factor_categorization(
    factors: Dict[str, float], mode: str = "3_factor"
) -> Dict[str, str]:
    """
    Categorize a stock based on its Fama-French factor exposures.

    Args:
        factors (Dict[str, float]): Dictionary containing factor betas for the stock
        mode (str): Mode for the Fama-French model ("3_factor" or "5_factor")

    Returns:
        Dict[str, str]: Dictionary with categorizations for each factor dimension
    """
    categorization = {}

    # Market beta categorization
    if factors["market_beta"] > 1.1:
        categorization["market_beta"] = "high"
    elif factors["market_beta"] < 0.9:
        categorization["market_beta"] = "low"
    else:
        categorization["market_beta"] = "neutral"

    # Size factor categorization
    if factors["size_beta"] > 0.2:
        categorization["size_beta"] = "small_cap"
    elif factors["size_beta"] < -0.2:
        categorization["size_beta"] = "large_cap"
    else:
        categorization["size_beta"] = "neutral"

    # Value factor categorization
    if factors["value_beta"] > 0.2:
        categorization["value_beta"] = "value"
    elif factors["value_beta"] < -0.2:
        categorization["value_beta"] = "growth"
    else:
        categorization["value_beta"] = "neutral"

    # For 5-factor model, add profitability and investment categorizations
    if mode == "5_factor":
        # Profitability factor categorization
        if factors["profitability_beta"] > 0.2:
            categorization["profitability_beta"] = "robust"
        elif factors["profitability_beta"] < -0.2:
            categorization["profitability_beta"] = "weak"
        else:
            categorization["profitability_beta"] = "neutral"

        # Investment factor categorization
        if factors["investment_beta"] > 0.2:
            categorization["investment_beta"] = "conservative"
        elif factors["investment_beta"] < -0.2:
            categorization["investment_beta"] = "aggressive"
        else:
            categorization["investment_beta"] = "neutral"

    return categorization


if __name__ == "__main__":
    # Choose a well-known stock
    symbol = "AAPL"

    # Set test period (1 year)
    start_date = "20230101"  # YYYYMMDD format
    end_date = "20231231"  # YYYYMMDD format

    print(
        f"Testing Fama-French 5-factor exposures for {symbol} from {start_date} to {end_date}"
    )

    # Calculate Fama-French exposures directly using the symbol
    result = get_fama_french_factors(symbol, start_date, end_date, mode="5_factor")

    # Print results
    print("\nFama-French 5-Factor Exposures:")
    print(
        f"Market Beta: {result['market_beta']:.4f}"
        if result["market_beta"] is not None
        else "Market Beta: None"
    )
    print(
        f"Size Beta (SMB): {result['size_beta']:.4f}"
        if result["size_beta"] is not None
        else "Size Beta: None"
    )
    print(
        f"Value Beta (HML): {result['value_beta']:.4f}"
        if result["value_beta"] is not None
        else "Value Beta: None"
    )
    print(
        f"Profitability Beta (RMW): {result['profitability_beta']:.4f}"
        if result["profitability_beta"] is not None
        else "Profitability Beta: None"
    )
    print(
        f"Investment Beta (CMA): {result['investment_beta']:.4f}"
        if result["investment_beta"] is not None
        else "Investment Beta: None"
    )
    print(
        f"R-squared: {result['r_squared']:.4f}"
        if result["r_squared"] is not None
        else "R-squared: None"
    )

    # Categorize the stock based on its factor exposures
    categorization = factor_categorization(result, mode="5_factor")

    # Add this code at the end of your if __name__ == "__main__": block

    # Categorize the stock based on its factor exposures
    categorization = factor_categorization(result, mode="5_factor")

    # Print the categorization results
    print("\nStock Factor Categorization:")
    print(f"Market: {categorization['market_beta']}")
    print(f"Size: {categorization['size_beta']}")
    print(f"Value: {categorization['value_beta']}")
    print(f"Profitability: {categorization['profitability_beta']}")
    print(f"Investment: {categorization['investment_beta']}")
