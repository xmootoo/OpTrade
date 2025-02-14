from typing import Tuple, Optional
from pathlib import Path
import numpy as np
import pandas as pd
import random

# Custom modules
from wsgiref import validate
from optrade.data.thetadata.listings import get_strikes
from optrade.data.thetadata.stocks import get_stock_data

def calculate_historical_volatility(stock_data: pd.DataFrame) -> float:
    """
    Calculate historical volatility using intraday data from regular trading hours (9:30AM-3:59PM).
    Uses mid prices ((bid+ask)/2) for return calculations to avoid bid-ask bounce. Properly accounts
    for overnight return removal between trading days.

    Args:
        stock_data: DataFrame with datetime column in format "YYYY-MM-DD HH:MM:SS"
                   Must be sorted and contain regular intervals during trading hours

    Returns:
        Scaled volatility as a decimal (e.g., 0.20 for 20% volatility) over the period
    """
    # Convert datetime strings to datetime objects
    datetimes = pd.to_datetime(stock_data['datetime'])

    # Verify data is sorted
    assert (datetimes.diff().dropna() > pd.Timedelta(0)).all(), \
        "DataFrame must be sorted by datetime in ascending order"

    # Calculate mid prices
    mid_prices = (stock_data['bid'] + stock_data['ask']) / 2

    # Get unique dates
    unique_dates = stock_data['datetime'].str[:10].unique()

    # Drop NaN values
    unique_dates = unique_dates[~pd.isna(unique_dates)]
    num_trading_days = len(unique_dates)

    # Get intervals per day by counting observations in first full day
    first_day = stock_data[stock_data['datetime'].str[:10] == unique_dates[0]]
    intervals_per_day = len(first_day)
    returns_per_day = intervals_per_day - 1

    # Get number of trading days
    num_rows = stock_data.shape[0] - 1
    num_trading_days = num_rows / intervals_per_day

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

    return period_vol

def find_optimal_strikes(
    root: str = "AAPL",
    start_date: str = "20241107",
    exp: str = "20241213",
    right: str = "C",
    interval_min: int=1,
    target_band: float = 0.05,
    moneyness: str = "OTM",
    volatility_scaled: bool = True,  # Default to True for more adaptive strike selection
    volatility_range: Tuple[str, str] = ("20231107", "20241107"),
    deterministic: Optional[bool] = True,
) -> Tuple[float, str]:
    """
    Finds the optimal strike price for option return forecasting, prioritizing strikes
    that are likely to provide meaningful price movement data.

    Args:
        root: The root symbol of the option
        start_date: The start date in YYYYMMDD format
        exp: The expiration date in YYYYMMDD format
        right: Option type - "C" for call or "P" for put
        target_band: Base percentage distance from current price for strike selection
        moneyness: Desired moneyness - "OTM", "ITM", or "ATM"
        volatility_scaled: Whether to adjust target_band based on historical volatility
        volatility_range: Period for volatility calculation
        deterministic: Use deterministic algorithm for strike selection

    Returns:
        Tuple of (optimal_strike, option_symbol)

    Notes:
        - Uses log returns for volatility calculation as they better capture the
          multiplicative nature of price movements
        - Incorporates historical volatility to adapt strike selection to the
          current market environment
        - For OTM options, ensures strikes aren't too far OTM to maintain sufficient
          price movement for modeling
    """
    script_dir = Path(__file__).parent
    strikes_dir = script_dir.parent / "historical_data/strikes"
    stocks_dir = script_dir.parent / "historical_data/stocks"

    # Get current price and available strikes
    current_price = get_current_price(root, start_date, prices_dir)
    strikes = get_strikes(root, exp, strikes_dir).values.squeeze()

    if volatility_scaled:
        # Get historical prices and calculate volatility
        stock_data = get_stock_data(
            root=root,
            start_date=volatility_range[0],
            end_date=volatility_range[1],
            interval_min=interval_min,
            save_dir=stocks_dir,
        )

        # Calculate historical volatility
        hist_vol = calculate_historical_volatility(stock_data)

        # Scale target band based on volatility
        # Use square root to dampen the effect (prevents too extreme adjustments)
        vol_scalar = np.sqrt(hist_vol)
        adjusted_band = target_band * vol_scalar

        # Cap the adjustment to prevent extreme strike selections
        adjusted_band = min(adjusted_band, 3 * target_band)
    else:
        adjusted_band = target_band

    # Calculate target strike based on moneyness
    if right == "C":
        if moneyness == "OTM":
            # For OTM calls, don't go too far OTM to ensure meaningful price movement
            target_strike = current_price * (1 + adjusted_band)
        elif moneyness == "ITM":
            target_strike = current_price * (1 - adjusted_band)
        else:  # ATM
            target_strike = current_price
    else:  # Put options
        if moneyness == "OTM":
            # For OTM puts, don't go too far OTM
            target_strike = current_price * (1 - adjusted_band)
        elif moneyness == "ITM":
            target_strike = current_price * (1 + adjusted_band)
        else:  # ATM
            target_strike = current_price

    # Find closest available strike, but ensure it exists and has sufficient liquidity
    valid_strikes = strikes[
        (strikes > current_price * 0.5) &  # Remove extremely low strikes
        (strikes < current_price * 2.0)    # Remove extremely high strikes
    ]

    if len(valid_strikes) == 0:
        raise ValueError(f"No valid strikes found for {root} with exp {exp}")

    # Find optimal strike among valid strikes
    optimal_strike = valid_strikes[np.abs(valid_strikes - target_strike).argmin()]

    # # Generate option symbol
    # option_symbol = generate_option_symbol(root, exp, optimal_strike, right)

    # return optimal_strike, option_symbol
    return optimal_strike, None


if __name__ == "__main__":
    # stock_data = get_stock_data(
    #     root="AAPL",
    #     start_date="20231107",
    #     end_date="20241107",
    #     interval_min=5,
    #     save_dir="../historical_data/stocks",
    # )
    path = "/Users/xaviermootoo/Projects/optrade/optrade/data/historical_data/stocks/AAPL/20221107_20231107/merged.csv"

    stock_data = pd.read_csv(path)


    calculate_historical_volatility(stock_data)


# def find_optimal_strikes(
#     root: str="AAPL",
#     start_date: str="20241107",
#     exp: str="20241213",
#     right: str="C",
#     target_band: float=0.05,
#     moneyness: str="OTM",
#     volatility_scaled: bool=False,
#     volatility_range: Tuple[str, str] = ("20231107", "20241107"),
#     deterministic: Optional[bool]=True,
# ) -> Tuple[float, str]:
#     """
#     Finds the optimal strike price for a given root symbol, expiration date, target band, and moneyness.

#     Args:
#         root (str): The root symbol of the option.
#         start_date (str): The start date of the option in the format YYYYMMDD.
#         exp (str): The expiration date of the option in the format YYYYMMDD.
#         right (str): The right of the option, either "C" for call or "P" for put.
#         target_band (float): The target band for the strike price, value between 0 and 1, representing the percentage of the
#                              underlying asset's price at a particular start date.
#         moneyness (str): The moneyness of the option, depending on the right of the option (call or put). Options: "OTM", "ITM", "ATM".
#         volatility_scaled (bool): Whether to select the target band based on a scaled version of underlying asset's price variation
#                                   (i.e. volatility) over a given period.
#         volatility_range (Tuple[str, str]): The period in (YYYYMMDD, YYYYMMDD) format used to calculated the underlying's volatility.
#         volatility_scaler (float): The scaling factor for the volatility.
#         deterministic (Optional[bool]): Whether to use a deterministic algorithm to find the optimal strike price.

#     Returns:
#         Tuple[float, str]: The optimal strike price and the corresponding option symbol.

#     """

#     script_dir = Path(__file__).parent  # Get the directory containing the current script
#     strikes_dir = script_dir.parent / "historical_data/strikes"

#     # Get all available strikes
#     strikes = get_strikes(root, exp, strikes_dir).values.squeeze()
