from typing import Tuple, Optional
import numpy as np
import pandas as pd
import os

# Custom modules
from optrade.data.thetadata.listings import get_strikes
from optrade.data.thetadata.stocks import load_stock_data

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

def find_optimal_strike(
    root: str="AAPL",
    start_date: str="20241107",
    exp: str="20241213",
    right: str="C",
    interval_min: int=1,
    moneyness: str="OTM",
    target_band: float=0.05,
    hist_vol: Optional[float]=None,
    volatility_scaled: bool=True,
    volatility_scalar: float=1.0,
    clean_up: bool=False,
    offline: bool=False,
    deterministic: Optional[bool] = True, # TODO: Implement deterministic algorithm or random selection
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
        volatility_type: Type of volatility to use for scaling target_band. Options: "daily", "period", "annualized".
                         For most usecases, "period" type is recommended.
        volatility_scaled: Whether to adjust target_band based on historical volatility
        volatility_scalar: The number of standard deviations to scale the target_band by.
        volatility_window: Proportion of data to use for historical volatility calculation (best practices: use training + validation data portion)
        clean_up (bool): Whether to clean up script temporary data directories
        offline (bool): Whether to use offline data (saved in historical_data directory).
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

    # Get current price and available strikes


    try:
        stock_data = load_stock_data(
            root=root,
            start_date=start_date,
            end_date=start_date,
            interval_min=interval_min,
            clean_up=clean_up,
            offline=offline,
        )
    except:
        # Shift start_date by 1 day if no data is found
        new_start_date = (pd.to_datetime(start_date, format='%Y%m%d') + pd.Timedelta(days=1)).strftime('%Y%m%d')
        stock_data = load_stock_data(
            root=root,
            start_date=new_start_date,
            end_date=new_start_date,
            interval_min=interval_min,
            clean_up=clean_up,
            offline=offline,
        )
        print(f"Stock data not found for {start_date}, shifting to {new_start_date}")

    # Get the average midprice for the day to use as the current price
    current_price = stock_data["mid_price"].mean()

    # Get all available strikes for a given expiration
    strikes = get_strikes(
        root=root,
        exp=exp,
        clean_up=clean_up,
        offline=offline
    ).values.squeeze()

    # Calculate the target strike band
    if moneyness in ["ITM", "OTM"]:

        # print(f"Volatility scalar: {volatility_scalar}. Historical volatility: {hist_vol}")

        # Get historical prices and calculate volatility
        if volatility_scaled:
            # Scale target band based on volatility
            scaled_vol = volatility_scalar * hist_vol # (SD) * (num_SDs)
            strike_band = np.array([current_price - current_price*scaled_vol, current_price + current_price*scaled_vol])
        else:
            strike_band = np.array([current_price - target_band*current_price, current_price + target_band*current_price])

    # Calculate target strike based on moneyness. Find closest strike to target band
    if right == "C":
        if moneyness == "OTM":
            optimal_strike = strikes[np.argmin(np.abs(strikes - strike_band[0]))]
        elif moneyness == "ITM":
            optimal_strike = strikes[np.argmin(np.abs(strikes - strike_band[1]))]
        elif moneyness== "ATM":
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


if __name__ == "__main__":
    optimal_strike = find_optimal_strike(
        root="AAPL",
        start_date="20241107",
        exp="20241206",
        right="C",
        moneyness="ITM",
        target_band=0.10,
        hist_vol=0.20,
        volatility_scaled=True,
        volatility_scalar=2.0,
        clean_up=True,
        offline=False,)

    from rich.console import Console

    console = Console()
    console.log(f"Optimal strike of {optimal_strike} found successfully!", style="bold green")