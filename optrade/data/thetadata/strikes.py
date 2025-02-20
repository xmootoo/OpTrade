from datetime import datetime
from typing import Tuple, Optional
import numpy as np
import pandas as pd

# Custom modules
from optrade.data.thetadata.listings import get_strikes
from optrade.data.thetadata.stocks import get_stock_data
from optrade.src.utils.data.volatility import get_historical_volatility


# TODO: Add cleanup of historical data directories
# TODO: Add assertions and checks that if clean_up then use /temp folder NOT historical data
def find_optimal_strike(
    start_date: datetime ,
    exp: datetime,
    /,
    root: str = "AAPL",
    right: str = "C",
    interval_min: int = 1,
    moneyness: str = "OTM",
    target_band: float = 0.05,
    volatility_type: str = "period",
    volatility_scaled: bool = True,
    volatility_scalar: float = 1.0,
    volatility_window: float = 0.8,
    clean_up: bool = False,
    offline: bool = False,
    deterministic: Optional[
        bool
    ] = True,  # TODO: Implement deterministic algorithm or random selection
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
        clean_up (bool): Whether to clean up temporary data directories
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
    # script_dir = Path(__file__).parent
    # strikes_dir = script_dir.parent / "historical_data/strikes"
    # stocks_dir = script_dir.parent / "historical_data/stocks"

    # Directory setup
    # script_dir = os.path.dirname(os.path.abspath(__file__))
    # strikes_dir = os.path.join(os.path.dirname(script_dir), "historical_data", "strikes")
    # stocks_dir = os.path.join(os.path.dirname(script_dir), "historical_data", "stocks")

    start_date = start_date.strftime("%Y%m%d")
    exp = exp.strftime("%Y%m%d")
    # Get current price and available strikes
    stock_data = get_stock_data(
        root=root,
        start_date=start_date,
        end_date=start_date,
        interval_min=interval_min,
        # save_dir=stocks_dir,
        clean_up=clean_up,
        offline=offline,
    )

    # Get the average midprice for the day to use as the current price
    current_price = stock_data["mid_price"].mean()

    # Get all available strikes for a given expiration
    strikes = get_strikes(
        root=root,
        exp=exp,
        # save_dir=strikes_dir,
        clean_up=clean_up,
        offline=offline,
    ).values.squeeze()

    # Calculate the target strike band
    if moneyness in ["ITM", "OTM"]:

        # Get historical prices and calculate volatility
        if volatility_scaled:

            # Calculate number of days to use for historical volatility
            total_days = (
                pd.to_datetime(exp, format="%Y%m%d")
                - pd.to_datetime(start_date, format="%Y%m%d")
            ).days
            num_vol_days = int(volatility_window * total_days)
            vol_end_date = (
                pd.to_datetime(start_date, format="%Y%m%d")
                + pd.Timedelta(days=num_vol_days)
            ).strftime("%Y%m%d")

            stock_data = get_stock_data(
                root=root,
                start_date=start_date,
                end_date=vol_end_date,
                interval_min=interval_min,
                # save_dir=stocks_dir,
                clean_up=clean_up,
                offline=offline,
            )

            # Calculate historical volatility
            hist_vol = get_historical_volatility(stock_data, volatility_type)

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
                    current_price - target_band * current_price,
                    current_price + target_band * current_price,
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


if __name__ == "__main__":
    optimal_strike = find_optimal_strike(
        root="AAPL",
        start_date="20241107",
        exp="20241206",
        right="C",
        moneyness="ITM",
        target_band=0.10,
        volatility_type="period",
        volatility_scaled=True,
        volatility_scalar=2.0,
        volatility_window=0.8,
        clean_up=False,
        offline=True,
    )

    from rich.console import Console

    console = Console()
    console.log(
        f"Optimal strike of {optimal_strike} found successfully!", style="bold green"
    )
