import pandas as pd
import numpy as np

from typing import Optional

from optrade.src.preprocessing.features.datetime_features import get_datetime_features
from optrade.src.preprocessing.features.tte_features import get_tte_features

def get_features(
    df: pd.DataFrame,
    core_feats: list,
    tte_feats: list,
    datetime_feats: list,
    strike: Optional[int]=None,
) -> pd.DataFrame:

    """
    Selects features from a DataFrame based on a list of feature names (i.e. columns).

    Args:
        df (pd.DataFrame): The DataFrame containing the features.
        core_feats (list): List of core features to select.
        tte_feats (list): List of Time to Expiration (TTE) features to select.
        datetime_feats (list): List of datetime features to select.

    Core features options:
        - "datetime"
        - f"{asset}_returns"
        - f"{asset}_mid_price"
        - f"{asset}_bid_size"
        - f"{asset}_bid_exchange"
        - f"{asset}_bid"
        - f"{asset}_bid_condition"
        - f"{asset}_ask_size"
        - f"{asset}_ask_exchange"
        - f"{asset}_ask"
        - f"{asset}_ask_condition"
        - f"{asset}_open"
        - f"{asset}_high"
        - f"{asset}_low"
        - f"{asset}_close"
        - f"{asset}_volume"
        - f"{asset}_count"
        - "distance_to_strike"
    where "asset" is either "option" or "stock".

    TTE features options:
        - "linear"
        - "linear"
        - "inverse"
        - "sqrt"
        - "inverse_sqrt"
        - "exp_decay"

    Datetime features options:
        - "minuteofday"
        - "sin_timeofday"
        - "cos_timeofday"
        - "dayofweek"
        - "sin_dayofweek"
        - "cos_dayofweek"
    """

    # Generate additional features
    df = get_datetime_features(df=df, feats=datetime_feats)
    df = get_tte_features(df=df, feats=tte_feats)

    if "option_returns" in core_feats:
        # Calculate returns and add to dataframe
        prices = df["option_mid_price"].to_numpy()
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        df["option_returns"] = returns

        # Drop the first market open (return=0)
        first_time = df["datetime"].iloc[0].time()
        if first_time.hour == 9 and first_time.minute == 30:
            df = df.iloc[1:].reset_index(drop=True)

    if "distance_to_strike" in core_feats:
        # Calculate distance to strike and add to dataframe
        distance = (float(strike) - df["stock_mid_price"]).to_numpy()
        df["distance_to_strike"] = distance

    # Select features
    tte_index = ["tte_" + tte_feats[i] for i in range(len(tte_feats))]
    datetime_index = ["dt_" + datetime_feats[i] for i in range(len(datetime_feats))]
    selected_feats = core_feats + tte_index + datetime_index

    return df[selected_feats]


if __name__ == "__main__":
    from optrade.data.thetadata.get_data import get_data
    from optrade.data.thetadata.contracts import Contract
    from rich.console import Console
    console = Console()

    contract = Contract()
    df = get_data(
        contract=contract,
        clean_up=True,
        offline=False
    )

    # TTE features
    tte_feats = ["sqrt", "exp_decay"]

    # Datetime features
    datetime_feats = ["sin_timeofday", "cos_timeofday", "dayofweek"]

    # Select features
    core_feats = [
        "option_returns",
        "distance_to_strike",
        "option_mid_price",
        "option_bid_size",
        "option_bid",
        "option_ask_size",
        "option_close",
        "option_volume",
        "option_count",
        "stock_mid_price",
        "stock_bid_size",
        "stock_bid",
        "stock_ask_size",
        "stock_ask",
        "stock_volume",
        "stock_count",
    ]

    df = get_features(
        df=df,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        strike=contract.strike,
    )

    print(df.columns == core_feats + [f"tte_{f}" for f in tte_feats] + [f"dt_{f}" for f in datetime_feats])

    print(df.head())
    print(df.to_numpy().shape)
