import pandas as pd
import numpy as np

from typing import Optional, List
from optrade.data.features.datetime_features import get_datetime_features
from optrade.data.features.tte_features import get_tte_features

def transform_features(
    df: pd.DataFrame,
    core_feats: List[str],
    tte_feats: List[str],
    datetime_feats: List[str],
    strike: Optional[int]=None,
    exp: Optional[str]=None,
) -> pd.DataFrame:

    """
    Selects features from a DataFrame based on a list of feature names (i.e. columns).

    Args:
        df (pd.DataFrame): The DataFrame containing the features.
        core_feats (list): List of core features to select.
        tte_feats (list): List of Time to Expiration (TTE) features to select.
        datetime_feats (list): List of datetime features to select.

    Core feature options (subset of NBBO and OHLCVC):
        - "datetime"
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

    Core feature options (advanced):
        # TODO: Add these features
        - f"{asset}_returns"
        - f"{asset}_lob_imbalance"
        - f"{asset}_quote_spread"
        - "moneyness" (log(S/K))
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
        - "minute_of_day"
        - "sin_minute_of_day"
        - "cos_minute_of_day"
        - "day_of_week"
        - "sin_day_of_week"
        - "cos_day_of_week"
        - "hour_of_week"
        - "sin_hour_of_week"
        - "cos_hour_of_week"
    """

    # Generate additional features
    df = get_datetime_features(df=df, feats=datetime_feats)
    df = get_tte_features(df=df, feats=tte_feats, exp=exp)

    if "option_returns" in core_feats:
        # Calculate option price returns and add to dataframe
        prices = df["option_mid_price"].to_numpy()
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        df["option_returns"] = returns

    if "stock_returns" in core_feats:
        # Calculate stock price returns and add to dataframe
        prices = df["stock_mid_price"].to_numpy()
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        df["stock_returns"] = returns

    if "option_returns" in core_feats or "stock_returns" in core_feats:
        # Drop the first market open (since returns=0)
        first_time = df["datetime"].iloc[0].time()
        if first_time.hour == 9 and first_time.minute == 30:
            df = df.iloc[1:].reset_index(drop=True)

    if "distance_to_strike" in core_feats:
        # Calculate distance to strike and add to dataframe
        distance = (float(strike) - df["stock_mid_price"])
        df["distance_to_strike"] = distance

    if "moneyness" in core_feats:
        # Calculate moneyness and add to dataframe
        df["moneyness"] = np.log(df["stock_mid_price"] / float(strike))

    if "stock_lob_imbalance" in core_feats:
        # Calculate limit order book (LOB) imbalance and add to dataframe
        df["stock_lob_imbalance"] = (df["stock_ask_size"] - df["stock_bid_size"]) / (df["stock_bid_size"] + df["stock_ask_size"])

    if "option_lob_imbalance" in core_feats:
        # Calculate limit order book (LOB) imbalance and add to dataframe
        df["option_lob_imbalance"] = (df["option_ask_size"] - df["option_bid_size"]) / (df["option_bid_size"] + df["option_ask_size"])

    if "stock_quote_spread" in core_feats:
        # Calculate stock quote spread normalized by mid-price
        df["stock_quote_spread"] = (df["stock_ask"] - df["stock_bid"]) / ((df["stock_ask"] + df["stock_bid"])/2)

    if "option_quote_spread" in core_feats:
        # Calculate option quote spread normalized by mid-price
        df["option_quote_spread"] = (df["option_ask"] - df["option_bid"]) / ((df["option_ask"] + df["option_bid"])/2)

    # Select features
    tte_index = ["tte_" + tte_feats[i] for i in range(len(tte_feats))]
    datetime_index = ["dt_" + datetime_feats[i] for i in range(len(datetime_feats))]
    selected_feats = core_feats + tte_index + datetime_index

    return df[selected_feats]


if __name__ == "__main__":
    from optrade.data.thetadata.get_data import load_all_data
    from optrade.data.thetadata.contracts import Contract
    from rich.console import Console
    console = Console()

    contract = Contract()
    df = load_all_data(
        contract=contract,
        clean_up=False,
        offline=False
    )

    # TTE features
    tte_feats = ["sqrt", "exp_decay"]

    # Datetime features
    datetime_feats = ["sin_minute_of_day", "cos_minute_of_day", "sin_hour_of_week", "cos_hour_of_week"]

    # Select features
    core_feats = [
        "option_returns",
        "stock_returns",
        "distance_to_strike",
        "moneyness",
        "option_lob_imbalance",
        "option_quote_spread",
        "stock_lob_imbalance",
        "stock_quote_spread",
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

    df = transform_features(
        df=df,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        strike=contract.strike,
    )

    print(df.columns == core_feats + [f"tte_{f}" for f in tte_feats] + [f"dt_{f}" for f in datetime_feats])

    print(df.head())
    print(df.to_numpy().shape)
