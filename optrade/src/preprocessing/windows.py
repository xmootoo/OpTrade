from re import split
import numpy as np
import pandas as pd

from typing import Tuple

def get_windows(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    window_stride: int,
    intraday: bool=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates rolling windows of data for a given DataFrame.
    Args:
        df (pd.DataFrame): DataFrame containing the data.
        seq_len (int): Length of the input sequence.
        pred_len (int): Length of the prediction sequence.
        window_stride (int): Number of steps to move the window forward.
        intraday (bool): Whether the data is intraday or not. If True, the function will first split the data
                         into separate trading days before creating individual windows that cannot crossover
                         between days. Otherwise, the function will create windows that can span multiple days.
    Returns:
        input (np.ndarray): Array of input windows.
        target (np.ndarray): Array of target windows.
    """

    datetime = df["datetime"]
    data = df.drop(columns=["datetime"]).to_numpy()

    inputs = []
    targets = []

    if intraday:
        # Get unique days from datetime column
        days = datetime.dt.date.unique()

        for day_idx, day in enumerate(days):
            # Get indices for this day
            day_mask = datetime.dt.date == day
            day_indices = np.where(day_mask)[0]


            if len(day_indices) < seq_len + pred_len:
                print(f"[DEBUG] Skipping day {day} - insufficient data points ({len(day_indices)} < {seq_len + pred_len})")
                continue

            day_windows = 0
            # Create windows for this day
            for i in range(0, len(day_indices) - seq_len - pred_len + 1, window_stride):
                start_idx = day_indices[i]
                input_end_idx = start_idx + seq_len
                target_end_idx = input_end_idx + pred_len

                # Ensure we don't cross day boundaries
                if target_end_idx <= day_indices[-1] + 1:
                    inputs.append(data[start_idx:input_end_idx])
                    targets.append(data[input_end_idx:target_end_idx])
                    day_windows += 1
    else:
        # Create windows across the entire dataset
        window_count = 0
        for i in range(0, len(data) - seq_len - pred_len + 1, window_stride):
            inputs.append(data[i:i + seq_len])
            targets.append(data[i + seq_len:i + seq_len + pred_len])
            window_count += 1

    input_array = np.array(inputs)
    target_array = np.array(targets)

    return input_array, target_array


if __name__ == "__main__":
    from optrade.data.thetadata.get_data import get_data
    from optrade.src.preprocessing.get_features import get_features
    from rich.console import Console
    console = Console()

    df = get_data(
        root="AAPL",
        start_date="20241107",
        end_date="20241114",
        exp="20250117",
        strike=225,
        interval_min=1,
        right="C",
        # save_dir="../historical_data/merged",
        clean_up=True,
        offline=False
    )

    # TTE features
    tte_feats = ["sqrt", "exp_decay"]

    # Datetime features
    datetime_feats = ["sin_timeofday", "cos_timeofday", "dayofweek"]

    # Select features
    core_feats = [
        "datetime",
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
    )

    x, y = get_windows(
        df=df,
        seq_len=10,
        pred_len=5,
        window_stride=1,
        intraday=True
    )
