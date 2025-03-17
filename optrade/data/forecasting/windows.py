import numpy as np
import pandas as pd

from typing import Tuple

# Not used in current implementation (deprecated)
def get_windows(
    df: pd.DataFrame,
    seq_len: int,
    pred_len: int,
    window_stride: int,
    intraday: bool=False,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates rolling windows of data for a given DataFrame.

    *NOTE: Should be primarily used for intraday ease of use, otherwise it is recommended to directly
           convert time series features into a ForecastingDataset found in datasets.py for computational
           efficiency.

    Args:
        df (pd.DataFrame): DataFrame containing the data.
        seq_len (int): Length of the input sequence.
        pred_len (int): Length of the prediction sequence.
        window_stride (int): Number of steps to move the window forward.
        intraday (bool): Whether the data is intraday or not. If True, the function will first split the data
                         into separate trading days before creating individual windows that cannot crossover
                         between days. Otherwise, the function will create windows that can span multiple days.
    Returns:
        input (np.ndarray): Array of input windows of shape (num_windows, seq_len, num_features) where num_features
                            is the number of columns in the DataFrame (removing datetime but adding returns).
        target (np.ndarray): Array of target windows of shape (num_windows, pred_len, 1).
                             Target contains only returns for the 'option_mid_price'.
    """
    datetime = df["datetime"]
    df_copy = df.copy()

    # Define input features (all columns except datetime)
    feature_columns = [col for col in df_copy.columns if col != 'datetime']
    inputs, targets = [], []

    if intraday:
        # Get all unique days
        days = datetime.dt.date.unique()

        # Iterate through each day independently
        for day in days:
            day_mask = datetime.dt.date == day
            day_data = df_copy.loc[day_mask].copy()

            print(f"Length of data: {len(day_data)}")

            # Since returns will be part of the input features but don't exist for 9:30am
            # we remove the market open (9:30am) of each day
            first_time = day_data['datetime'].iloc[0].time()
            if first_time.hour == 9 and first_time.minute == 30:
                day_data = day_data.iloc[1:].reset_index(drop=True)
                print(f"Day data after removing 9:30am: {day_data.tail()}")

            # Raise an error if the length of the day is less than the sum of seq_len+pred_len
            if len(day_data) < seq_len + pred_len:
                raise ValueError(
                    f"seq_len + pred_len = {seq_len + pred_len} exceeds the length of the day. \
                    Either set intraday=False or reduce seq_len and/or pred_len."
                )

            # Get input features and targets, convert to NumPy arrays
            day_features = day_data[feature_columns].to_numpy()
            day_targets = day_data['option_returns'].to_numpy().reshape(-1, 1)

            # Apply the sliding window technique to obtain windows for inputs and targets
            for i in range(0, len(day_data) - seq_len - pred_len + 1, window_stride):
                inputs.append(day_features[i:i+seq_len])
                targets.append(day_targets[i+seq_len:i+seq_len+pred_len])
    else:
        # Since returns will be part of the input features but don't exist for first market open
        # i.e. 9:30am on the first day, we remove it
        first_time = datetime.iloc[0].time()
        if first_time.hour == 9 and first_time.minute == 30:
            df_copy = df_copy.iloc[1:].reset_index(drop=True)

        # Extract features and targets
        features = df_copy[feature_columns].to_numpy()
        targets_data = df_copy['option_returns'].to_numpy().reshape(-1, 1)

        # Create windows
        for i in range(0, len(df_copy) - seq_len - pred_len + 1, window_stride):
            inputs.append(features[i:i+seq_len])
            targets.append(targets_data[i+seq_len:i+seq_len+pred_len])

    # Convert to numpy arrays
    return np.array(inputs), np.array(targets)

if __name__ == "__main__":
    from optrade.data.thetadata.get_data import load_all_data
    from optrade.data.features.get_features import transform_features
    from optrade.data.thetadata.contracts import Contract
    from rich.console import Console
    console = Console()

    contract = Contract(
        root="AAPL",
        start_date="20241107",
        exp="20250117",
        strike=225,
        interval_min=1,
        right="C"
    )

    df = load_all_data(
        contract=contract,
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
        "option_returns",
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

    df = transform_features(
        df=df,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
    )

    x, y = get_windows(
        df=df,
        seq_len=30,
        pred_len=6,
        window_stride=1,
        intraday=False
    )

    print(x.shape, y.shape)
