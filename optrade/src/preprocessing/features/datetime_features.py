import numpy as np
import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt
from datetime import datetime, time
import matplotlib.dates as mdates

# TODO: Implement sin/cos day of week
def get_datetime_features(
    df: pd.DataFrame,
    feats: List=[
        "minuteofday",
        "sin_timeofday",
        "cos_timeofday",
        "dayofweek",
    ],
    dt_col: Optional[str] = "datetime",
    market_open_time: str = "09:30:00",
    market_close_time: str = "16:00:00",
) -> pd.DataFrame:
    """
    Generates optimized datetime features for short-term options forecasting.

    Focused on intraday patterns and weekly option expiration cycles for
    ATM options with 30-45 day expiry using 1-minute market data.

    Args:
        df (pd.DataFrame): DataFrame containing a datetime column
        feats (List): List of datetime features to generate. Options include:
            - "minuteofday": Minute of trading day (0-389 for standard session)
            - "sin_timeofday": Sine transformation of time of day (continuous circular feature)
            - "cos_timeofday": Cosine transformation of time of day (continuous circular feature)
            - "dayofweek": Day of week (0=Monday, 4=Friday)
        dt_col (Optional[str]): Name of datetime column. If None, will attempt to detect it.
        market_open_time (str): Market open time in HH:MM:SS format
        market_close_time (str): Market close time in HH:MM:SS format

    Returns:
        pd.DataFrame: Original DataFrame with additional datetime feature columns
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Find the datetime column if not specified
    if dt_col is None:
        if 'datetime' in df.columns:
            dt_col = 'datetime'
        else:
            # Look for any datetime64 column
            datetime_cols = df.select_dtypes(include=['datetime64']).columns
            if len(datetime_cols) > 0:
                dt_col = datetime_cols[0]
            else:
                # As a fallback, look for any column with a name containing 'date' or 'time'
                time_related_cols = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()]
                if time_related_cols:
                    dt_col = time_related_cols[0]
                else:
                    raise ValueError("Could not find a datetime column")

    # Ensure column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
        result_df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')

    # Parse market hours
    market_open = pd.to_datetime(market_open_time).time()
    market_close = pd.to_datetime(market_close_time).time()

    # Convert times to minutes
    def time_to_minutes(t):
        return t.hour * 60 + t.minute

    open_minutes = time_to_minutes(market_open)
    close_minutes = time_to_minutes(market_close)

    if "minuteofday" in feats:
        time_minutes = result_df[dt_col].dt.hour * 60 + result_df[dt_col].dt.minute
        # Normalize to trading day (0 = market open)
        result_df['dt_minuteofday'] = (time_minutes - open_minutes).astype('float64')

    # Cyclic time encoding - continuous through the trading day
    # Scale from market open to market close
    trading_minutes = close_minutes - open_minutes
    if "sin_timeofday" in feats or "cos_timeofday" in feats:
        time_minutes = result_df[dt_col].dt.hour * 60 + result_df[dt_col].dt.minute
        # Normalize to [0, 2π] across trading day
        normalized_time = 2 * np.pi * (time_minutes - open_minutes) / trading_minutes

        if "sin_timeofday" in feats:
            result_df['dt_sin_timeofday'] = np.sin(normalized_time).astype('float64')

        if "cos_timeofday" in feats:
            result_df['dt_cos_timeofday'] = np.cos(normalized_time).astype('float64')

    if "dayofweek" in feats:
        result_df['dt_dayofweek'] = result_df[dt_col].dt.dayofweek.astype('float64')


    return result_df

def visualize_datetime_features(df, features=None, dt_col='datetime'):
    """
    Create separate plots for each datetime feature with appropriate x-axis units.

    Args:
        df (pd.DataFrame): DataFrame with datetime features
        features (list, optional): List of features to plot. If None, plots all dt_ columns
        dt_col (str): Name of the datetime column
    """
    if features is None:
        features = [col for col in df.columns if col.startswith('dt_')]

    # Group features by their time granularity
    intraday_features = ["dt_minuteofday", "dt_near_market_open", "dt_near_market_close",
                        "dt_sin_timeofday", "dt_cos_timeofday"]
    daily_features = ["dt_dayofweek", "dt_sin_dayofweek", "dt_cos_dayofweek", "dt_is_opex_week"]

    fig_width, fig_height = 12, 6

    # Plot intraday features with minute/hour x-axis
    for feature in [f for f in features if f in intraday_features]:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # For intraday features, use hour:minute format for x-axis
        if df[dt_col].dt.date.nunique() == 1:
            # Single day data - use hour:minute format
            ax.plot(df[dt_col].dt.strftime('%H:%M'), df[feature])
            plt.xticks(rotation=45)

            # Add more x-tick marks for intraday data
            num_ticks = min(24, len(df) // 30)  # Show up to 24 ticks or fewer for sparse data
            tick_indices = np.linspace(0, len(df) - 1, num_ticks, dtype=int)
            plt.xticks(tick_indices, df[dt_col].iloc[tick_indices].dt.strftime('%H:%M'))

        else:
            # Multiple days - use date + hour format
            ax.plot(df[dt_col], df[feature])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            plt.xticks(rotation=45)

        feature_name = feature.replace('dt_', '')
        plt.title(f'{feature_name} over time')
        plt.xlabel('Time')
        plt.ylabel(feature_name)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    # Plot daily features with day x-axis
    for feature in [f for f in features if f in daily_features]:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # For day-based features, use date for x-axis
        if df[dt_col].dt.date.nunique() > 7:
            # If more than a week of data, resample to daily
            daily_data = df.groupby(df[dt_col].dt.date)[feature].mean().reset_index()
            ax.plot(daily_data[dt_col], daily_data[feature])
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
        else:
            # Plot each day's data
            ax.plot(df[dt_col].dt.strftime('%Y-%m-%d'), df[feature])

        plt.xticks(rotation=45)
        feature_name = feature.replace('dt_', '')
        plt.title(f'{feature_name} over time')
        plt.xlabel('Date')
        plt.ylabel(feature_name)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage
if __name__ == "__main__":
    from optrade.data.thetadata.get_data import get_data
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

    # Generate datetime features
    result = get_datetime_features(df=df)

    # Display results with dtypes
    print(result[['datetime'] +
                [col for col in result.columns if col.startswith('dt_')]].head())
    print("\nColumn dtypes:")
    for col in result.columns:
        if col.startswith('dt_'):
            print(f"{col}: {result[col].dtype}")

    visualize_datetime_features(result)
