import numpy as np
import pandas as pd
from typing import List, Optional
import matplotlib.pyplot as plt
from datetime import datetime, time
import matplotlib.dates as mdates

def get_datetime_features(
    df: pd.DataFrame,
    feats: List=[
        "minute_of_day",
        "sin_minute_of_day",
        "cos_minute_of_day",
        "day_of_week",
        "hour_of_week",
        "sin_hour_of_week",
        "cos_hour_of_week",
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
            - "minute_of_day": Minute of trading day (0-389 for standard session)
            - "sin_minute_of_day": Sine transformation of time of day (continuous circular feature)
            - "cos_minute_of_day": Cosine transformation of time of day (continuous circular feature)
            - "day_of_week": Day of week (0=Monday, 4=Friday)
            - "hour_of_week": Hour position in trading week as proportion (0.0-1.0)
            - "sin_hour_of_week": Sine transformation of hour of week (continuous circular feature)
            - "cos_hour_of_week": Cosine transformation of hour of week (continuous circular feature)
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
    trading_minutes_per_day = close_minutes - open_minutes

    if "minute_of_day" in feats:
        time_minutes = result_df[dt_col].dt.hour * 60 + result_df[dt_col].dt.minute
        # Normalize to trading day (0 = market open)
        result_df['dt_minute_of_day'] = (time_minutes - open_minutes).astype('float64')

    # Cyclic time encoding - continuous through the trading day
    # Scale from market open to market close
    if "sin_minute_of_day" in feats or "cos_minute_of_day" in feats:
        time_minutes = result_df[dt_col].dt.hour * 60 + result_df[dt_col].dt.minute
        # Normalize to [0, 2π] across trading day
        normalized_time = 2 * np.pi * (time_minutes - open_minutes) / trading_minutes_per_day

        if "sin_minute_of_day" in feats:
            result_df['dt_sin_minute_of_day'] = np.sin(normalized_time).astype('float64')
        if "cos_minute_of_day" in feats:
            result_df['dt_cos_minute_of_day'] = np.cos(normalized_time).astype('float64')

    if "day_of_week" in feats:
        result_df['dt_day_of_week'] = result_df[dt_col].dt.day_of_week.astype('float64')

    # Hour of week features - considering a 5-day trading week
    if any(f in feats for f in ["hour_of_week", "sin_hour_of_week", "cos_hour_of_week"]):
        # Calculate total trading hours in a week (5 trading days)
        trading_hours_per_day = trading_minutes_per_day / 60
        total_trading_hours_per_week = 5 * trading_hours_per_day

        # Get day of week (0=Monday, 4=Friday)
        dow = result_df[dt_col].dt.day_of_week

        # Calculate hours elapsed in the week for each timestamp
        # First, calculate full days completed
        hours_from_completed_days = dow * trading_hours_per_day

        # Then add hours elapsed in the current day
        time_minutes = result_df[dt_col].dt.hour * 60 + result_df[dt_col].dt.minute
        # Only count minutes during market hours
        market_minutes = np.maximum(0, np.minimum(time_minutes - open_minutes, trading_minutes_per_day))
        hours_from_current_day = market_minutes / 60

        # Total hours elapsed in the trading week
        hours_elapsed = hours_from_completed_days + hours_from_current_day

        if "hour_of_week" in feats:
            # Normalize to [0, 1] across the trading week
            result_df['dt_hour_of_week'] = (hours_elapsed / total_trading_hours_per_week).astype('float64')

        if "sin_hour_of_week" in feats or "cos_hour_of_week" in feats:
            # Normalize to [0, 2π] across the trading week
            normalized_week_time = 2 * np.pi * hours_elapsed / total_trading_hours_per_week

            if "sin_hour_of_week" in feats:
                result_df['dt_sin_hour_of_week'] = np.sin(normalized_week_time).astype('float64')
            if "cos_hour_of_week" in feats:
                result_df['dt_cos_hour_of_week'] = np.cos(normalized_week_time).astype('float64')

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
    intraday_features = ["dt_minute_of_day", "dt_near_market_open", "dt_near_market_close",
                        "dt_sin_minute_of_day", "dt_cos_minute_of_day"]
    daily_features = ["dt_day_of_week", "dt_sin_day_of_week", "dt_cos_day_of_week", "dt_is_opex_week"]
    weekly_features = ["dt_hour_of_week", "dt_sin_hour_of_week", "dt_cos_hour_of_week"]

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

    # Plot weekly features with appropriate x-axis
    for feature in [f for f in features if f in weekly_features]:
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))

        # For hour_of_week features, use a specialized plot that shows progression through the week
        # Create a categorical x-axis showing days of the week
        days_of_week = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']

        # Determine the date range in the data
        date_range = pd.date_range(start=df[dt_col].min().date(), end=df[dt_col].max().date())

        # Plot based on the number of unique days
        if df[dt_col].dt.date.nunique() <= 5:  # Less than or equal to one week of data
            # Plot with day labels and time within day
            df['dow_name'] = df[dt_col].dt.day_of_week.map(lambda x: days_of_week[x])
            df['time_str'] = df[dt_col].dt.strftime('%H:%M')

            # Sort data for consistent plotting
            plot_df = df.sort_values([dt_col])

            # Create compound x-tick labels: Day + Time
            x_labels = plot_df['dow_name'] + ' ' + plot_df['time_str']

            # Plot feature against these labels
            ax.plot(range(len(x_labels)), plot_df[feature])

            # Set x-ticks at reasonable intervals
            num_ticks = min(20, len(x_labels))  # Limit number of ticks for readability
            tick_positions = np.linspace(0, len(x_labels)-1, num_ticks, dtype=int)
            ax.set_xticks(tick_positions)
            ax.set_xticklabels([x_labels.iloc[pos] for pos in tick_positions], rotation=45)

        else:  # More than one week of data
            # Create aggregated view across weeks
            df['day_of_week'] = df[dt_col].dt.day_of_week
            df['hour'] = df[dt_col].dt.hour
            df['minute'] = df[dt_col].dt.minute

            # Create a compound key for time within the week (day + hour + minute)
            df['week_time_key'] = df['day_of_week'].astype(str) + '_' + df['hour'].astype(str) + ':' + df['minute'].astype(str).str.zfill(2)

            # Aggregate values by week_time_key
            agg_data = df.groupby('week_time_key')[feature].mean().reset_index()

            # Sort by natural order within week
            agg_data['day'] = agg_data['week_time_key'].str.split('_').str[0].astype(int)
            agg_data['time'] = agg_data['week_time_key'].str.split('_').str[1]
            agg_data = agg_data.sort_values(['day', 'time'])

            # Plot with appropriate x-axis
            ax.plot(range(len(agg_data)), agg_data[feature])

            # Create better x-labels combining day and time
            agg_data['x_label'] = agg_data['day'].map(lambda x: days_of_week[x][:3]) + ' ' + agg_data['time']

            # Set x-ticks at reasonable intervals
            num_labels = min(15, len(agg_data))  # Limit for readability
            positions = np.linspace(0, len(agg_data)-1, num_labels, dtype=int)
            ax.set_xticks(positions)
            ax.set_xticklabels([agg_data['x_label'].iloc[pos] for pos in positions], rotation=45)

        feature_name = feature.replace('dt_', '')
        plt.title(f'{feature_name} through trading week')
        plt.xlabel('Trading week position')
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

    # Plot sine and cosine pairs on the same plot for cyclical features
    cyclical_pairs = [
        ('dt_sin_minute_of_day', 'dt_cos_minute_of_day', 'minute_of_day'),
        ('dt_sin_hour_of_week', 'dt_cos_hour_of_week', 'hour_of_week'),
        ('dt_sin_day_of_week', 'dt_cos_day_of_week', 'day_of_week')
    ]

    for sin_feat, cos_feat, feature_base in cyclical_pairs:
        if sin_feat in features and cos_feat in features:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height))

            # Create scatter plot of sine vs cosine values to show the cyclical nature
            ax.scatter(df[sin_feat], df[cos_feat], alpha=0.5,
                       c=df[f'dt_{feature_base}'] if f'dt_{feature_base}' in df.columns else None)

            # Add a unit circle for reference
            theta = np.linspace(0, 2*np.pi, 100)
            ax.plot(np.sin(theta), np.cos(theta), 'r--', alpha=0.3)

            # Mark the cardinal points on the circle
            ax.plot([0, 0], [-1.1, 1.1], 'k--', alpha=0.2)  # Vertical line
            ax.plot([-1.1, 1.1], [0, 0], 'k--', alpha=0.2)  # Horizontal line

            plt.title(f'Circular encoding of {feature_base}')
            plt.xlabel(f'sin_{feature_base}')
            plt.ylabel(f'cos_{feature_base}')
            plt.grid(True, alpha=0.3)
            ax.set_aspect('equal')
            plt.tight_layout()
            plt.show()

# Example usage
if __name__ == "__main__":
    from optrade.data.thetadata.get_data import load_all_data
    from rich.console import Console
    console = Console()

    from optrade.data.thetadata.contracts import Contract

    contract = Contract(
        root="AAPL",
        start_date="20241107",
        end_date="20241114",
        exp="20250117",
        strike=225,
        interval_min=1,
        right="C",
    )

    df = load_all_data(
        contract=contract,
        clean_up=False,
        offline=True,
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
