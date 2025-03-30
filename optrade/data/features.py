import pandas as pd
import numpy as np
from typing import Optional, List
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime


def dt_features(
    df: pd.DataFrame,
    feats: List[str],
    dt_col: Optional[str] = "datetime",
    market_open_time: Optional[str] = "09:30:00",
    market_close_time: Optional[str] = "16:00:00",
) -> pd.DataFrame:
    """Generates datetime features for options.

    Args:
        df: DataFrame containing a datetime column.
        feats: List of datetime features to generate. Options include:
            - minute_of_day: Minute of trading day (0-389 for standard session)
            - sin_minute_of_day: Sine transformation of time of day (continuous circular feature)
            - cos_minute_of_day: Cosine transformation of time of day (continuous circular feature)
            - day_of_week: Day of week (0=Monday, 4=Friday)
            - hour_of_week: Hour position in trading week as proportion (0.0-1.0)
            - sin_hour_of_week: Sine transformation of hour of week (continuous circular feature)
            - cos_hour_of_week: Cosine transformation of hour of week (continuous circular feature)
        dt_col: Name of datetime column. If None, will attempt to detect it.
            Defaults to datetime.
        market_open_time: Market open time in HH:MM:SS format.
            Defaults to 09:30:00.
        market_close_time: Market close time in HH:MM:SS format.
            Defaults to 16:00:00.

    Returns:
        Original DataFrame with additional datetime feature columns, prefixed with dt\\_.

    Examples:
        Basic usage:

        >>> import pandas as pd
        >>> data = pd.DataFrame({
        ...     "datetime": pd.date_range("2023-01-02 09:30:00", periods=5, freq="1min")
        ... })
        >>> feats = ["minute_of_day", "day_of_week"]
        >>> result = dt_features(data, feats)
        >>> result.columns
        Index(['datetime', 'dt_minute_of_day', 'dt_day_of_week'], dtype='object')

        Using custom datetime column name:

        >>> data = pd.DataFrame({
        ...     "timestamp": pd.date_range("2023-01-02 09:30:00", periods=5, freq="1min")
        ... })
        >>> result = dt_features(data, feats, dt_col="timestamp")
        >>> result.columns
        Index(['timestamp', 'dt_minute_of_day', 'dt_day_of_week'], dtype='object')
    """
    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Find the datetime column if not specified
    if dt_col is None:
        if "datetime" in df.columns:
            dt_col = "datetime"
        else:
            # Look for any datetime64 column
            datetime_cols = df.select_dtypes(include=["datetime64"]).columns
            if len(datetime_cols) > 0:
                dt_col = datetime_cols[0]
            else:
                # As a fallback, look for any column with a name containing "date" or "time"
                time_related_cols = [
                    col
                    for col in df.columns
                    if "date" in col.lower() or "time" in col.lower()
                ]
                if time_related_cols:
                    dt_col = time_related_cols[0]
                else:
                    raise ValueError("Could not find a datetime column")

    # Ensure column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
        result_df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

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
        result_df["dt_minute_of_day"] = (time_minutes - open_minutes).astype("float64")

    # Cyclic time encoding - continuous through the trading day
    # Scale from market open to market close
    if "sin_minute_of_day" in feats or "cos_minute_of_day" in feats:
        time_minutes = result_df[dt_col].dt.hour * 60 + result_df[dt_col].dt.minute
        # Normalize to [0, 2π] across trading day
        normalized_time = (
            2 * np.pi * (time_minutes - open_minutes) / trading_minutes_per_day
        )

        if "sin_minute_of_day" in feats:
            result_df["dt_sin_minute_of_day"] = np.sin(normalized_time).astype(
                "float64"
            )
        if "cos_minute_of_day" in feats:
            result_df["dt_cos_minute_of_day"] = np.cos(normalized_time).astype(
                "float64"
            )

    if "day_of_week" in feats:
        result_df["dt_day_of_week"] = result_df[dt_col].dt.day_of_week.astype("float64")

    # Hour of week features - considering a 5-day trading week
    if any(
        f in feats for f in ["hour_of_week", "sin_hour_of_week", "cos_hour_of_week"]
    ):
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
        market_minutes = np.maximum(
            0, np.minimum(time_minutes - open_minutes, trading_minutes_per_day)
        )
        hours_from_current_day = market_minutes / 60

        # Total hours elapsed in the trading week
        hours_elapsed = hours_from_completed_days + hours_from_current_day

        if "hour_of_week" in feats:
            # Normalize to [0, 1] across the trading week
            result_df["dt_hour_of_week"] = (
                hours_elapsed / total_trading_hours_per_week
            ).astype("float64")

        if "sin_hour_of_week" in feats or "cos_hour_of_week" in feats:
            # Normalize to [0, 2π] across the trading week
            normalized_week_time = (
                2 * np.pi * hours_elapsed / total_trading_hours_per_week
            )

            if "sin_hour_of_week" in feats:
                result_df["dt_sin_hour_of_week"] = np.sin(normalized_week_time).astype(
                    "float64"
                )
            if "cos_hour_of_week" in feats:
                result_df["dt_cos_hour_of_week"] = np.cos(normalized_week_time).astype(
                    "float64"
                )

    return result_df


def tte_features(
    df: pd.DataFrame,
    feats: List[str],
    exp: str,
) -> pd.DataFrame:
    """
    Generate Time to Expiration (TTE) features for a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing datetime column in format "YYYY-MM-DD HH:MM:SS".
            The function will try to identify a datetime column if not explicitly named "datetime".
        feats (List): List of features to generate. Options include:
            - "linear": raw TTE in minutes
            - "inverse": 1/TTE (in minutes)
            - "sqrt": √(TTE minutes)
            - "inverse_sqrt": 1/√(TTE minutes)
            - "exp_decay": exp(-TTE/contract_length)
        exp (str): The expiration date of the option in YYYYMMDD format. The expiration time
            is assumed to be 16:30 (4:30 PM) on the expiration date.

    Returns:
        pd.DataFrame: The original DataFrame with additional TTE feature columns. Each requested
            feature will be added with a prefix "tte\\_" (e.g., "tte\\_inverse").
            All TTE features are guaranteed to be float64 type.
    """
    if feats == []:
        return df

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Convert expiration date string to datetime
    exp_date = datetime.strptime(exp, "%Y%m%d")

    # Set expiration time to 4:30 PM on expiration date
    exp_datetime = exp_date.replace(hour=16, minute=30, second=0)

    # Find the datetime column - standardized approach
    # First check if "datetime" column exists
    if "datetime" in df.columns:
        dt_col = "datetime"
    else:
        # Look for any datetime64 column
        datetime_cols = df.select_dtypes(include=["datetime64"]).columns
        if len(datetime_cols) > 0:
            dt_col = datetime_cols[0]
        else:
            # As a fallback, look for any column with a name containing "date" or "time"
            time_related_cols = [
                col
                for col in df.columns
                if "date" in col.lower() or "time" in col.lower()
            ]
            if time_related_cols:
                dt_col = time_related_cols[0]
            else:
                raise ValueError("Could not find a datetime column")

    # Ensure column is datetime type - using pandas" robust datetime conversion
    if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
        result_df[dt_col] = pd.to_datetime(df[dt_col], errors="coerce")

    # Calculate TTE in minutes as float64
    result_df["tte_minutes"] = (
        exp_datetime - result_df[dt_col]
    ).dt.total_seconds().astype("float64") / 60

    # Calculate maximum TTE (contract length in minutes)
    contract_length = result_df["tte_minutes"].max()

    # Generate requested features
    if "tte" in feats or "all" in feats:
        # Linear TTE (raw minutes)
        result_df["tte"] = result_df["tte_minutes"].astype("float64")

    if "inverse" in feats or "all" in feats:
        # Inverse TTE (1/minutes)
        # Handle potential division by zero with np.inf handling
        result_df["tte_inverse"] = np.where(
            result_df["tte_minutes"] > 0, 1 / result_df["tte_minutes"], np.inf
        ).astype("float64")

    if "sqrt" in feats or "all" in feats:
        # Square root of TTE
        result_df["tte_sqrt"] = np.sqrt(result_df["tte_minutes"]).astype("float64")

    if "inverse_sqrt" in feats or "all" in feats:
        # Inverse square root of TTE
        # Handle potential division by zero
        result_df["tte_inverse_sqrt"] = np.where(
            result_df["tte_minutes"] > 0, 1 / np.sqrt(result_df["tte_minutes"]), np.inf
        ).astype("float64")

    if "exp_decay" in feats or "all" in feats:
        # Exponential decay with lambda = 1/contract_length
        result_df["tte_exp_decay"] = np.exp(
            -result_df["tte_minutes"] / contract_length
        ).astype("float64")

    # Remove intermediate calculation if not requested
    if "linear" not in feats and "all" not in feats:
        result_df = result_df.drop("tte_minutes", axis=1)
    else:
        # If we"re keeping tte_minutes, ensure it"s float64
        result_df["tte_minutes"] = result_df["tte_minutes"].astype("float64")

    return result_df


def transform_features(
    df: pd.DataFrame,
    core_feats: List[str],
    tte_feats: Optional[List[str]] = None,
    datetime_feats: Optional[List[str]] = None,
    strike: Optional[float] = None,
    exp: Optional[str] = None,
    keep_datetime: bool = False,
) -> pd.DataFrame:
    """
    Selects and transforms features from a DataFrame based on specified feature lists.

    This function allows the selection of core features from NBBO and OHLCVC data,
    as well as the generation of time-to-expiration features and datetime-based features.
    It can also calculate derived features such as returns, moneyness, and LOB imbalance.

    Args:
        df: The DataFrame containing the raw features.
        core_feats: List of core features to select.
        tte_feats: List of Time to Expiration (TTE) features to generate.
        datetime_feats: List of datetime features to generate.
        strike: Strike price of the option, required for moneyness and distance_to_strike calculations.
        exp: Expiration date string in YYYYMMDD format, required for TTE feature generation.

    Returns:
        DataFrame containing only the requested features.

    Core feature options (subset of NBBO and OHLCVC):
        - datetime: Timestamp of the data point
        - {asset}_mid_price: Mid price of the asset
        - {asset}_bid_size: Size of the bid
        - {asset}_bid_exchange: Exchange of the bid
        - {asset}_bid: Bid price
        - {asset}_bid_condition: Condition of the bid
        - {asset}_ask_size: Size of the ask
        - {asset}_ask_exchange: Exchange of the ask
        - {asset}_ask: Ask price
        - {asset}_ask_condition: Condition of the ask
        - {asset}_open: Opening price
        - {asset}_high: High price
        - {asset}_low: Low price
        - {asset}_close: Closing price
        - {asset}_volume: Volume
        - {asset}_count: Count

    where "{asset}" is either "option" or "stock".

    Advanced core feature options:
        - {asset}_returns: Mid-price returns
        - log_{asset}_returns: Log mid-price returns
        - {asset}_lob_imbalance: Limit order book imbalance
        - {asset}_quote_spread: Quote spread normalized by mid-price
        - moneyness: Log(S/K)
        - distance_to_strike: Linear distance to strike price

    where "{asset}" is either "option" or "stock".

    TTE features options:
        - tte: Time to expiration
        - inverse: Inverse time to expiration
        - sqrt: Square root of time to expiration
        - inverse_sqrt: Inverse square root of time to expiration
        - exp_decay: Exponential decay of time to expiration

    Datetime features options:
        - minute_of_day: Minute of the day
        - sin_minute_of_day: Sine of minute of the day
        - cos_minute_of_day: Cosine of minute of the day
        - day_of_week: Day of the week
        - sin_day_of_week: Sine of day of the week
        - cos_day_of_week: Cosine of day of the week
        - hour_of_week: Hour of the week
        - sin_hour_of_week: Sine of hour of the week
        - cos_hour_of_week: Cosine of hour of the week

    Examples:
        Basic usage::

            from optrade.data.thetadata.contracts import Contract
            contract = Contract()
            df = contract.load_data()

            # TTE features
            tte_feats = ["sqrt", "exp_decay"]

            # Datetime features
            datetime_feats = ["sin_minute_of_day", "cos_minute_of_day",
                              "sin_hour_of_week", "cos_hour_of_week"]

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
                exp=contract.exp
            )
    """

    # Generate additional features
    if datetime_feats is not None:
        df = dt_features(df=df, feats=datetime_feats)

    if tte_feats is not None and exp is not None:
        assert exp is not None, "Expiration date is required for TTE feature generation"
        df = tte_features(df=df, feats=tte_feats, exp=exp)

    if "option_returns" or "log_option_returns" in core_feats:
        # Calculate option price returns and add to dataframe
        prices = df["option_mid_price"].to_numpy()
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        df["option_returns"] = returns

        if "log_option_returns" in core_feats:
            df["log_option_returns"] = np.log(1 + returns)

    if "stock_returns" in core_feats or "log_stock_returns" in core_feats:
        # Calculate stock price returns and add to dataframe
        prices = df["stock_mid_price"].to_numpy()
        returns = np.zeros_like(prices)
        returns[1:] = (prices[1:] - prices[:-1]) / prices[:-1]
        df["stock_returns"] = returns
        if "log_stock_returns" in core_feats:
            df["log_stock_returns"] = np.log(1 + returns)

    if "option_returns" in core_feats or "stock_returns" in core_feats:
        # Drop the first market open (since returns=0)
        first_time = df["datetime"].iloc[0].time()
        if first_time.hour == 9 and first_time.minute == 30:
            df = df.iloc[1:].reset_index(drop=True)

    if "distance_to_strike" in core_feats:
        assert (
            strike is not None
        ), "Strike price required for distance_to_strike feature"

        # Calculate distance to strike and add to dataframe
        distance = float(strike) - df["stock_mid_price"]
        df["distance_to_strike"] = distance

    if "moneyness" in core_feats:
        assert strike is not None, "Strike price required for moneyness feature"

        # Calculate moneyness and add to dataframe
        df["moneyness"] = np.log(df["stock_mid_price"] / float(strike))

    if "stock_lob_imbalance" in core_feats:
        # Calculate limit order book (LOB) imbalance and add to dataframe
        df["stock_lob_imbalance"] = (df["stock_ask_size"] - df["stock_bid_size"]) / (
            df["stock_bid_size"] + df["stock_ask_size"]
        )

    if "option_lob_imbalance" in core_feats:
        # Calculate limit order book (LOB) imbalance and add to dataframe
        df["option_lob_imbalance"] = (df["option_ask_size"] - df["option_bid_size"]) / (
            df["option_bid_size"] + df["option_ask_size"]
        )

    if "stock_quote_spread" in core_feats:
        # Calculate stock quote spread normalized by mid-price
        df["stock_quote_spread"] = (df["stock_ask"] - df["stock_bid"]) / (
            (df["stock_ask"] + df["stock_bid"]) / 2
        )

    if "option_quote_spread" in core_feats:
        # Calculate option quote spread normalized by mid-price
        df["option_quote_spread"] = (df["option_ask"] - df["option_bid"]) / (
            (df["option_ask"] + df["option_bid"]) / 2
        )

    # Select features
    tte_index = (
        ["tte_" + tte_feats[i] for i in range(len(tte_feats))]
        if tte_feats is not None
        else []
    )
    datetime_index = (
        ["dt_" + datetime_feats[i] for i in range(len(datetime_feats))]
        if datetime_feats is not None
        else []
    )
    selected_feats = core_feats + tte_index + datetime_index

    if keep_datetime:
        selected_feats += ["datetime"]

    return df[selected_feats]
