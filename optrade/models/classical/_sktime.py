import numpy as np
import pandas as pd
import random
from pydantic import BaseModel
from typing import Tuple, Union, Optional
from rich.console import Console

# sklearn
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

# sktime
from sktime.forecasting.compose import make_reduction
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.var import VAR
from sktime.forecasting.arima import AutoARIMA
from sktime.split import temporal_train_test_split
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error
from sktime.forecasting.compose._reduce import (
    RecursiveTabularRegressionForecaster,
    DirectTabularRegressionForecaster,
    MultioutputTabularRegressionForecaster
)

def get_forecaster(
    model_id: str,
    strategy: str,
    seq_len: int,
    model_params: Optional[BaseModel] = None,
    seed: int = 1995,
) -> Union[
    RecursiveTabularRegressionForecaster,
    DirectTabularRegressionForecaster,
    MultioutputTabularRegressionForecaster
]:
    """Create a linear regression reduction forecaster with the specified strategy.

    Args:
        model_id (str): Identifier for the base model.
        strategy (str): Strategy for multi-step forecasting. Options:
            - "recursive": Trains a single model to predict one step ahead, then iteratively
              uses predictions as inputs for subsequent steps. Error may accumulate.
              Training: Each window maps to the next single value.
              Prediction: Iteratively applies the model, feeding predictions back as inputs.

            - "direct": Trains separate models for each forecast horizon. No error accumulation,
              but may not capture inter-horizon dependencies.
              Training: Each window maps to a specific horizon, with separate models for each step.
              Prediction: Each model predicts its specific horizon independently.

            - "multioutput": Trains a single model that predicts all horizons at once. Captures
              inter-horizon dependencies without error accumulation.
              Training: Each window maps to a vector of all future values in the horizon.
              Prediction: The model directly outputs all future values in one step.

        seq_len (int): Lookback window size (window_length parameter in make_reduction).
            Defines how many past observations are used to predict future values.
            Defaults to 12.

    Returns:
        Union[RecursiveTabularRegressionForecaster, DirectTabularRegressionForecaster, MultioutputTabularRegressionForecaster]:
            Configured linear regression reduction forecaster based on the selected strategy.

    Raises:
        ValueError: If an unsupported strategy is specified.

    Notes:
        Window Creation Logic:
        - Recursive: For a window_length=9 and training data with 14 observations, creates
          windows like:
          |----------------------------|
          | x x x x x x x x x y * * * *|
          | * x x x x x x x x x y * * *|
          | * * x x x x x x x x x y * *|
          | * * * x x x x x x x x x y *|
          | * * * * x x x x x x x x x y|
          |----------------------------|

        - Direct (windows_identical=True): For fh=[2,4], uses only windows that can accommodate
          the maximum horizon:
          |----------------------------|
          | x x x x x x x x x * * * y *| (for horizon 4)
          | * x x x x x x x x x * * * y|
          |----------------------------|

          |----------------------------|
          | x x x x x x x x x * y * * *| (for horizon 2, using same windows)
          | * x x x x x x x x x * y * *|
          |----------------------------|

        - Multioutput: Similar to direct with windows_identical=True, but each window
          maps to multiple target horizons simultaneously.
          |----------------------------|
          | x x x x x x x x x y y y y *| (for horizon 4)
          | * x x x x x x x x x y y y y|
          |----------------------------|
    """

    if model_id == "linear":
        regressor = LinearRegression()
    elif model_id == "random_forest":
        regressor = RandomForestRegressor(
            n_estimators=model_params.n_estimators,
            max_depth=model_params.max_depth,
            min_samples_leaf=model_params.min_samples_leaf,
            random_state=seed,
        )
    elif model_id == "gradient_boosting":
        regressor = GradientBoostingRegressor(
            n_estimators=model_params.n_estimators,
            learning_rate=model_params.learning_rate,
            max_depth=model_params.max_depth,
            random_state=seed
        )
    elif model_id == "var":
        regressor = VAR(
            maxlags=model_params.maxlags,
            trend=model_params.trend,
            seasonal=model_params.seasonal,
            seasonal_periods=model_params.seasonal_periods,
        )
    elif model_id == "arima":
        regressor = AutoARIMA(
            start_p=model_params.start_p,
            start_q=model_params.start_q,
            max_p=model_params.max_p,
            max_q=model_params.max_q,
            seasonal=model_params.seasonal,
            information_criterion=model_params.information_criterion,
        )
    else:
        raise ValueError(f"Unsupported model_id: {model_id}. Please select: 'linear', 'random_forest', 'gradient_boosting', 'var', or 'arima'.")

    # Create the reduction forecaster with the specified window size and forecasting strategy
    forecaster = make_reduction(
        estimator=regressor,
        window_length=seq_len,  # lookback window size
        strategy=strategy       # strategy for multi-step forecasting
    )
    return forecaster

def prepare_data(X: pd.DataFrame, y: pd.Series, pred_len: int, test_size: float=0.2) -> Tuple[
    pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, ForecastingHorizon
]:
    """Split the data temporally into training and testing sets."""
    y_train, y_test = temporal_train_test_split(y, test_size=test_size)
    X_train, X_test = temporal_train_test_split(X, test_size=test_size)

    # Ensure the indices have frequency set
    if isinstance(y_test.index, pd.DatetimeIndex):
        # Get the frequency from the original index if possible
        freq = y_test.index.freq

        # If freq is None, try to infer it
        if freq is None:
            freq = pd.infer_freq(y_test.index)

        # If still None, use the most common difference
        if freq is None:
            # Use the most common time difference between points as the frequency
            if len(y_test.index) > 1:
                diff = y_test.index.to_series().diff().dropna()
                if not diff.empty:
                    most_common_diff = diff.mode()[0]
                    # Reindex with explicit frequency
                    y_test = y_test.asfreq(most_common_diff)
                    y_train = y_train.asfreq(most_common_diff)
                    X_test = X_test.asfreq(most_common_diff)
                    X_train = X_train.asfreq(most_common_diff)

        # Create forecast horizon with proper frequency
        fh_indices = y_test.index[:pred_len]
        fh = ForecastingHorizon(fh_indices, is_relative=False)
        print(f"ForecastingHorizon: {fh}")
    else:
        # For numeric indices, create a forecasting horizon with relative steps
        fh = ForecastingHorizon(np.arange(1, pred_len + 1))

    return X_train, X_test, y_train, y_test, fh


def run_forecast(
    forecaster: Union[
        RecursiveTabularRegressionForecaster,
        DirectTabularRegressionForecaster,
        MultioutputTabularRegressionForecaster
    ],
    X: pd.DataFrame,
    y: pd.Series,
    strategy: str = "recursive",
    test_size: float = 0.2,
    seq_len: int = 12,
    pred_len: int = 6
) -> Tuple[
    Union[
        RecursiveTabularRegressionForecaster,
        DirectTabularRegressionForecaster,
        MultioutputTabularRegressionForecaster
    ],
    pd.Series
]:
    """Run the linear regression forecasting pipeline.

    Args:
        X (pd.DataFrame): Feature dataframe with shape (num_time_steps, num_input_features)
        y (pd.Series): Target series with shape (num_time_steps,)
        strategy (str): Strategy for multi-step forecasting. Options: "recursive", "direct",
            or "multioutput". Defaults to "recursive".
        test_size (float or int): Size of the test set. Defaults to 0.2.
        seq_len (int): Lookback window size. Defaults to 12.
        pred_len (int): Forecast horizon size. Defaults to 6.

    Returns:
        Tuple[Union[RecursiveTabularRegressionForecaster, DirectTabularRegressionForecaster,
              MultioutputTabularRegressionForecaster], pd.Series]:
            - forecaster: The fitted forecasting model
            - y_pred: The forecast predictions as a pandas Series
    """
    # Prepare data
    X_train, X_test, y_train, y_test, fh = prepare_data(X, y, pred_len, test_size)

    # Fit forecaster - handling different strategies appropriately
    if strategy == "direct" or strategy == "multioutput":
        # Both direct and multioutput strategies need the fh during fit
        forecaster.fit(y_train, X=X_train, fh=fh)
    else:
        # Recursive strategy doesn't need fh during fit
        forecaster.fit(y_train, X=X_train, fh=fh)

    # Generate forecasts
    y_pred = forecaster.predict(fh, X=X_test)

    # Evaluate
    mape = mean_absolute_percentage_error(y_test.iloc[:pred_len], y_pred)
    print(f"MAPE: {mape:.4f}")

    return forecaster, y_pred


# def create_data(periods=120, freq='D', noise_level=0.5, n_series=5):
#     """Create a simpler complex time series dataset with only numeric features.

#     Args:
#         periods (int): Number of time periods to generate. Defaults to 120.
#         freq (str): Frequency of the time series ('D' for daily, etc). Defaults to 'D'.
#         noise_level (float): Amount of random noise to add. Defaults to 0.5.

#     Returns:
#         tuple: (X, y) where X is a DataFrame of features and y is a Series of target values
#     """

#     # Create date range with explicit frequency
#     dates = pd.date_range(start='2020-01-01', periods=periods, freq=freq)

#     # Create target variable with trend, seasonality and noise
#     trend = 10 + 0.5 * np.sqrt(np.arange(periods))
#     seasonality = 5 * np.sin(2 * np.pi * np.arange(periods) / 30)  # Monthly cycle
#     noise = np.random.normal(0, noise_level, periods)

#     # Combine components for target
#     y_values = trend + seasonality + noise

#     # Create target series with explicit frequency
#     y = pd.Series(y_values, index=dates)

#     # Create a small set of purely numeric features
#     X_data = {
#         # Time features converted to numeric
#         'day_of_year': dates.dayofyear,
#         'month_num': dates.month,

#         # Lag features (3 lags)
#         'lag_1': y.shift(1).values,
#         'lag_2': y.shift(2).values,
#         'lag_3': y.shift(3).values,

#         # Rolling feature
#         'rolling_mean_7': y.shift(1).rolling(window=7).mean().values
#     }

#     # Create dataframe
#     X = pd.DataFrame(X_data, index=dates)

#     # Drop rows with NaN values
#     X = X.dropna()
#     y = y[X.index]

#     # Ensure all data is float type
#     X = X.astype(float)

#     return X, y


import pandas as pd
import numpy as np
import random
from rich.console import Console

def create_data(periods=120, freq='D', noise_level=0.5, n_series=5, random_seed=42, ensure_aligned_index=True):
    """Create a panel dataset with multiple time series having slightly different parameters.

    Args:
        periods (int): Base number of time periods to generate. Defaults to 120.
        freq (str): Frequency of the time series ('D' for daily, etc). Defaults to 'D'.
        noise_level (float): Base amount of random noise to add. Defaults to 0.5.
        n_series (int): Number of time series to generate. Defaults to 5.
        random_seed (int): Random seed for reproducibility. Defaults to 42.

    Returns:
        tuple: (X_panel, y_panel) where X_panel is a DataFrame with hierarchical columns
               and y_panel is a DataFrame with each series as a column
    """
    # Set random seed for reproducibility
    np.random.seed(random_seed)
    random.seed(random_seed)

    # Create console for output
    console = Console()

    # Containers for panel data
    X_frames = []
    y_dict = {}

    # If ensure_aligned_index is True, use a common date range for all series
    common_start = pd.Timestamp('2020-01-01') if ensure_aligned_index else None
    common_end = None
    if ensure_aligned_index:
        # Create a common end date based on the maximum possible length
        max_periods = int(periods * 1.1) + 20  # Add buffer for any regeneration
        common_end = common_start + pd.Timedelta(days=max_periods)

    for i in range(n_series):
        console.print(f"Creating series {i+1}...", style="green")

        # Randomize parameters for each series
        series_periods = random.randint(int(periods * 0.9), int(periods * 1.1))
        series_noise = noise_level * random.uniform(0.8, 1.2)
        trend_factor = 0.5 * random.uniform(0.8, 1.2)
        seasonal_amp = 5 * random.uniform(0.7, 1.3)
        seasonal_freq = 30 * random.uniform(0.9, 1.1)  # Slight variation in cycle length

        if ensure_aligned_index:
            # Use common date range but select a subset for each series
            # This ensures all series have the same index, which helps prevent NaN issues
            all_dates = pd.date_range(start=common_start, end=common_end, freq=freq)

            # Choose a random start point within the first 20 days
            start_idx = random.randint(0, min(20, len(all_dates)-series_periods))

            # Get a slice of the dates
            dates = all_dates[start_idx:start_idx+series_periods]
        else:
            # Random start date with slight variations
            start_date = pd.Timestamp('2020-01-01') + pd.Timedelta(days=random.randint(-10, 10))

            # Create date range with explicit frequency
            dates = pd.date_range(start=start_date, periods=series_periods, freq=freq)

        # Create target variable with trend, seasonality and noise
        trend = 10 + trend_factor * np.sqrt(np.arange(series_periods))
        seasonality = seasonal_amp * np.sin(2 * np.pi * np.arange(series_periods) / seasonal_freq)
        noise = np.random.normal(0, series_noise, series_periods)

        # Combine components for target
        y_values = trend + seasonality + noise

        # Create target series with explicit frequency
        y = pd.Series(y_values, index=dates, name=f"series_{i}")

        # Create a small set of purely numeric features
        X_data = {
            # Time features converted to numeric
            'day_of_year': dates.dayofyear,
            'month_num': dates.month,

            # Lag features (3 lags)
            'lag_1': y.shift(1).values,
            'lag_2': y.shift(2).values,
            'lag_3': y.shift(3).values,

            # Series-specific feature
            f'specific_feature_{i}': np.linspace(0, 1, series_periods) * random.uniform(0.5, 1.5),

            # Rolling feature
            'rolling_mean_7': y.shift(1).rolling(window=7).mean().values
        }

        # Create dataframe
        X = pd.DataFrame(X_data, index=dates)

        # Before dropping, print info about NaNs if any exist
        if X.isna().any().any() or y.isna().any():
            console.print(f"Series {i} has NaN values before cleaning", style="yellow")
            console.print(f"NaN count in X: {X.isna().sum().sum()}", style="yellow")
            console.print(f"NaN count in y: {y.isna().sum()}", style="yellow")

        # Drop rows with NaN values
        X = X.dropna()
        y = y[X.index]

        # Ensure we have enough data after dropping NaNs
        if len(X) < 10:
            console.print(f"Warning: Series {i} has too few data points after dropping NaNs", style="red")
            # Regenerate with more data points if needed
            series_periods += 20
            # Recalculate...

        # Ensure all data is float type
        X = X.astype(float)

        # Add series identifier as prefix to all columns
        series_id = f"series_{i}"
        X_with_id = X.copy()
        X_with_id.columns = pd.MultiIndex.from_product([[series_id], X.columns])

        # Append to containers
        X_frames.append(X_with_id)
        y_dict[series_id] = y

    # If using aligned index, reindex all series to a common index
    if ensure_aligned_index:
        # Get union of all indices
        all_indices = sorted(set().union(*[frame.index for frame in X_frames]))
        common_index = pd.DatetimeIndex(all_indices)

        # Reindex all X frames to common index
        reindexed_X_frames = []
        for frame in X_frames:
            reindexed = frame.reindex(common_index)
            reindexed_X_frames.append(reindexed)

        # Reindex all y series to common index
        reindexed_y_dict = {}
        for key, series in y_dict.items():
            reindexed = series.reindex(common_index)
            reindexed_y_dict[key] = reindexed

        # Update containers
        X_frames = reindexed_X_frames
        y_dict = reindexed_y_dict

    # Combine all X dataframes to create panel X
    X_panel = pd.concat(X_frames, axis=1)

    # Create panel y (each series as a column)
    y_panel = pd.DataFrame(y_dict)

    # Final check for any remaining NaN values
    if X_panel.isna().any().any():
        console.print("WARNING: X_panel still contains NaN values", style="red")
        console.print(f"Total NaNs in X_panel: {X_panel.isna().sum().sum()}", style="red")

        # Find which columns have NaNs
        nan_cols = X_panel.columns[X_panel.isna().any()].tolist()
        console.print(f"Columns with NaNs: {nan_cols[:10]}... ({len(nan_cols)} total)", style="red")

        # Fill remaining NaNs with column means
        console.print("Filling remaining NaNs with column means", style="yellow")
        # Ensure we have at least some non-NaN values in each column
        for col in X_panel.columns:
            if X_panel[col].isna().all():
                console.print(f"Column {col} is all NaNs, filling with zeros", style="red")
                X_panel[col] = 0

        X_panel = X_panel.fillna(X_panel.mean())

    if y_panel.isna().any().any():
        console.print("WARNING: y_panel still contains NaN values", style="red")
        console.print(f"Total NaNs in y_panel: {y_panel.isna().sum().sum()}", style="red")

        # Fill remaining NaNs with column means
        console.print("Filling remaining NaNs with column means", style="yellow")

        # Ensure we have at least some non-NaN values in each column
        for col in y_panel.columns:
            if y_panel[col].isna().all():
                console.print(f"Column {col} is all NaNs, filling with zeros", style="red")
                y_panel[col] = 0

        y_panel = y_panel.fillna(y_panel.mean())

    # Final check - replace any remaining NaNs (in case mean filling failed)
    X_panel = X_panel.fillna(0)
    y_panel = y_panel.fillna(0)

    console.print(f"Created panel data with {n_series} series", style="blue")
    console.print(f"X_panel shape: {X_panel.shape}", style="blue")
    console.print(f"y_panel shape: {y_panel.shape}", style="blue")

    return X_panel, y_panel

# Example usage
if __name__ == "__main__":
    X_panel, y_panel = create_data(periods=120, freq='D', noise_level=0.5, n_series=5)

    # Display information about the panel data
    print("\nX_panel columns (sample):")
    print(X_panel.columns[:10])

    print("\ny_panel columns:")
    print(y_panel.columns)

    print("\nX_panel index (first 5):")
    print(X_panel.index[:5])

    print("\ny_panel index (first 5):")
    print(y_panel.index[:5])

# def run_test_example(seq_len=12, pred_len=6):
#     """Run the full forecasting pipeline with synthetic data.

#     Args:
#         seq_len (int): Lookback window size. Defaults to 12.
#         pred_len (int): Forecast horizon size. Defaults to 6.

#     Returns:
#         tuple: (recursive_forecaster, direct_forecaster, multioutput_forecaster,
#                recursive_pred, direct_pred, multioutput_pred)
#     """
#     print("Creating synthetic test data...")
#     X, y = create_data(periods=365, freq='D')

#     print("\nData shapes:")
#     print(f"X: {X.shape}, y: {y.shape}")

#     print("\nFeature columns:")
#     print(X.columns.tolist())




#     model_id = "linear"
#     class ModelParameters(BaseModel):
#         n_estimators: int = 100
#         criterion: str = "squared_error"
#         max_depth: int = 10
#         min_samples_leaf: int = 5
#         learning_rate: float = 0.1
#     model_params = ModelParameters()


#     print("\nRunning forecast (recursive strategy):")
#     forecaster = get_forecaster(
#         model_id=model_id,
#         strategy="direct",
#         seq_len=seq_len,
#         model_params=model_params
#     )
#     recursive_forecaster, recursive_pred = run_forecast(
#         X=X,
#         y=y,
#         forecaster=forecaster,
#         strategy="recursive",
#         test_size=0.3,
#         seq_len=seq_len,
#         pred_len=pred_len
#     )

#     print("\nRunning forecast (direct strategy):")
#     forecaster = get_forecaster(
#         model_id=model_id,
#         strategy="direct",
#         seq_len=seq_len,
#         model_params=model_params,
#     )
#     direct_forecaster, direct_pred = run_forecast(
#         X=X,
#         y=y,
#         forecaster=forecaster,
#         strategy="direct",
#         test_size=0.3,
#         seq_len=seq_len,
#         pred_len=pred_len
#     )

#     print("\nRunning forecast (multioutput strategy):")
#     forecaster = get_forecaster(
#         model_id=model_id,
#         strategy="multioutput",
#         seq_len=seq_len,
#         model_params=model_params
#     )
#     multioutput_forecaster, multioutput_pred = run_forecast(
#         X=X,
#         y=y,
#         forecaster=forecaster,
#         strategy="multioutput",
#         test_size=0.3,
#         seq_len=seq_len,
#         pred_len=pred_len
#     )

#     # Plot results
#     try:
#         import matplotlib.pyplot as plt

#         # Get test data for plotting
#         _, X_test, _, y_test, _ = prepare_data(X, y, pred_len, test_size=0.3)

#         print(f"y_test shape: {y_test.shape}")
#         print(f"X_test shape: {X_test.shape}")

#         plt.figure(figsize=(12, 6))
#         plt.plot(y_test.index[:pred_len], y_test.iloc[:pred_len], 'k-', linewidth=2, label='Actual')
#         plt.plot(recursive_pred.index, recursive_pred, 'b--', linewidth=2, label='(Recursive)')
#         plt.plot(direct_pred.index, direct_pred, 'r:', linewidth=2, label='(Direct)')
#         plt.plot(multioutput_pred.index, multioutput_pred, 'g-.', linewidth=2, label='(Multioutput)')
#         plt.legend()
#         plt.title('Forecasting Results')
#         plt.xlabel('Date')
#         plt.ylabel('Value')
#         plt.grid(True)
#         plt.tight_layout()
#         plt.show()

#     except ImportError:
#         print("\nMatplotlib not available for plotting. Install with: pip install matplotlib")

#     return (
#         recursive_forecaster, direct_forecaster, multioutput_forecaster,
#         recursive_pred, direct_pred, multioutput_pred
#     )

# # Run the test
# if __name__ == "__main__":
#     results = run_test_example(seq_len=12, pred_len=24)
#     # Unpack the results for further analysis if needed
#     recursive_model, direct_model, multioutput_model, recursive_pred, direct_pred, multioutput_pred = results
