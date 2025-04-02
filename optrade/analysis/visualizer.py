import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import Union, Tuple, Optional, List, Dict, Any
from sklearn.base import BaseEstimator
import torch.nn as nn

from optrade.data.forecasting import ForecastingDataset
from optrade.utils.misc import tensor_to_datetime, datetime_to_tensor
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error


class Analyzer:
    """
    Comprehensive analysis tool for model forecast performance evaluation.
    """

    def __init__(self):
        pass

    def period_visualize(
        self,
        period: str,
        period_interval: int,
        model: Union[nn.Module, BaseEstimator],
        dataset: Union[ForecastingDataset, np.ndarray],
        metrics: List[str],
        batch_size: int = 128,
        x_axis: str = "Time of Day",
        y_axis: str = "Normalized Error",
        title: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6),
        normalize: bool = False,
        use_secondary_axis: bool = False,
        dpi: int = 300,
        output_format: str = "png",
        save: bool = False,
    ) -> plt.Figure:
        """
        Visualize metrics aggregated by specific time periods across multiple days.
        Args:
            period (str): Type of period to group by. Options: "daily". Other periods not yet implemented.
            period_interval (int): If period="daily", period_interval represents the number of minutes to group by.
            model: Model object (PyTorch or scikit-learn)
            dataset: ForecastingDataset or numpy array of time series data
            metrics (List[str]): List of metrics to calculate ("mse", "mae", "rmse", "mape", "r^2")
            batch_size (int): Batch size for DataLoader
            x_axis (str): Label for x-axis
            y_axis (str): Label for y-axis
            title (Optional[str]): Plot title
            figsize (Tuple[int, int]): Figure size (width, height)
            normalize (bool): Whether to normalize metrics to [0,1] range for comparison
            use_secondary_axis (bool): Use a secondary y-axis for the second metric
            dpi (int): Dots per inch for image resolution
            output_format (str): Output format for saving the image
            save (bool): Save the plot to disk
        Returns:
            plt.Figure: The matplotlib figure object
        """
        # Get predictions and targets
        if isinstance(model, nn.Module):
            preds, targets, targets_dt = self.get_torch_preds(
                model, dataset, batch_size
            )
            # Flatten and convert targets_dt to pandas datetime
            preds = preds.reshape(-1)
            targets = targets.reshape(-1)
            targets_dt = pd.to_datetime(targets_dt.reshape(-1))
        else:
            raise NotImplementedError(
                "Only PyTorch models are supported for now. Scikit-learn will be implemented later."
            )

        # Extract the time component based on period_type
        if period == "daily":
            # Set intervals from 9:30 AM to 4:00 PM (trading hours). Stride by period_interval
            intervals = pd.date_range(
                "09:30", "16:00", freq=f"{period_interval}min"
            ).time

            # Create a dictionary to store metrics for each interval
            results = {}
            interval_labels = []

            # Iterate through each interval
            for i in range(len(intervals) - 1):
                start_time = intervals[i]
                end_time = intervals[i + 1]

                # Create a label for the interval
                interval_label = f"{start_time.strftime('%I:%M%p').lstrip('0').lower()}"
                interval_labels.append(interval_label)

                # Filter data for the current interval
                mask = (targets_dt.time >= start_time) & (targets_dt.time < end_time)
                interval_preds = preds[mask]
                interval_targets = targets[mask]

                # Skip empty intervals
                if len(interval_preds) == 0:
                    continue

                # Calculate metrics for this interval
                interval_metrics = self._calculate_metrics(
                    interval_preds, interval_targets, metrics
                )

                # Store results
                for metric_name, metric_value in interval_metrics.items():
                    if metric_name not in results:
                        results[metric_name] = []
                    results[metric_name].append(metric_value)
        else:
            raise ValueError(f"Unsupported period: {period}. Please use 'daily'.")

        # Normalize metrics if requested
        if normalize and len(results) > 0:
            for metric_name, metric_values in results.items():
                min_val = min(metric_values)
                max_val = max(metric_values)
                # Avoid division by zero
                if max_val > min_val:
                    results[metric_name] = [
                        (v - min_val) / (max_val - min_val) for v in metric_values
                    ]
                else:
                    # If all values are the same, set to 0.5 for visualization
                    results[metric_name] = [0.5 for _ in metric_values]

        # Set up the plot
        fig, ax = plt.subplots(figsize=figsize)

        # Secondary axis if requested and we have exactly 2 metrics
        ax2 = None
        if use_secondary_axis and len(results) == 2 and not normalize:
            ax2 = ax.twinx()

        # Plot each metric as a smooth line with markers
        colors = plt.cm.tab10.colors  # Use a colorful palette
        metric_names = list(results.keys())

        for i, (metric_name, metric_values) in enumerate(results.items()):
            color = colors[i % len(colors)]

            # Choose which axis to plot on
            plot_ax = ax
            if i == 1 and ax2 is not None:
                plot_ax = ax2
                plot_ax.set_ylabel(metric_names[1].upper(), color=color)
                plot_ax.tick_params(axis="y", labelcolor=color)

            # Plot with interpolation for smoother lines
            plot_ax.plot(
                range(len(interval_labels)),
                metric_values,
                "-",
                marker="o",
                markersize=6,
                linewidth=2.5,
                label=metric_name.upper(),
                color=color,
                alpha=0.7,
            )  # Semi-transparent to show overlaps

        # Add labels and title
        ax.set_xlabel(x_axis)
        if normalize:
            ax.set_ylabel("Normalized Error (0-1)")
        else:
            ax.set_ylabel(y_axis)

        if title:
            ax.set_title(title)
        else:
            if normalize:
                ax.set_title(
                    f"Normalized Performance Metrics by Time of Day (Interval: {period_interval} min)"
                )
            else:
                ax.set_title(
                    f"Performance Metrics by Time of Day (Interval: {period_interval} min)"
                )

        # Format x-axis for better readability - show labels at wider intervals
        skip_factor = max(
            1, int(len(interval_labels) / 6)
        )  # Show approximately 6 labels on x-axis
        ax.set_xticks(range(0, len(interval_labels), skip_factor))
        ax.set_xticklabels(
            [interval_labels[i] for i in range(0, len(interval_labels), skip_factor)],
            rotation=0,
        )

        # Add legend
        if ax2 is None:
            ax.legend(
                loc="best", framealpha=0.9, fancybox=True, shadow=True, fontsize=11
            )
        else:
            # For dual axis, combine legends from both axes
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(
                lines1 + lines2,
                labels1 + labels2,
                loc="best",
                framealpha=0.9,
                fancybox=True,
                shadow=True,
                fontsize=11,
            )

        # Add grid for better readability of values
        ax.grid(True, linestyle="--", alpha=0.6)

        # Style the plot to look more like a financial chart
        ax.spines["top"].set_visible(False)
        if ax2 is None:
            ax.spines["right"].set_visible(False)

        plt.tight_layout()

        # Save if requested
        if save:
            plt.savefig(f"period_visualize.{output_format}", dpi=dpi)

        return fig

    def _calculate_metrics(
        self, preds: np.ndarray, targets: np.ndarray, metrics: List[str]
    ) -> Dict[str, float]:
        """
        Calculate the specified error metric between predictions and targets.

        Args:
            preds (np.ndarray): Model predictions of shape (num_examples, pred_len)
            targets (np.ndarray): Ground truth values of shape (num_examples, pred_len)
            metric (str): Metric to calculate ("mse", "mae", "rmse")

        Returns:
            np.ndarray: Error values with same shape as inputs
        """
        metrics = list(map(str.lower, metrics))

        metric_values = dict()

        if "mse" in metrics:
            metric_values["mse"] = np.mean((preds - targets) ** 2)
        if "mae" in metrics:
            metric_values["mae"] = np.mean(np.abs(preds - targets))
        if "rmse" in metrics:
            metric_values["rmse"] = np.sqrt(np.mean((preds - targets) ** 2))
        if "mape" in metrics:
            epsilon = 1e-10
            metric_values["mape"] = np.mean(
                np.abs((preds - targets) / (targets + epsilon))
            )
        if "smape" in metrics:
            epsilon = 1e-10
            metric_values["smape"] = np.mean(
                2
                * np.abs(preds - targets)
                / (np.abs(preds) + np.abs(targets) + epsilon)
            )
        if "r^2" in metrics:
            ssr = ((preds - targets) ** 2).sum()
            sst = ((targets - targets.mean()) ** 2).sum()
            metric_values["r^2"] = 1 - ssr / sst

        assert (
            len(metric_values) > 0
        ), f"Invalid metrics: {metrics}. Supported metrics: mse, mae, rmse, mape, r^2"

        return metric_values

    def information_coefficient_analysis(
        self, forward_periods: List[int] = [1, 5, 10, 20], rolling_window: int = 20
    ):
        """
        Calculate IC (Information Coefficient) for different forward periods

        Mathematical explanation:
        - Forward Period: Number of time steps ahead that the prediction is targeting
        - IC: Spearman rank correlation between predictions and actual future values
          IC = corr_spearman(prediction_t, actual_{t+forward_period})
        - IC IR (Information Coefficient Information Ratio):
          The ratio of mean IC to standard deviation of IC over time
          IC IR = mean(IC) / std(IC)

        Why it's relevant for forecasting and alpha research:
        - IC measures how well your model ranks outcomes (essential for relative value strategies)
        - IC across different horizons shows decay pattern of your signal
        - IC IR quantifies signal-to-noise ratio - a high IC IR indicates consistent predictive power
        - IC > 0.05 is often considered meaningful in practice for daily forecasts
        - IC analysis helps determine optimal holding periods and trading frequency

        Args:
            forward_periods: List of forward periods to analyze
            rolling_window: Window size for calculating IC IR

        Returns:
            DataFrame with IC metrics by forward period
        """
        ic_metrics = []

        for period in forward_periods:
            # Shift target to align with prediction
            forward_target = self.data["target"].shift(-period)

            # Calculate overall IC - Spearman rank correlation
            ic = stats.spearmanr(
                self.data["prediction"], forward_target, nan_policy="omit"
            )[0]

            # Calculate rolling IC for IR calculation
            self.data[f"rolling_ic_{period}"] = (
                self.data["prediction"]
                .rolling(window=rolling_window)
                .corr(forward_target, method="spearman")
            )

            # Calculate IC IR (Information Ratio of the IC)
            # Mean IC divided by standard deviation of IC
            mean_rolling_ic = self.data[f"rolling_ic_{period}"].mean()
            std_rolling_ic = self.data[f"rolling_ic_{period}"].std()

            # Handle division by zero
            ic_ir = mean_rolling_ic / std_rolling_ic if std_rolling_ic > 0 else np.nan

            # Get time-series stats on the rolling IC
            ic_positive_rate = (self.data[f"rolling_ic_{period}"] > 0).mean()
            ic_significant_rate = (self.data[f"rolling_ic_{period}"] > 0.05).mean()

            ic_metrics.append(
                {
                    "forward_period": period,
                    "ic": ic,
                    "mean_rolling_ic": mean_rolling_ic,
                    "std_rolling_ic": std_rolling_ic,
                    "ic_ir": ic_ir,
                    "ic_positive_rate": ic_positive_rate,
                    "ic_significant_rate": ic_significant_rate,
                }
            )

        return pd.DataFrame(ic_metrics)

    def error_autocorrelation_analysis(self, lags=20, plot=True):
        """
        Analyze autocorrelation in prediction errors

        Args:
            lags: Number of lags to analyze
            plot: Whether to generate visualization

        Returns:
            Series with autocorrelation values by lag
        """
        error_acf = acf(self.data["error"].dropna(), nlags=lags)

        if plot:
            plt.figure(figsize=(12, 6))
            plt.bar(range(len(error_acf)), error_acf)
            plt.axhline(y=0, color="r", linestyle="-", alpha=0.3)
            plt.axhline(
                y=1.96 / np.sqrt(len(self.data)), color="k", linestyle="--", alpha=0.3
            )
            plt.axhline(
                y=-1.96 / np.sqrt(len(self.data)), color="k", linestyle="--", alpha=0.3
            )
            plt.xlabel("Lag")
            plt.ylabel("Autocorrelation")
            plt.title("Error Autocorrelation Function")
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        return pd.Series(error_acf, index=range(len(error_acf)))

    def event_study_analysis(self, event_dates, window=5):
        """
        Analyze model performance around specific market events

        Args:
            event_dates: List of event dates
            window: Window size around events to analyze

        Returns:
            DataFrame with performance metrics around events
        """
        event_metrics = []

        for event_date in event_dates:
            # Filter data around event
            event_window = self.data.loc[
                (self.data.index >= event_date - pd.Timedelta(days=window))
                & (self.data.index <= event_date + pd.Timedelta(days=window))
            ]

            # Calculate relative position to event
            event_window = event_window.copy()
            event_window["event_day"] = (
                event_window.index - pd.to_datetime(event_date)
            ).days

            # Group by relative day
            for day, group in event_window.groupby("event_day"):
                if len(group) > 0:
                    mse = mean_squared_error(group["target"], group["prediction"])
                    mae = mean_absolute_error(group["target"], group["prediction"])
                    ic = stats.spearmanr(group["prediction"], group["target"])[0]

                    event_metrics.append(
                        {
                            "event_date": event_date,
                            "event_day": day,
                            "mse": mse,
                            "mae": mae,
                            "ic": ic,
                        }
                    )

        return pd.DataFrame(event_metrics)

    def analyze_forecast_features(self, features=None, n_bins=10, plot=True):
        """
        Analyze model performance conditional on input feature values
        This is especially useful for alpha research to understand what market
        conditions lead to better or worse predictions

        Args:
            features: List of feature columns to analyze, if None use all available columns
            n_bins: Number of quantile bins to divide feature values into
            plot: Whether to generate visualizations

        Returns:
            Dict of DataFrames with performance metrics by feature quantiles
        """
        if features is None:
            # Use all numeric columns except target, prediction, and computed metrics
            exclude_cols = [
                "target",
                "prediction",
                "error",
                "abs_error",
                "squared_error",
                "pct_error",
                "hour",
                "minute",
                "day_of_week",
            ]
            features = [
                col
                for col in self.data.columns
                if col not in exclude_cols
                and np.issubdtype(self.data[col].dtype, np.number)
            ]

        results = {}

        for feature in features:
            if feature not in self.data.columns:
                continue

            # Create quantile bins
            self.data[f"{feature}_bin"] = pd.qcut(
                self.data[feature].fillna(self.data[feature].median()),
                q=n_bins,
                duplicates="drop",
            )

            # Group by feature bins
            feature_metrics = []

            for bin_val, group in self.data.groupby(f"{feature}_bin"):
                if len(group) > 10:  # Ensure enough samples
                    mse = mean_squared_error(group["target"], group["prediction"])
                    mae = mean_absolute_error(group["target"], group["prediction"])
                    ic = stats.spearmanr(group["prediction"], group["target"])[0]
                    dir_acc = np.mean(
                        np.sign(group["prediction"]) == np.sign(group["target"])
                    )

                    feature_metrics.append(
                        {
                            "feature": feature,
                            "bin": bin_val,
                            "bin_center": bin_val.mid,
                            "mse": mse,
                            "mae": mae,
                            "ic": ic,
                            "dir_acc": dir_acc,
                            "samples": len(group),
                        }
                    )

            # Convert to DataFrame
            feature_df = pd.DataFrame(feature_metrics).sort_values("bin_center")
            results[feature] = feature_df

            # Generate plot if requested
            if plot and not feature_df.empty:
                fig, ax1 = plt.subplots(figsize=(10, 6))

                # Plot MSE on left axis
                ax1.plot(
                    range(len(feature_df)),
                    feature_df["mse"],
                    "o-",
                    color="blue",
                    label="MSE",
                )
                ax1.set_xlabel(feature)
                ax1.set_ylabel("MSE", color="blue")
                ax1.tick_params(axis="y", labelcolor="blue")

                # Plot IC on right axis
                ax2 = ax1.twinx()
                ax2.plot(
                    range(len(feature_df)),
                    feature_df["ic"],
                    "o-",
                    color="red",
                    label="IC",
                )
                ax2.set_ylabel("IC", color="red")
                ax2.tick_params(axis="y", labelcolor="red")

                # Set x-ticks
                bin_labels = [f"{b.left:.2f}-{b.right:.2f}" for b in feature_df["bin"]]
                plt.xticks(range(len(feature_df)), bin_labels, rotation=45)

                plt.title(f"Model Performance by {feature} Quantiles")
                plt.tight_layout()
                plt.show()

        return results

    def get_torch_preds(
        self,
        model: nn.Module,
        dataset: Any,  # ForecastingDataset
        batch_size: int = 128,
        channel: int = 0,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Generate predictions from a PyTorch model and ForecastingDataset.
        Args:
            model (nn.Module): PyTorch model
            dataset (ForecastingDataset): ForecastingDataset object
            batch_size (int): Batch size for DataLoader
            device (str): Device to run model on ('cuda' or 'cpu')
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                (predictions, targets, target_datetimes)
        """
        all_preds = []
        all_targets = []
        all_targets_dt = []
        loader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, drop_last=False
        )

        # Set model to evaluation mode
        model.eval()

        with torch.no_grad():
            for batch in loader:
                x, y, x_dt, y_dt = (
                    batch  # x: (batch_size, num_feats, seq_len), y: (batch_size, num_feats, pred_len)
                )

                # Add batch dimension if needed
                if x.dim() == 2:
                    x = x.unsqueeze(0)
                    y = y.unsqueeze(0)
                    x_dt = x_dt.unsqueeze(0)
                    y_dt = y_dt.unsqueeze(0)

                # Compute predictions
                x = x.to(device)
                preds = model(x)  # preds: (batch_size, num_feats, pred_len)

                # Append to lists
                all_preds.append(preds)
                all_targets.append(y)
                all_targets_dt.append(y_dt)

        # Stack all predictions, targets, and datetimes
        preds = torch.cat(all_preds).detach().cpu().numpy()
        targets = torch.cat(all_targets).detach().cpu().numpy()
        targets_dt = tensor_to_datetime(
            timestamp_tensor=torch.cat(all_targets_dt).detach().cpu(), batch_mode=True
        )

        # Select the target channel
        preds = preds[:, channel, :]
        targets = targets[:, channel, :]

        return preds, targets, targets_dt


if __name__ == "__main__":
    # Step 1: Find and initialize the optimal contract
    from optrade.data.contracts import Contract

    contract = Contract.find_optimal(
        root="AAPL",
        right="C",  # Call option
        start_date="20230103",  # First trading day of 2023
        target_tte=30,  # Desired expiration: 30 days
        tte_tolerance=(20, 40),  # Min 20, max 40 days expiration
        interval_min=1,  # Data requested at 1-min level
        moneyness="ATM",  # At-the-money option
        dev_mode=True,
    )

    # Step 2: Load market data (NBBO quotes and OHLCV)
    df = contract.load_data(dev_mode=True)

    # Step 3: Transform raw data into ML-ready features
    from optrade.data.features import transform_features

    core_feats = [
        "option_returns",  # Option price returns
        "stock_returns",  # Underlying stock returns
        "moneyness",  # Log(S/K)
        "option_lob_imbalance",  # Order book imbalance
        "stock_quote_spread",  # Bid-ask spread normalized
    ]
    tte_feats = ["sqrt", "exp_decay"]  # Time-to-expiration features
    datetime_feats = ["minute_of_day", "hour_of_week"]

    data = transform_features(
        df=df,
        core_feats=core_feats,
        tte_feats=tte_feats,  # Time-to-expiration features
        datetime_feats=datetime_feats,  # Time features
        strike=contract.strike,
        exp=contract.exp,
        keep_datetime=True,
    )

    # Step 4: Create dataset for time series forecasting
    from optrade.data.forecasting import ForecastingDataset
    from torch.utils.data import DataLoader

    target_channels = ["option_returns"]
    torch_dataset = ForecastingDataset(
        data=data,
        seq_len=100,  # 100-minute lookback window
        pred_len=10,  # 10-minute forecast horizon
        target_channels=target_channels,
    )

    # Test the _get_torch_predictions method
    import torch.nn as nn

    model = nn.Linear(100, 10)

    # Randomly initialize weights
    with torch.no_grad():
        model.weight = nn.Parameter(torch.randn_like(model.weight))
        model.bias = nn.Parameter(torch.randn_like(model.bias))

    visualizer = Analyzer()

    # Test period_visualize
    fig = visualizer.period_visualize(
        period="daily",
        period_interval=5,
        model=model,
        dataset=torch_dataset,
        metrics=["mse", "mae", "r^2", "mape"],
        batch_size=128,
        x_axis="Time (min)",
        y_axis="MSE & MAE",
        title="Model Performance by Time of Day",
        figsize=(12, 6),
        dpi=300,
        save=False,
        normalize=True,
    )
    plt.show()
