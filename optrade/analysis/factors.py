import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
from datetime import datetime
import warnings
import numpy as np
from typing import Dict, Any, Optional, Union, List

warnings.filterwarnings("ignore", message="The argument 'date_parser' is deprecated")

# Custom modules
from optrade.data.thetadata import load_stock_data_eod


def get_factor_exposures(
    root: str,
    start_date: str,
    end_date: str,
    mode: str = "ff3",
) -> Dict[str, Any]:
    """
    Calculate factor model exposures for a stock over the specified period.
    Supports Fama-French 3-factor (ff3), Fama-French 5-factor (ff5), and Carhart 4-factor (c4) models.

    Args:
        root (str): Root symbol of the underlying security
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format
        mode (str): Mode for the factor model. Options: "ff3" (Fama-French 3 factor), "ff5" (Fama-French 5 factor), or "c4" (Carhart 4 factor).

    Returns:
        Dictionary containing the factor betas:
        - market_beta: Market excess return sensitivity
        - size_beta: Small Minus Big (SMB) factor exposure
        - value_beta: High Minus Low (HML) book-to-market factor exposure
        - momentum_beta: Winners Minus Losers (WML) momentum factor (Carhart model only)
        - profitability_beta: Robust Minus Weak (RMW) profitability factor (5-factor only)
        - investment_beta: Conservative Minus Aggressive (CMA) investment factor (5-factor only)
        - r_squared: Proportion of return variation explained by the factors
    """

    # Suppress the date_parser deprecation warning
    warnings.filterwarnings(
        "ignore", message="The argument 'date_parser' is deprecated"
    )

    # Convert date strings to datetime objects
    factor_start_date = datetime.strptime(start_date, "%Y%m%d")
    factor_end_date = datetime.strptime(end_date, "%Y%m%d")

    # Get stock data
    stock_data = load_stock_data_eod(
        root=root,
        start_date=start_date,
        end_date=end_date,
        clean_up=True,
        offline=False,
    )

    # Calculate daily returns
    stock_data["returns"] = stock_data["close"].pct_change().dropna()
    stock_data["Date"] = stock_data["datetime"].dt.date

    # Drop NaN
    stock_data = stock_data.dropna()

    # Drop all other columns besides Date and returns
    stock_data = stock_data[["Date", "returns"]]

    # Get factor data based on mode
    if mode == "ff3":
        factor_data = web.DataReader(
            "F-F_Research_Data_Factors_daily",
            "famafrench",
            start=factor_start_date,
            end=factor_end_date,
        )[0]
        factor_columns = ["Mkt-RF", "SMB", "HML"]
    elif mode == "ff5":
        factor_data = web.DataReader(
            "F-F_Research_Data_5_Factors_2x3_daily",
            "famafrench",
            start=factor_start_date,
            end=factor_end_date,
        )[0]
        factor_columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    elif mode == "c4":
        # Get FF 3-factor data
        factor_data = web.DataReader(
            "F-F_Research_Data_Factors_daily",
            "famafrench",
            start=factor_start_date,
            end=factor_end_date,
        )[0]

        # Get momentum factor data - note the column name has trailing spaces
        mom_data = web.DataReader(
            "F-F_Momentum_Factor_daily",
            "famafrench",
            start=factor_start_date,
            end=factor_end_date,
        )[0]

        # Fix the column name to remove trailing spaces
        mom_data.columns = [col.strip() for col in mom_data.columns]

        # Merge FF 3-factor with momentum
        factor_data = pd.merge(factor_data, mom_data, left_index=True, right_index=True)
        factor_columns = ["Mkt-RF", "SMB", "HML", "Mom"]
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose 'ff3', 'c4', or 'ff5'.")

    # Convert percentages to decimals
    factor_data = factor_data / 100

    # Truncate factor_data to the same date range as stock_data["Date"]
    valid_dates = pd.DatetimeIndex(stock_data["Date"])
    factor_data = factor_data.loc[factor_data.index.intersection(valid_dates)]

    # Reset index to make Date a column
    factor_data_reset = factor_data.reset_index()
    factor_data_reset["Date"] = factor_data_reset["Date"].dt.date

    # Merge stock_data with factor_data on Date
    aligned_data = pd.merge(stock_data, factor_data_reset, on="Date", how="inner")

    # Linear regression
    X = aligned_data[factor_columns]
    X = sm.add_constant(X)
    y = aligned_data["returns"] - aligned_data["RF"]  # Excess return

    # Ensure y is 1-dimensional
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Run regression
    model = sm.OLS(y, X).fit()

    # Prepare results
    result = {
        "market_beta": model.params.get("Mkt-RF", None),
        "size_beta": model.params.get("SMB", None),
        "value_beta": model.params.get("HML", None),
        "r_squared": model.rsquared,
    }

    # Add momentum for Carhart model
    if mode == "c4":
        result["momentum_beta"] = model.params.get("Mom", None)

    # Add additional factors for 5-factor model
    if mode == "ff5":
        result["profitability_beta"] = model.params.get("RMW", None)
        result["investment_beta"] = model.params.get("CMA", None)

    return result


def factor_categorization(
    factors: Dict[str, Dict[str, float]], mode: str = "ff3"
) -> Dict[str, Dict[str, str]]:
    """
    Categorize stocks based on their factor model exposures using percentiles.

    Args:
        factors: Nested dictionary where:
            - Outer key is the root symbol
            - Inner key is the factor type
            - Value is the factor beta
        mode: Factor model type ("ff3", "ff5", or "c4")

    Returns:
        Nested dictionary with categorizations for each stock and factor
    """
    # Define factor sets for each model
    model_factors = {
        "ff3": {"market_beta", "size_beta", "value_beta"},
        "ff5": {
            "market_beta",
            "size_beta",
            "value_beta",
            "profitability_beta",
            "investment_beta",
        },
        "c4": {"market_beta", "size_beta", "value_beta", "momentum_beta"},
    }

    # Get relevant factors for this mode
    relevant_factors = model_factors[mode]

    # Define factor category mappings
    factor_mappings = {
        "market_beta": {"high": "high", "low": "low", "neutral": "neutral"},
        "size_beta": {"high": "small_cap", "low": "large_cap", "neutral": "neutral"},
        "value_beta": {"high": "value", "low": "growth", "neutral": "neutral"},
        "momentum_beta": {"high": "high", "low": "low", "neutral": "neutral"},
        "profitability_beta": {"high": "robust", "low": "weak", "neutral": "neutral"},
        "investment_beta": {
            "high": "conservative",
            "low": "aggressive",
            "neutral": "neutral",
        },
    }

    # Calculate percentiles for each relevant factor
    percentiles = {}
    for factor_type in relevant_factors:
        # Extract values, ignoring None
        values = [
            f[factor_type]
            for f in factors.values()
            if factor_type in f and f[factor_type] is not None
        ]

        if values:
            percentiles[factor_type] = [
                np.percentile(values, 30),  # 30th percentile
                np.percentile(values, 70),  # 70th percentile
            ]

    # Function to categorize a single factor value
    def categorize_factor(factor_type, value):
        # Special case for market beta
        if factor_type == "market_beta":
            if value > 1.1:
                return "high"
            elif value < 0.9:
                return "low"
            else:
                return "neutral"

        # For other factors, use percentiles if available
        if factor_type in percentiles:
            if value > percentiles[factor_type][1]:
                return factor_mappings[factor_type]["high"]
            elif value < percentiles[factor_type][0]:
                return factor_mappings[factor_type]["low"]

        # Default case
        return factor_mappings[factor_type]["neutral"]

    # Build the categorization result
    result = {}
    for root, root_factors in factors.items():
        result[root] = {}

        for factor_type in relevant_factors:
            if factor_type in root_factors and root_factors[factor_type] is not None:
                result[root][factor_type] = categorize_factor(
                    factor_type, root_factors[factor_type]
                )

    return result


# Function to calculate factor exposures for multiple stocks
def get_universe_factor_exposures(
    roots: List[str], start_date: str, end_date: str, mode: str = "ff3"
) -> Dict[str, Dict[str, float]]:
    """
    Calculate factor model exposures for multiple stocks over the specified period.

    Args:
        roots: List of stock roots to analyze
        start_date: Start date in YYYYMMDD format
        end_date: End date in YYYYMMDD format
        mode: Factor model to use ("ff3", "ff5", or "c4")

    Returns:
        Nested dictionary where:
        - Outer key is the root symbol
        - Inner key is the factor type
        - Value is the factor beta
    """
    # Collect factor betas for all stocks
    all_factors = {}

    for root in roots:
        try:
            # Get factor exposures for the stock
            factors = get_factor_exposures(
                root=root, start_date=start_date, end_date=end_date, mode=mode
            )
            all_factors[symbol] = factors
        except Exception as e:
            all_factors[symbol] = None
            print(f"Error processing {symbol}: {e}")
            continue

    return all_factors


# Example usage
if __name__ == "__main__":
    from rich.console import Console

    ctx = Console()

    # Set test period (1 year)
    start_date = "20230101"  # YYYYMMDD format
    end_date = "20231231"  # YYYYMMDD format

    # Define a sample universe of stocks
    sample_universe = [
        "AAPL",
        "MSFT",
        "AMZN",
        "GOOGL",
        "META",
        "TSLA",
        "NVDA",
        "JPM",
        "V",
        "PG",
    ]

    print(
        f"Testing Carhart 4-factor model for sample universe from {start_date} to {end_date}"
    )

    # Calculate factor exposures for all stocks in the universe
    universe_factors = get_universe_factor_exposures(
        sample_universe, start_date, end_date, mode="c4"
    )

    # Categorize the stocks based on their factor exposures
    universe_categorization = factor_categorization(universe_factors, mode="c4")
    print("\nStock Factor Categorization and Values:")
    for symbol, factors in universe_categorization.items():
        ctx.log(f"{symbol}: {factors}")
    print("\nFactor Exposures:")
    for symbol, factors in universe_factors.items():
        ctx.log(f"{symbol}: {factors}")
