import numpy as np
import pandas as pd
from typing import List
from datetime import datetime

def get_tte_features(
    df: pd.DataFrame,
    feats: List=[
        "tte",
        "inverse",
        "sqrt",
        "inverse_sqrt",
        "exp_decay"
    ],
    exp: str="20250117",
) -> pd.DataFrame:
    """
    Generate Time to Expiration (TTE) features for a given DataFrame.

    Args:
        df (pd.DataFrame): DataFrame containing datetime column in format 'YYYY-MM-DD HH:MM:SS'.
                           The function will try to identify a datetime column if not explicitly named 'datetime'.
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
                     feature will be added with a prefix 'tte_' (e.g., 'tte_inverse').
                     All TTE features are guaranteed to be float64 type.
    """
    if feats==[]:
        return df

    # Create a copy to avoid modifying the original
    result_df = df.copy()

    # Convert expiration date string to datetime
    exp_date = datetime.strptime(exp, '%Y%m%d')
    # Set expiration time to 4:30 PM on expiration date
    exp_datetime = exp_date.replace(hour=16, minute=30, second=0)

    # Find the datetime column - standardized approach
    # First check if 'datetime' column exists
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

    # Ensure column is datetime type - using pandas' robust datetime conversion
    if not pd.api.types.is_datetime64_any_dtype(df[dt_col]):
        result_df[dt_col] = pd.to_datetime(df[dt_col], errors='coerce')

    # Calculate TTE in minutes as float64
    result_df['tte_minutes'] = (exp_datetime - result_df[dt_col]).dt.total_seconds().astype('float64') / 60

    # Calculate maximum TTE (contract length in minutes)
    contract_length = result_df['tte_minutes'].max()

    # Generate requested features
    if "linear" in feats or "all" in feats:
        # Linear TTE (raw minutes)
        result_df['tte'] = result_df['tte_minutes'].astype('float64')

    if "inverse" in feats or "all" in feats:
        # Inverse TTE (1/minutes)
        # Handle potential division by zero with np.inf handling
        result_df['tte_inverse'] = np.where(
            result_df['tte_minutes'] > 0,
            1 / result_df['tte_minutes'],
            np.inf
        ).astype('float64')

    if "sqrt" in feats or "all" in feats:
        # Square root of TTE
        result_df['tte_sqrt'] = np.sqrt(result_df['tte_minutes']).astype('float64')

    if "inverse_sqrt" in feats or "all" in feats:
        # Inverse square root of TTE
        # Handle potential division by zero
        result_df['tte_inverse_sqrt'] = np.where(
            result_df['tte_minutes'] > 0,
            1 / np.sqrt(result_df['tte_minutes']),
            np.inf
        ).astype('float64')

    if "exp_decay" in feats or "all" in feats:
        # Exponential decay with lambda = 1/contract_length
        result_df['tte_exp_decay'] = np.exp(-result_df['tte_minutes'] / contract_length).astype('float64')

    # Remove intermediate calculation if not requested
    if "linear" not in feats and "all" not in feats:
        result_df = result_df.drop('tte_minutes', axis=1)
    else:
        # If we're keeping tte_minutes, ensure it's float64
        result_df['tte_minutes'] = result_df['tte_minutes'].astype('float64')

    return result_df

# Example usage
if __name__ == "__main__":
    # Create sample data
    from optrade.data.thetadata.get_data import load_all_data
    from optrade.data.thetadata.contracts import Contract
    from rich.console import Console
    console = Console()

    contract = Contract(
        root="AAPL",
        start_date="20241107",
        end_date="20241108",
        exp="20250117",
        strike=225,
        interval_min=1,
        right="C",
    )

    df = load_all_data(
        contract=contract,
        clean_up=True,
        offline=False
    )

    # Generate TTE features
    result = get_tte_features(
        df=df,
        feats=["tte", "inverse", "sqrt", "inverse_sqrt", "exp_decay"],
        exp="20250117"
    )

    # Display results with dtypes to verify float64
    console.log(result[['datetime', 'option_mid_price'] + [col for col in result.columns if col.startswith('tte_')]].head())
    print("\nColumn dtypes:")
    for col in result.columns:
        if col.startswith('tte_'):
            print(f"{col}: {result[col].dtype}")
