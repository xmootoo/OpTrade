import os
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

# Custom modules
from optrade.data.thetadata.listings import get_expirations

def find_optimal_exp(
    root: str="AAPL",
    start_date: str="20230407",
    target_tte: int=30,
    tte_tolerance: Tuple[int, int]=(25, 35),
    clean_up: bool=True,
) -> Tuple[Optional[str], Optional[int]]:
    """
    Returns the closest valid TTE to target_tte within tolerance range and its expiration date.

    Args:
        root: The root symbol of the underlying security
        start_date: The start date in YYYYMMDD format
        target_tte: Desired days to expiry (e.g., 30)
        tte_tolerance: (min_tte, max_tte) acceptable range
        save_dir: Directory to save the data files

    Returns:
        Tuple of (Expiration date string 'YYYYMMDD', Optimal TTE) if found, (None, None) otherwise
    """
    min_tte, max_tte = tte_tolerance

    # Get expirations and convert to list of strings
    expirations = get_expirations(
        root=root,
        clean_up=clean_up,
    ).values.squeeze()

    # Convert start_date to datetime
    start_dt = datetime.strptime(start_date, "%Y%m%d")

    # Calculate TTEs for each expiration
    valid_pairs = []
    for exp in expirations:
        # Convert expiration to datetime
        exp_dt = datetime.strptime(str(exp), "%Y%m%d")

        # Calculate days to expiry
        tte = (exp_dt - start_dt).days

        # Check if TTE is within tolerance range
        if min_tte <= tte <= max_tte:
            valid_pairs.append((tte, exp))

    # If we found valid TTEs, return the one closest to target
    if valid_pairs:
        # Sort by distance to target TTE
        optimal_tte, optimal_exp = min(valid_pairs, key=lambda x: abs(x[0] - target_tte))
        return str(optimal_exp), optimal_tte
    else:
        raise ValueError(
            f"No valid TTE found within tolerance range {tte_tolerance}. "
            "Please try a wider tolerance band."
        )

if __name__ == "__main__":
    try:
        optimal_exp, optimal_tte = find_optimal_exp(
            start_date="20241107",
            tte_tolerance=(20, 40),
            target_tte=37,
        )
        print(f"Found valid TTE: {optimal_tte}")
        print(f"Expiration date: {optimal_exp}")
    except ValueError as e:
        print(f"Error: {e}")