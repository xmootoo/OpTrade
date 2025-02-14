import os
from token import OP
from typing import Optional, Tuple
from pathlib import Path
from datetime import datetime

# Custom modules
from optrade.data.thetadata.listings import get_expirations

def find_optimal_tte(
    root: str = "AAPL",
    start_date: str = "20230407",
    target_tte: int = 30,
    tolerance: Optional[tuple] = (25, 35),
    save_dir: str = "../temp"
) -> Tuple[Optional[int], Optional[str]]:
    """
    Returns the closest valid TTE to target_tte within tolerance range and its expiration date.

    Args:
        root: The root symbol of the underlying security
        start_date: The start date in YYYYMMDD format
        target_tte: Desired days to expiry (e.g., 30)
        tolerance: (min_tte, max_tte) acceptable range
        save_dir: Directory to save the data files

    Returns:
        Tuple of (Valid TTE, Expiration date string) if found, (None, None) otherwise
    """
    script_dir = Path(__file__).parent  # Get the directory containing the current script
    expirations_dir = script_dir.parent / "historical_data/expirations"
    min_tte, max_tte = tolerance

    # Get expirations and convert to list of strings
    expirations = get_expirations(
        root=root,
        save_dir=expirations_dir,
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
        return optimal_tte, str(optimal_exp)
    else:
        raise ValueError(
            f"No valid TTE found within tolerance range {tolerance}. "
            "Please try a wider tolerance band."
        )

if __name__ == "__main__":
    try:
        optimal_tte, optimal_exp = find_optimal_tte(
            start_date="20241107",
            tolerance=(20, 40),
            target_tte=37,
        )
        print(f"Found valid TTE: {optimal_tte}")
        print(f"Expiration date: {optimal_exp}")
    except ValueError as e:
        print(f"Error: {e}")
