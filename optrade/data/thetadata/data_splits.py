import os
import shutil
from typing import Optional
from optrade.data.thetadata.historical_options import get_historical_data

def find_optimal_tte(
    root: str = "AAPL",
    start_date: str = "20230407",
    target_tte: int = 30,
    tolerance: tuple = (25, 35),
    strike: int = 225,
    right: str = "C",
    save_dir: str = "../temp"
) -> Optional[int]:
    """
    Find the closest valid TTE to target_tte within tolerance range.

    Args:
        root: The root symbol of the underlying security
        start_date: The start date in YYYYMMDD format
        target_tte: Desired days to expiry (e.g., 30)
        tolerance: (min_tte, max_tte) acceptable range
        interval_min: The interval in minutes between data points
        strike: The strike price of the option in dollars
        right: The type of option, either 'C' for call or 'P' for put
        save_dir: Directory to save the data files

    Returns:
        Valid TTE if found, None otherwise
    """
    min_tte, max_tte = tolerance

    # Store valid TTEs and their distance from target
    valid_ttes = {}

    # First check target_tte itself
    try:
        get_historical_data(
            root=root,
            start_date=start_date,
            end_date=start_date,
            tte=target_tte,
            strike=strike,
            interval_min=60,
            right=right,
            save_dir=save_dir
        )
        return target_tte
    except:
        pass

    # Search outward from target_tte
    for offset in range(max(target_tte - min_tte, max_tte - target_tte)):
        # Try above target
        tte_above = target_tte + offset
        if min_tte <= tte_above <= max_tte:
            try:
                get_historical_data(
                    root=root,
                    start_date=start_date,
                    tte=tte_above,
                    strike=strike,
                    interval_min=1,
                    right=right,
                    save_dir=save_dir
                )
                valid_ttes[tte_above] = abs(tte_above - target_tte)
            except:
                pass

        # Try below target
        tte_below = target_tte - offset
        if min_tte <= tte_below <= max_tte:
            try:
                get_historical_data(
                    root=root,
                    start_date=start_date,
                    tte=tte_below,
                    strike=strike,
                    interval_min=1,
                    right=right,
                    save_dir=save_dir
                )
                valid_ttes[tte_below] = abs(tte_below - target_tte)
            except:
                pass

        # If we found any valid TTEs, return the closest to target
        if valid_ttes:
            return min(valid_ttes.items(), key=lambda x: x[1])[0]

    # Clean all files within save_dir
    try:
        # Remove the entire directory tree
        shutil.rmtree(save_dir)

        # Recreate empty directory
        os.makedirs(save_dir)
    except Exception as e:
            print(f"Warning: Could not clean directory {save_dir}: {str(e)}")

    # Throw an error if no valid TTEs found
    raise ValueError(f"No valid TTE found within tolerance range {tolerance}. Please try a wider tolerance band.")

if __name__ == "__main__":
    optimal_tte = find_optimal_tte(start_date="20241107", tolerance=(20, 40))
    print(f"Found valid TTE: {optimal_tte}")
