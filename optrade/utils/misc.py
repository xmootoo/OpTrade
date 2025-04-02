import pandas as pd
import random
import string
import time
import os
import torch
import numpy as np
from typing import Union


def format_time_dynamic(seconds):
    seconds = int(seconds)
    if seconds < 60:
        return f"{seconds}s"
    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}min {seconds}s"
    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}hr {minutes}min {seconds}s"
    days, hours = divmod(hours, 24)
    return f"{days}d {hours}hr {minutes}min {seconds}s"


def fill_open_zeros(group: pd.DataFrame) -> pd.DataFrame:
    """Fills any zero values in the 'mid_price' column with the first non-zero value that occurs
    before 9:35 AM within each group of data."""
    if group.iloc[0]["mid_price"] == 0:
        first_nonzero = group[
            (group["mid_price"] != 0)
            & (group["datetime"].dt.time <= pd.Timestamp("09:35").time())
        ]["mid_price"].iloc[0]
        group.loc[group["mid_price"] == 0, "mid_price"] = first_nonzero
    return group


def generate_random_id(length: int = 10):
    # Seed the random number generator with current time and os-specific random data
    random.seed(int(time.time() * 1000) ^ os.getpid())

    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def datetime_to_tensor(
    datetime_array: Union[str, np.datetime64], unit: str = "s"
) -> torch.Tensor:
    """
    Convert various datetime formats to PyTorch tensor of timestamps.

    Args:
        datetime_array: array-like of datetime values (can be strings, datetime objects, or datetime64)
        unit: time unit for conversion ('s' for seconds, 'ms' for milliseconds, etc.)

    Returns:
        PyTorch tensor of integer timestamps in the specified unit
    """
    timestamps = []

    for dt in datetime_array:
        try:
            # Handle string datetimes
            if isinstance(dt, str):
                dt = pd.to_datetime(dt)

            # Convert to numpy datetime64 if it's a Python datetime
            if not isinstance(dt, np.datetime64):
                dt = np.datetime64(dt)

            # Convert to the desired unit
            timestamp = dt.astype(f"datetime64[{unit}]").astype(np.int64)
            timestamps.append(timestamp)
        except Exception as e:
            print(f"Error converting datetime {dt}: {e}")
            # Use epoch as fallback
            timestamps.append(0)

    return torch.tensor(timestamps, dtype=torch.int64)


def tensor_to_datetime(
    timestamp_tensor: torch.Tensor,
    unit: str = "s",
    batch_mode: bool = False,
) -> np.ndarray:
    """
    Convert PyTorch tensor of timestamps back to numpy datetime64 array.

    Args:
        timestamp_tensor: PyTorch tensor of integer timestamps
        unit: time unit used in conversion ('s' for seconds, 'ms' for milliseconds,
              'us' for microseconds, 'ns' for nanoseconds)
        batch_mode: If True, handles tensor as batched (2D) data

    Returns:
        numpy array of datetime64 values
    """
    # Convert tensor to numpy array
    timestamp_array = timestamp_tensor.cpu().numpy()

    # NumPy requires a specific string format for conversion from timestamps
    epoch = np.datetime64("1970-01-01T00:00:00", unit)

    if batch_mode:
        # Handle batched data (2D array)
        result = []
        for batch in timestamp_array:
            batch_result = []
            for ts in batch:
                # Add the timestamp to the epoch
                dt = epoch + np.timedelta64(ts, unit)
                batch_result.append(dt)
            result.append(batch_result)
        out = np.array(result)
    else:
        # Handle non-batched data (1D array)
        result = []
        for ts in timestamp_array:
            # Add the timestamp to the epoch
            dt = epoch + np.timedelta64(ts, unit)
            result.append(dt)
        out = np.array(result)

    return out


# Test: datetime conversion functions
import pandas as pd
from datetime import datetime


def run_test():
    print("Test 1: Single datetime array conversion")
    # Create some datetime samples in different formats
    dt_samples = [
        np.datetime64("2023-01-01 12:00:00"),
        datetime(2023, 1, 2, 13, 30, 45),
        "2023-01-03 14:45:30",
        pd.Timestamp("2023-01-04 16:15:00"),
    ]
    print(f"Original sample: {dt_samples}")
    # Convert to tensor
    tensor = datetime_to_tensor(dt_samples)
    print(f"Tensor: {tensor}")
    # Convert back
    dt_array = tensor_to_datetime(tensor)
    print(f"Converted back: {dt_array}")
    print(f"Original type: {type(dt_samples[0])}, Converted type: {type(dt_array[0])}")

    print("\nTest 2: Batched datetime conversion")
    # Create a batched tensor (2 batches, 3 timestamps each)
    batch_1 = [
        np.datetime64("2023-01-05 09:00:00"),
        np.datetime64("2023-01-05 10:00:00"),
        np.datetime64("2023-01-05 11:00:00"),
    ]
    batch_2 = [
        np.datetime64("2023-01-06 09:00:00"),
        np.datetime64("2023-01-06 10:00:00"),
        np.datetime64("2023-01-06 11:00:00"),
    ]
    batched_samples = [batch_1, batch_2]

    # Convert each batch to tensor
    tensor_batch_1 = datetime_to_tensor(batch_1)
    tensor_batch_2 = datetime_to_tensor(batch_2)

    # Stack them into a batched tensor
    batched_tensor = torch.stack([tensor_batch_1, tensor_batch_2])
    print(f"Batched tensor shape: {batched_tensor.shape}")
    print(f"Batched tensor: {batched_tensor}")

    # Convert batched tensor back to datetime
    batched_dt = tensor_to_datetime(batched_tensor, batch_mode=True)
    print(f"Batched datetime array shape: {batched_dt.shape}")
    print(f"Batched datetime array: {batched_dt}")

    # Test full round-trip conversion for batched data
    print("\nTest 2b: Batched round-trip conversion")
    # Original datetime arrays
    print(f"Original batch 1: {batch_1}")
    print(f"Original batch 2: {batch_2}")

    # Convert to tensors and stack
    tensor_batch_1 = datetime_to_tensor(batch_1)
    tensor_batch_2 = datetime_to_tensor(batch_2)
    batched_tensor = torch.stack([tensor_batch_1, tensor_batch_2])
    print(f"Batched tensor: {batched_tensor}")

    # Convert back to datetime
    batched_dt = tensor_to_datetime(batched_tensor, batch_mode=True)
    print(f"Converted back: {batched_dt}")

    # Compare original and converted values
    print("Verification - original vs converted:")
    for i in range(2):
        for j in range(3):
            original = batched_samples[i][j]
            converted = batched_dt[i][j]
            equal = original == converted
            print(f"  Batch {i}, index {j}: {original} → {converted}, Equal: {equal}")

    print("\nTest 3: Real-world timestamp values")
    # Use timestamps similar to what appears in your data
    test_timestamps = torch.tensor(
        [[1673256600, 1673260200, 1673263800], [1674729000, 1674732600, 1674736200]]
    )
    print(f"Test timestamps shape: {test_timestamps.shape}")
    print(f"Test timestamps: {test_timestamps}")

    test_dt = tensor_to_datetime(test_timestamps, batch_mode=True)
    print(f"Converted datetime array: {test_dt}")

    # Convert back to timestamps to verify round-trip
    print("\nTest 3b: Real-world round-trip conversion")
    # Flatten the 2D datetime array for conversion
    test_dt_flat_1 = [dt for dt in test_dt[0]]
    test_dt_flat_2 = [dt for dt in test_dt[1]]

    # Convert back to tensors
    reconverted_tensor_1 = datetime_to_tensor(test_dt_flat_1)
    reconverted_tensor_2 = datetime_to_tensor(test_dt_flat_2)

    # Stack back to original shape
    reconverted_tensor = torch.stack([reconverted_tensor_1, reconverted_tensor_2])

    print(f"Original timestamps: {test_timestamps}")
    print(f"Reconverted timestamps: {reconverted_tensor}")
    print(f"Equal: {torch.all(test_timestamps == reconverted_tensor).item()}")


if __name__ == "__main__":
    run_test()
