import os
import gc
from itertools import chain
import pickle
import random
from collections import Counter
import statistics as stats
from datetime import datetime

import torch
from torch.functional import _return_counts
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

import pandas as pd
import numpy as np

# Dataset Classes
from optrade.src.utils.data.dataset import ForecastingDataset, UnivariateForecastingDataset, ClassificationDataset

# Cyclical Features (datetime encoding)
from time_series.utils.sincos_pos_emb import CyclicalFeatureEncoder

# Normalization, Preprocessing, Dataloading
from sklearn.preprocessing import StandardScaler
from scipy.io import loadmat
from scipy import interpolate

#<-----------------Forecasting------------------------>
def load_forecasting(
    dataset_name="ETTh1",
    seq_len=512,
    pred_len=96,
    window_stride=1,
    scale=True,
    train_split=0.7,
    val_split=0.1,
    univariate=False,
    resizing_mode="None",
    single_channel=False,
    target_channel=0,
    clip=False,
    thresh=1.0,
    date=False,
    full_channels=False,
    differencing=0,
    datetime_numeric=False,
    cyclical_encoding=False,
    datetime_features=["hour", "day"],
    average_turkey=False,
    average_italy=False):
    """
    Args:
        dataset_name (str): Name of the dataset to load. Options: "ETTh1", "ETTh2", "ETTm1", "ETTm2", "weather", "electricity", "traffic", "illness".
        seq_len (int): Length of the input sequence.
        pred_len (int): Length of the prediction sequence.
        window_stride (int): Stride of the window for sliding window sampling (if univariate is True).
        scale (bool): Whether to normalize the data.
        train_split (float): Fraction of the data to use for training.
        val_split (float): Fraction of the data to use for validation.
        univariate (bool): Whether to use univariate or multivariate data, this processes the data differently and creates windows before
                           being input into the dataloader, so as to allow appropriate separation between the channels.
    Returns:
        train_data (np.array): Training data of shape (num_channels, train_len).
        val_data (np.array): Validation data of shape (num_channels, val_len).
        test_data (np.array): Test data of shape (num_channels, test_len).
    """

    # Load data. NumPy array of shape (seq_len, num_channels)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.dirname(os.path.dirname(current_dir))

    if dataset_name == "rf_emf":
        data = loadmat(os.path.join(root_dir, "data/forecasting/rf_emf/LongTerm17.mat"))
        data = data["York_LongTerm17"].T
        data = time_window_average(data=data,m=24) if average_turkey else data # Return 6min average (24hrs, 15s sampling rate, 5760 seq_len)

    elif dataset_name in {"rf_emf_det", "rf_emf_det2"}:
        file_name = "DET_Monitoring.csv" if dataset_name == "rf_emf_det" else "DET_Monitoring2.csv" # DET1 = 2022, DET2 = 2023
        data = pd.read_csv(os.path.join(root_dir, "data/forecasting/rf_emf2", file_name)).to_numpy()
        data[:, 1] = clip_and_interpolate(data[:, 1], thresh) if clip else data[:, 1]
        data = data[:, 1].reshape(-1,1) if not date else data

    elif dataset_name in {"rf_emf_ptv", "rf_emf_ptv2"}:
        file_name = "PTV_Monitoring.csv" if dataset_name == "rf_emf_ptv" else "PTV_Monitoring2.csv" # PTV1 = 2022, PTV2 = 2023
        data = pd.read_csv(os.path.join(root_dir, "data/forecasting/rf_emf2", file_name)).to_numpy()

        cutoff = 85251 if dataset_name == "rf_emf_ptv2" else -1 # All values after 85251 after for rf_emf_ptv2 are NaN
        data = data[:cutoff]
        data[:, 1] = data[:, 1].astype(np.float64)
        data[:, 1] = clip_and_interpolate(data[:, 1], thresh) if clip else data[:, 1]
        data = data[:, 1].reshape(-1,1) if not date else data

    elif dataset_name in {"rf_emf_tur", "rf_emf_tur2", "rf_emf_nov"}:
        map = {"rf_emf_tur": "Tur_Monitoring.csv", "rf_emf_tur2": "Tur_Monitoring2.csv", "rf_emf_nov": "Nov_Monitoring.csv"}
        file_name = map[dataset_name] # Turin1 = poli (polytechnic), Turin2 = Porta Nova (train station), Nov = Novara
        data = pd.read_csv(os.path.join(root_dir, "data/forecasting/rf_emf2", file_name), sep=';')

        # Drop 'Data' and 'Ora' columns and reorder remaining columns
        data = data.drop(['Data', 'Ora'], axis=1)
        data = data[['DateTime', 'CE']].to_numpy()  # Reorder columns properly and convert to numpy
        data[:, 1] = clip_and_interpolate(data[:, 1], thresh) if clip else data[:, 1]
        data = data[:, 1].reshape(-1,1) if not date else data

    else:
        data = pd.read_csv(os.path.join(root_dir, "data/forecasting", f"{dataset_name}.csv"))
        data = data.drop(columns=["date"]).values if not date else data

    dtype = type(data[0, 1]) if date else type(data[0, 0])


    if average_italy:
        data = time_window_average(data=data, m=5) # 30min average (6min sampling rate)

    def convert_array(arr, dtype=np.float32):
        # Convert each column to float32
        converted_arr = np.zeros(arr.shape, dtype=dtype)
        for i in range(arr.shape[1]):
            converted_arr[:, i] = arr[:, i].astype(dtype)

        return converted_arr

    if date and datetime_numeric:
        date_numeric_array = convert_array(datetime_to_numerical(data[:, 0]).astype(data[:,1].dtype))

        if cyclical_encoding:
            encoder = CyclicalFeatureEncoder(datetime_features)
            date_numeric_array = encoder.encode_sequence(date_numeric_array)
        data = np.concatenate((convert_array(data[:, 1].reshape(-1,1)), date_numeric_array), axis=1)

    if full_channels:
        if differencing!=0:
            data = difference_series(data, differencing)
        return data.T

    if date: assert datetime_numeric, "If date is True and full_channels is False, datetime_numeric must also be True for window preprocessing."

    # Define train, validation, and test indices
    if "etth" in dataset_name.lower():
        train_idx = [0, 12 * 30 * 24]
        val_idx = [12 * 30 * 24 - seq_len, 12 * 30 * 24 + 4 * 30 * 24]
        test_idx = [12 * 30 * 24 + 4 * 30 * 24 - seq_len, 12 * 30 * 24 + 8 * 30 * 24]
    elif "ettm" in dataset_name.lower():
        train_idx = [0, 12 * 30 * 24 * 4]
        val_idx = [12 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 4 * 30 * 24 * 4]
        test_idx = [12 * 30 * 24 * 4 + 4 * 30 * 24 * 4 - seq_len, 12 * 30 * 24 * 4 + 8 * 30 * 24 * 4]
    else:
        test_split = 1 - train_split - val_split
        num_train = int(len(data) * train_split)
        num_test = int(len(data) * test_split)
        num_val = len(data) - num_train - num_test

        train_idx = [0, num_train]
        val_idx = [num_train - seq_len, num_train + num_val]
        test_idx = [len(data) - num_test - seq_len, len(data)]

    # Split data
    train_data = data[train_idx[0]:train_idx[1]]
    val_data = data[val_idx[0]:val_idx[1]]
    test_data = data[test_idx[0]:test_idx[1]]

    # Normalization. Data must be in shape (seq_len, num_channels)
    if scale:
        scaler = StandardScaler()
        if date:
            # Reshape the column to 2D array for StandardScaler
            train_col = train_data[:, 0].reshape(-1, 1)
            val_col = val_data[:, 0].reshape(-1, 1)
            test_col = test_data[:, 0].reshape(-1, 1)

            # Fit and transform
            scaler.fit(train_col)

            # Transform and assign back
            train_data[:, 0] = scaler.transform(train_col).ravel()
            val_data[:, 0] = scaler.transform(val_col).ravel()
            test_data[:, 0] = scaler.transform(test_col).ravel()
        else:
            # If no date, transform entire array
            train_data = scaler.fit_transform(train_data)
            val_data = scaler.transform(val_data)
            test_data = scaler.transform(test_data)

    # Univariate. Create windows directly (separated by channel), and return input and labels separately
    # Data must be shape (num_samples, 2) for date-inclusion or (num_samples, 1) for no dates
    if univariate:

        train_inputs, train_labels, val_inputs, val_labels, test_inputs, test_labels = [], [], [], [], [], []

        if date:
            train_window = create_windows(torch.from_numpy(train_data), seq_len+pred_len, window_stride, resizing_mode)
            val_window = create_windows(torch.from_numpy(val_data), seq_len+pred_len, window_stride, resizing_mode)
            test_window = create_windows(torch.from_numpy(test_data), seq_len+pred_len, window_stride, resizing_mode)

            train_data = train_window[:, :seq_len]; train_labels = train_window[:, seq_len:]
            val_data = val_window[:, :seq_len]; val_labels = val_window[:, seq_len:]
            test_data = test_window[:, :seq_len]; test_labels = test_window[:, seq_len:]

        else:
            channels = [target_channel] if single_channel else range(data.shape[1])
            for i in channels:
                train_window = create_windows(torch.from_numpy(train_data.T[i]), seq_len+pred_len, window_stride, resizing_mode)
                val_window = create_windows(torch.from_numpy(val_data.T[i]), seq_len+pred_len, window_stride, resizing_mode)
                test_window = create_windows(torch.from_numpy(test_data.T[i]), seq_len+pred_len, window_stride, resizing_mode)

                train_x = train_window[:, :seq_len]; train_y = train_window[:, seq_len:]
                val_x = val_window[:, :seq_len]; val_y = val_window[:, seq_len:]
                test_x = test_window[:, :seq_len]; test_y = test_window[:, seq_len:]

                train_inputs.append(train_x); train_labels.append(train_y);
                val_inputs.append(val_x); val_labels.append(val_y);
                test_inputs.append(test_x); test_labels.append(test_y)

            train_data = torch.cat(train_inputs, dim=0)
            train_labels = torch.cat(train_labels, dim=0)
            val_data = torch.cat(val_inputs, dim=0)
            val_labels = torch.cat(val_labels, dim=0)
            test_data = torch.cat(test_inputs, dim=0)
            test_labels = torch.cat(test_labels, dim=0)

        if datetime_numeric:

            # Ensure (num_windows, seq_len) format (discard datetime numeric data)
            train_labels = train_labels[:, :, 0]
            val_labels = val_labels[:, :, 0]
            test_labels = test_labels[:, :, 0]

            # Ensure (num_windows, num_channels, seq_len) format
            train_data = train_data.permute(0, 2, 1)
            val_data = val_data.permute(0, 2, 1)
            test_data = test_data.permute(0, 2, 1)

        return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)

    return train_data.T, val_data.T, test_data.T

def analyze_nan_locations(data):
    """
    Analyzes the location and patterns of NaN values in the data.
    Handles both string and numeric data types.

    Args:
        data: numpy array or pandas Series

    Returns:
        dict containing analysis results
    """
    # Convert to numpy array if needed
    if isinstance(data, pd.Series):
        data = data.values
    data = np.asarray(data).ravel()

    # Handle string data types
    if data.dtype.kind in ['U', 'S', 'O']:  # Unicode, String, or Object dtype
        # Consider empty strings, whitespace, or 'nan' strings as NaN
        nan_mask = np.vectorize(lambda x: (
            isinstance(x, str) and (
                not x.strip() or  # empty or whitespace
                x.strip().lower() == 'nan' or  # 'nan' string
                x.strip() == ' '  # single space
            )
        ) or x is None or x != x)(data)  # None or actual NaN
    else:
        # For numeric data, use standard isnan
        try:
            nan_mask = np.isnan(data)
        except TypeError:
            # If type conversion failed, try converting to float
            try:
                numeric_data = data.astype(float)
                nan_mask = np.isnan(numeric_data)
            except (ValueError, TypeError):
                raise TypeError(f"Unable to analyze data of type {data.dtype} for NaN values")

    nan_indices = np.where(nan_mask)[0]
    total_nans = len(nan_indices)

    if total_nans == 0:
        return {
            "has_nans": False,
            "message": "No NaN values found in the data",
            "data_type": str(data.dtype)
        }

    # Find consecutive NaN sequences
    nan_diff = np.diff(nan_indices)
    sequence_starts = np.where(nan_diff > 1)[0] + 1
    sequence_starts = np.insert(sequence_starts, 0, 0)

    # Analyze NaN sequences
    sequences = []
    for i in range(len(sequence_starts)):
        start_idx = nan_indices[sequence_starts[i]]
        if i == len(sequence_starts) - 1:
            end_idx = nan_indices[-1]
        else:
            end_idx = nan_indices[sequence_starts[i + 1] - 1]
        sequence_length = end_idx - start_idx + 1

        # Get some context around the NaN sequence
        context_start = max(0, start_idx - 2)
        context_end = min(len(data), end_idx + 3)
        context_values = data[context_start:context_end]

        sequences.append({
            "start_index": start_idx,
            "end_index": end_idx,
            "length": sequence_length,
            "context": context_values
        })

    return {
        "has_nans": True,
        "total_nans": total_nans,
        "percent_nan": (total_nans / len(data)) * 100,
        "first_nan_index": nan_indices[0],
        "last_nan_index": nan_indices[-1],
        "nan_sequences": sequences,
        "sequence_count": len(sequences),
        "longest_sequence": max(seq["length"] for seq in sequences),
        "nan_indices": nan_indices.tolist(),
        "data_type": str(data.dtype)
    }

def print_nan_analysis(data):
    """
    Prints a human-readable analysis of NaN values in the data.

    Args:
        data: numpy array or pandas Series
    """
    results = analyze_nan_locations(data)

    if not results["has_nans"]:
        print(results["message"])
        print(f"Data type: {results['data_type']}")
        return

    print(f"NaN Analysis Results:")
    print(f"Data type: {results['data_type']}")
    print(f"Total NaN values: {results['total_nans']}")
    print(f"Percentage of NaN: {results['percent_nan']:.2f}%")
    print(f"First NaN at index: {results['first_nan_index']}")
    print(f"Last NaN at index: {results['last_nan_index']}")
    print(f"\nNaN Sequences ({results['sequence_count']} found):")

    for i, seq in enumerate(results['nan_sequences'], 1):
        print(f"\nSequence {i}:")
        print(f"  Start index: {seq['start_index']}")
        print(f"  End index: {seq['end_index']}")
        print(f"  Length: {seq['length']}")
        print(f"  Context values around sequence: {seq['context']}")

def difference_series(data: np.ndarray, order: int = 1) -> np.ndarray:
    """
    Apply differencing of specified order to a multivariate time series.

    Args:
        data: Array of shape (seq_len, num_channels)
        order: Order of differencing (default=1)

    Returns:
        Differenced series of shape (seq_len-order, num_channels)
    """
    if order < 0:
        raise ValueError("Order must be non-negative")
    if order == 0:
        return data

    seq_len, num_channels = data.shape
    diff_data = data.copy()

    # Apply differencing 'order' times
    for _ in range(order):
        # For each channel
        for channel in range(num_channels):
            diff_data[1:, channel] = np.diff(diff_data[:, channel])

        # Remove the first row which is undefined after differencing
        diff_data = diff_data[1:]

    return diff_data

def time_window_average(data: np.ndarray, m: int) -> np.ndarray:
    """
    Compute m-minute averages over a time series with stride m.

    Args:
        data: numpy array of shape (num_time_steps, num_channels)
        m: number of time steps to average over

    Returns:
        numpy array of shape (num_windows, num_channels) where
        num_windows = num_time_steps // m
    """
    # Ensure data is numpy array
    data = np.asarray(data)
    num_time_steps, num_channels = data.shape

    # Calculate number of complete windows
    num_windows = num_time_steps // m

    # Reshape data to (num_windows, m, num_channels)
    reshaped = data[:num_windows * m].reshape(num_windows, m, num_channels)

    # Take mean along the m dimension
    return np.mean(reshaped, axis=1)

def clip_and_interpolate(data, thresh=1.0):
    """
    Clips the data values to a threshold and interpolates the clipped values.
    If neighboring values are also above the threshold, it looks further for valid neighbors.

    Args:
        data (numpy.ndarray): The time series data as a 1D array of shape (num_time_steps,).
        thresh (float): The threshold value to clip the data.
    Returns:
        numpy.ndarray: Data where any values above the threshold are interpolated.
    """
    result = np.copy(data)
    mask = result > thresh

    def find_valid_neighbor(index, direction):
        step = 1 if direction == 'right' else -1
        i = index + step
        while 0 <= i < len(result):
            if result[i] <= thresh:
                return i, result[i]
            i += step
        return None, None

    for i in range(len(result)):
        if mask[i]:
            left_idx, left_val = find_valid_neighbor(i, 'left')
            right_idx, right_val = find_valid_neighbor(i, 'right')

            if left_idx is not None and right_idx is not None:
                # Interpolate between valid neighbors
                total_distance = right_idx - left_idx
                left_distance = i - left_idx
                right_distance = right_idx - i
                result[i] = (right_distance * left_val + left_distance * right_val) / total_distance
            elif left_idx is not None:
                result[i] = left_val
            elif right_idx is not None:
                result[i] = right_val
            # If both are None, we keep the original value

    return result

def load_splits(data, train_split=0.6, val_split=0.2, scale=True):
    """
    Loads the training, validation, and test splits of the data tensor.

    Args:
        data (numpy.ndarray): The time series data in a tensor of shape (num_time_steps, num_channels).
        train_split (float): The proportion of the data to use for training.
        val_split (float): The proportion of the data to use for validation.
    Returns:
        numpy.ndarray: The training split of the data tensor (num_channels, train_len).
        numpy.ndarray: The validation split of the data tensor (num_channels, val_len).
        numpy.narray: The test split of the data tensor (num_channels, test_len).
    """

    num_time_steps = data.shape[0]
    train_slice = slice(None, int(train_split * num_time_steps))
    val_slice = slice(int(train_split * num_time_steps), int((train_split + val_split) * num_time_steps))
    test_slice = slice(int((train_split + val_split) * num_time_steps), None)
    train_data, val_data, test_data = data[train_slice, :], data[val_slice, :], data[test_slice, :]

    if scale:
        scaler = StandardScaler()
        scaler.fit(train_data)
        train_data, val_data, test_data = scaler.transform(train_data), scaler.transform(val_data), scaler.transform(test_data)

    return train_data.T, val_data.T, test_data.T

def datetime_to_numerical(datetime_array):
    """
    Convert an array of datetime strings to numerical features.

    Args:
        datetime_array: 1D numpy array of strings in formats:
                       'YYYY/MM/DD HH:MM:SS' or
                       'YYYY-MM-DD HH:MM:SS' or
                       'DD/MM/YYYY HH:MM:SS'
    Returns:
        2D numpy array of shape (num_time_steps, 6) containing:
        - Year: normalized by subtracting min_year and scaling
        - Month: 1-12
        - Day: 1-31
        - Hour: 0-23
        - Minute: 0-59
        - Second: 0-59
    """
    num_samples = len(datetime_array)
    features = np.zeros((num_samples, 6))

    # Parse each datetime string
    for i, dt_str in enumerate(datetime_array):
        try:
            # Try YYYY/MM/DD format
            dt = datetime.strptime(dt_str, '%Y/%m/%d %H:%M:%S')
        except ValueError:
            try:
                # Try YYYY-MM-DD format
                dt = datetime.strptime(dt_str, '%Y-%m-%d %H:%M:%S')
            except ValueError:
                try:
                    # Try DD/MM/YYYY format
                    dt = datetime.strptime(dt_str, '%d/%m/%Y %H:%M:%S')
                except ValueError:
                    raise ValueError(f"Datetime string '{dt_str}' does not match any of the formats: "
                                  f"'YYYY/MM/DD HH:MM:SS', 'YYYY-MM-DD HH:MM:SS', or 'DD/MM/YYYY HH:MM:SS'")

        # Extract features
        features[i, 0] = dt.year
        features[i, 1] = dt.month
        features[i, 2] = dt.day
        features[i, 3] = dt.hour
        features[i, 4] = dt.minute
        features[i, 5] = dt.second

    # Normalize year to prevent large numbers
    min_year = features[:, 0].min()
    features[:, 0] = features[:, 0] - min_year

    return features

def convert_scientific_notation(data_array):
    """
    Converts an array of scientific notation strings to floats, handling whitespace
    and common formatting issues.

    Args:
        data_array: Array-like object containing strings of numbers

    Returns:
        Array with converted float values
    """

    def clean_and_convert(value):
        if isinstance(value, (float, int)):
            return float(value)

        # Strip whitespace and handle empty strings
        value = str(value).strip()
        if not value or value == ' ':
            return np.nan

        try:
            # Handle scientific notation and regular numbers
            return float(value)
        except ValueError:
            # Remove any extra spaces within the string
            value = ''.join(value.split())
            try:
                return float(value)
            except ValueError:
                return np.nan

    # Convert to numpy array if not already
    if not isinstance(data_array, np.ndarray):
        data_array = np.array(data_array)

    # Apply conversion to each element
    vectorized_convert = np.vectorize(clean_and_convert)
    return vectorized_convert(data_array)


#<-----------------OpenNeuro SOZ Classification------------------------>
def create_windows(tensor, window_size, window_stride, resizing_mode="None"):
    """
    Uses the sliding window technique to sample windows for a time series tensor.
    Handles both 1D and 2D input tensors.

    Args:
        tensor (torch.Tensor): The input tensor of shape (seq_len,) for 1D or (seq_len, num_channels) for 2D.
        window_size (int): The size of the window to sample.
        window_stride (int): The stride of the window.
        resizing_mode (str): The mode for handling sequences longer than window_size.

    Returns:
        torch.Tensor: Windows of shape (num_windows, window_size) for 1D input
                     or (num_windows, window_size, num_channels) for 2D input
    """
    # Handle special case of window_size = -1
    if window_size == -1:
        return tensor.unsqueeze(0)

    # Get tensor dimensions and handle 1D case
    if tensor.dim() == 1:
        seq_len = tensor.size(0)
        is_1d = True
    else:
        seq_len, num_channels = tensor.size()
        is_1d = False

    # Handle resizing cases
    if window_size > seq_len:
        if resizing_mode not in {"pad_trunc", "resize"}:
            raise ValueError("Window size must be less than or equal to the sequence length")
        data = [tensor]
        return resize_sequence(data, window_size, resizing_mode)

    # Set window stride for non-overlapping sampling if -1
    if window_stride == -1:
        window_stride = window_size

    # Calculate number of windows
    num_windows = (seq_len - window_size) // window_stride + 1
    if num_windows <= 0:
        raise ValueError("No windows can be formed with the given parameters")

    if is_1d:
        # Handle 1D case
        return torch.as_strided(
            tensor,
            size=(num_windows, window_size),
            stride=(window_stride, 1)
        )
    else:
        # Handle 2D case
        channel_stride = tensor.stride(-1)
        return torch.as_strided(
            tensor,
            size=(num_windows, window_size, num_channels),
            stride=(window_stride * num_channels, num_channels, channel_stride)
        )

def create_equidistant_windows(time_series, num_windows, window_length):
    """
    Sample equidistant windows from a multi-channel time series tensor.

    This function samples 'num_windows' windows of length 'window_length' from the input 'time_series'.
    The windows are as equidistant as possible given the constraints of the input length and desired window parameters.

    Args:
        time_series (torch.Tensor): Input tensor of shape (sequence_length, num_channels).
        num_windows (int): Number of windows to sample.
        window_length (int): Length of each window.

    Returns:
        torch.Tensor: Tensor of sampled windows with shape (num_windows, window_length, num_channels).

    Raises:
        ValueError: If the input tensor does not have 2 dimensions.

    Notes:
        - If sequence_length < window_length, the time series is padded to window_length and returned as a single window.
        - If num_windows == 1, the center window is returned.
        - If sequence_length/window_length < num_windows, windows will overlap to provide the requested number of windows.
    """
    # Check input dimensions
    if time_series.dim() != 2:
        raise ValueError("Input tensor must have shape (sequence_length, num_channels)")

    L, C = time_series.shape
    N = num_windows
    W = window_length

    # Handle the case where L < W
    if L < W:
        # Pad the time series to length W
        padded_series = torch.nn.functional.pad(time_series, (0, 0, 0, W - L))
        return padded_series.unsqueeze(0)  # Return shape (1, W, C)

    if N == 1:
        # If only one window is requested, return the center window
        start = (L - W) // 2
        return time_series[start:start+W].unsqueeze(0)

    # Calculate the step size between window starts
    if L - W >= N - 1:
        # No overlap needed
        step = (L - W) // (N - 1)
    else:
        # Overlap needed
        step = (L - W) / (N - 1)

    # Calculate the start indices of each window
    if isinstance(step, int):
        start_indices = torch.arange(0, L - W + 1, step)[:N]
    else:
        start_indices = torch.linspace(0, L - W, N).long()

    # Sample the windows
    windows = torch.stack([time_series[i:i+W] for i in start_indices])

    return windows

def resize_sequence(data, max_seq_len=3000, resizing_mode="none"):
    if not isinstance(data, list):
        raise ValueError("Input must be a list of 1D numpy arrays.")

    resized_data = np.zeros((len(data), max_seq_len))

    for i, seq in enumerate(data):
        if isinstance(seq, torch.Tensor):
            seq = seq.numpy().squeeze()
        if not isinstance(seq, np.ndarray) or seq.ndim != 1:
            raise ValueError("Each element in the list must be a 1D numpy array.")

        if resizing_mode == "pad_trunc":
            if seq.shape[0] >= max_seq_len:
                resized_data[i] = seq[:max_seq_len]
            else:
                resized_data[i, :seq.shape[0]] = seq

        elif resizing_mode == "resize":
            if seq.shape[0] == 1:
                resized_data[i] = np.full(max_seq_len, seq[0])
            else:
                x = np.linspace(0, 1, seq.shape[0])
                f = interpolate.interp1d(x, seq)
                x_new = np.linspace(0, 1, max_seq_len)
                resized_data[i] = f(x_new)

        else:
            raise ValueError("Invalid resizing mode. Choose from 'pad_trunc' or 'resize'.")

    return torch.from_numpy(resized_data)

#<-----------------General Usage------------------------>

def get_loader(args,
               rank,
               world_size,
               data,
               labels=None,
               ch_ids=None,
               flag="sl",
               shuffle=False,
               generator=torch.Generator()):
    """
    Returns a DataLoader for a specific task.


    Returns:
        DataLoader (Optional): A PyTorch DataLoader object for the a dataset class.
        Dataset (Optional): A PyTorch Dataset object for the a dataset class.
    """
    dataset_class = eval(f"args.{flag}.dataset_class")

    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data).float()

    if dataset_class == "forecasting":
        dataset = ForecastingDataset(data, args.data.seq_len, args.data.pred_len, args.data.dtype)
    elif dataset_class == "univariate_forecasting":
        dataset = UnivariateForecastingDataset(data[0], data[1], args.data.dtype)
    elif dataset_class == "classification":
        dataset = ClassificationDataset(data, labels, ch_ids, args.open_neuro.task, args.data.full_channels)
    else:
        raise ValueError(f"Invalid dataset class: {dataset_class}")

    if args.data.dataset_only:
        return dataset

    sampler = ...
    return DataLoader(dataset,
                      batch_size=eval(f"args.{flag}.batch_size"),
                      shuffle=shuffle,
                      drop_last=args.data.drop_last,
                      num_workers=args.data.num_workers,
                      generator=generator if not args.ddp.ddp else None,  # Only pass generator if not using ddp (seed from distributed sampler is used instead)
                      sampler=sampler,
                      persistent_workers=True,
                      prefetch_factor=args.data.prefetch_factor,
                      pin_memory=args.data.pin_memory)

def get_loaders(args, flag="sl", generator=torch.Generator(), rank=0, world_size=1, dataset_class="forecasting", loader_type="train", dataset_only=False):
    if args.data.dataset in {"ETTh1", "ETTh2", "ETTm1", "ETTm2", "electricity", "traffic", "weather", "illness", "rf_emf", "rf_emf_det", "rf_emf_ptv", "rf_emf_ptv2", "rf_emf_tur"}:
        train_data, val_data, test_data = load_forecasting(dataset_name=args.data.dataset,
                                                           seq_len=args.data.seq_len,
                                                           pred_len=args.data.pred_len,
                                                           window_stride=args.data.window_stride,
                                                           scale=args.data.scale,
                                                           clip=args.data.clip,
                                                           thresh=args.data.clip_thresh,
                                                           train_split=args.data.train_split,
                                                           val_split=args.data.val_split,
                                                           univariate=args.data.univariate,
                                                           resizing_mode=args.data.resizing_mode,
                                                           single_channel=args.data.single_channel,
                                                           target_channel=args.data.target_channel,
                                                           date=args.data.datetime,
                                                           datetime_numeric=args.data.datetime,
                                                           cyclical_encoding=args.data.cyclical_encoding,
                                                           datetime_features=args.data.datetime_features,
                                                           average_italy=args.data.average_italy)
        train_labels = val_labels = test_labels = train_ch_ids = val_ch_ids = test_ch_ids = None

    if loader_type in {"train", "all"}:
        train_loader = get_loader(args=args,
                                data=train_data,
                                labels=train_labels,
                                shuffle=True,
                                rank=rank,
                                world_size=world_size,
                                generator=generator,
                                ch_ids=train_ch_ids,
                                flag=flag)
        val_loader = get_loader(args=args,
                                data=val_data,
                                labels=val_labels,
                                shuffle=True,
                                rank=rank,
                                world_size=world_size,
                                generator=generator,
                                ch_ids=val_ch_ids,
                                flag=flag)
    if loader_type in {"test", "all"}:
        test_loader = get_loader(args=args,
                                data=test_data,
                                labels=test_labels,
                                shuffle=args.data.shuffle_test,
                                rank=rank,
                                world_size=world_size,
                                generator=generator,
                                ch_ids=test_ch_ids,
                                flag=flag)

    loaders = []

    if loader_type=="train":
        loaders.append(train_loader); loaders.append(val_loader)
    elif loader_type=="test":
        loaders.append(test_loader)
    elif loader_type=="all":
        loaders.append(train_loader); loaders.append(val_loader); loaders.append(test_loader)
    else:
        raise ValueError(f"Invalid loader type: {loader_type}")

    return tuple(loaders)
