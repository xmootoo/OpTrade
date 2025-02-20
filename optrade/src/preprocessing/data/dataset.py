import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

class ForecastingDataset(Dataset):
    """
    A standard forecasting dataset class for PyTorch.

    Args:
        data (torch.Tensor): The time series data in a tensor of shape (num_channels, num_time_steps).
        seq_len (int): The length of the input window.
        pred_len (int): The length of the forecast window.
        target_channels (list): The channels to forecast. If None, all channels are forecasted.
        dtype (str): The datatype of the tensor.

    __getitem__ method:

        Args:
            idx (int): The index of the input window in the time series.
        Returns:
            input_data (torch.Tensor): The lookback window of length seq_len for the given index.
            target_data (torch.Tensor): The forecast window for the given index (continuation of input_data
                                        shifted by pred_len)

    """

    def __init__(self, data, seq_len, pred_len, target_channels=None, dtype="float32"):

        # Convert the data to a tensor and set the datatype
        dtype = eval("torch." + dtype)
        if not torch.is_tensor(data):
            self.data = torch.from_numpy(data).type(dtype)
        else:
            self.data = data.type(dtype)
        self.seq_len = seq_len
        self.pred_len = pred_len

        if target_channels:
            self.target_channels = target_channels
        else:
            self.target_channels = list(range(self.data.shape[0]))

    def __len__(self):
        return self.data.shape[1] - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        input_data = self.data[:, idx:idx+self.seq_len]
        # target_data = self.data[:, idx+self.seq_len:idx+self.seq_len+self.pred_len]
        target_data = self.data[self.target_channels, idx+self.seq_len:idx+self.seq_len+self.pred_len]
        return input_data, target_data

class UnivariateForecastingDataset(Dataset):
    """
    A standard forecasting dataset class for PyTorch.

    Args:
        data_x (torch.Tensor): The time series data in a tensor of shape (num_windows, seq_len).
        data_y (torch.Tensor): The time series data in a tensor of shape (num_windows, pred_len).

    __getitem__ method: Returns the input and target data for a given index, where the target window follows
                        immediately after the input window in the time series.

    """

    def __init__(self, x, y, dtype="float32"):

        # Convert the data to a tensor and set the datatype
        dtype = eval("torch." + dtype)

        if not torch.is_tensor(x):
            self.x = torch.from_numpy(x).type(dtype)
            self.y = torch.from_numpy(y).type(dtype)
        else:
            self.x = x.type(dtype)
            self.y = y.type(dtype)

            print(f"x: {self.x.shape}, y: {self.y.shape}")

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class ClassificationDataset(Dataset):
    """
    A classification dataset class for time series.

    Args:
        x (torch.Tensor): The input data in a tensor of shape (num_windows, seq_len).
        y (torch.Tensor): The target data in a tensor of shape (num_windows).
        ch_ids (torch.Tensor): The channel IDs in a tensor of shape (num_windows).
        t (torch.Tensor): The time indices in a tensor of shape (num_windows).

    """

    def __init__(self, x, y, ch_ids=None, task="binary", full_channels=False):

        # Data
        self.x = x
        self.y = torch.tensor(y) if isinstance(y, list) else y
        ch_ids = torch.tensor(ch_ids) if isinstance(ch_ids, list) else ch_ids
        self.ch_ids = ch_ids
        self.full_channels = full_channels


        # Parameters
        self.task = task
        self.len = x.size(0)
        self.num_classes = len(torch.unique(self.y))

        # Channel IDs
        if not full_channels:
            self.unique_ch_ids = torch.unique(ch_ids, sorted=True).tolist()
            label_indices = torch.tensor([torch.where(ch_ids == unique_id)[0][0] for unique_id in self.unique_ch_ids])
            self.unique_ch_labels = self.y[label_indices].tolist()
            self.ch_labels = dict()
            for i, ch_id in enumerate(self.unique_ch_ids):
                self.ch_labels[ch_id] = int(self.unique_ch_labels[i])

        if ch_ids is not None and not full_channels:
            unique_ch_ids, indices = torch.unique(ch_ids, sorted=True, return_inverse=True)

            self.ch_id_list = unique_ch_ids.tolist()
            self.num_channels = len(unique_ch_ids)
            self.ch_targets = torch.zeros(self.num_channels)

            # Get the unique labels for each channel
            for i, ch_id in enumerate(unique_ch_ids):
                matching_indices = torch.where(ch_ids == ch_id)
                label = self.y[matching_indices][0]
                self.ch_targets[i] = label

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        """
        In a dataloader it returns appropriate tensors for CrossEntropy loss.
            x: (batch_size, 1, seq_len)
            y: (batch_size,)
            ch_ids: (batch_size,)
        """

        output = []

        if self.task=="multi":
            label = self.y[idx].long()
        elif self.task=="binary":
            label = self.y[idx].float()
        else:
            raise ValueError("Task must be either 'binary' or 'multi'.")

        if self.ch_ids is not None and not self.full_channels:
            output += [self.x[idx].unsqueeze(0), label, self.ch_ids[idx]]
        else:
            output += [self.x[idx].unsqueeze(0), label]

        return tuple(output)
