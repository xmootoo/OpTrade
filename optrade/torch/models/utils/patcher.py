import torch
import torch.nn as nn


class Patcher(nn.Module):
    """
    Splits the input time series into patches.
    """

    def __init__(self, patch_dim: int = 16, stride: int = 8):
        super(Patcher, self).__init__()
        self.patch_dim = patch_dim
        self.stride = stride

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, M, L). B: batch_size, M: channels, L: sequence_length.
        Returns:
            patches: tensor of shape (B, M, N, P). N: number of patches, P: patch_dim.
            patches_combined: tensor of shape (B * M, N, P). N: number of patches, P: patch_dim. This is more efficient
                              to input into the Transformer encoder, as we are applying it to channels independently, thus,
                              we can combine the batch and channel dimensions and then reshape it afterwards.
        """
        B, M, L = x.shape

        # Number of patches.
        N = int((L - self.patch_dim) / self.stride) + 2

        # Pad the time series with the last value on each channel repeated S times
        last_column = x[:, :, -1:]  # index
        padding = last_column.repeat(1, 1, self.stride)
        x = torch.cat((x, padding), dim=2)

        # Extract patches
        patches = x.unfold(
            dimension=2, size=self.patch_dim, step=self.stride
        )  # Unfold the input tensor to extract patches.
        patches = patches.contiguous().view(
            B, M, N, self.patch_dim
        )  # Reshape the tensor to (B, M, N, P).
        patches_combined = patches.view(
            B * M, N, self.patch_dim
        )  # Reshape the tensor to (B * M, N, P).

        return patches


class VerticalPatcher(nn.Module):
    """
    Splits the input time series into patches, vertically stacking channels within each patch.
    """

    def __init__(self, patch_dim: int = 16, stride: int = 8):
        super(VerticalPatcher, self).__init__()
        self.patch_dim = patch_dim
        self.stride = stride

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, M, L). B: batch_size, M: channels, L: sequence_length.
        Returns:
            patches: tensor of shape (B, N, M*P). N: number of patches, M*P: channels * patch_dim
        """
        B, M, L = x.shape

        # Number of patches
        N = ((L - self.patch_dim) // self.stride) + 1

        # Pad if needed
        if (L - self.patch_dim) % self.stride != 0:
            pad_size = self.stride - ((L - self.patch_dim) % self.stride)
            last_values = x[:, :, -1:]
            padding = last_values.repeat(1, 1, pad_size)
            x = torch.cat((x, padding), dim=2)
            N = ((x.shape[2] - self.patch_dim) // self.stride) + 1

        # Initialize output tensor
        patches = torch.zeros(B, N, M * self.patch_dim, device=x.device)

        # Create patches
        for i in range(N):
            start_idx = i * self.stride
            end_idx = start_idx + self.patch_dim

            # Extract and stack patches from each channel
            for c in range(M):
                patch_start = c * self.patch_dim
                patch_end = (c + 1) * self.patch_dim
                patches[:, i, patch_start:patch_end] = x[:, c, start_idx:end_idx]

        return patches
