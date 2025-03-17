# Taken from: https://github.com/thuml/Time-Series-Library/blob/main/layers/Conv_Blocks.py
import torch
import torch.nn as nn

import torch.nn.functional as F


class InceptionBlock(nn.Module):
    """
    Inception Block for TimesNet (v1)
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, init_weight=True):
        super(InceptionBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_kernels = num_kernels
        kernels = []
        for i in range(self.num_kernels):
            kernels.append(nn.Conv2d(in_channels, out_channels, kernel_size=2 * i + 1, padding=i))
        self.kernels = nn.ModuleList(kernels)
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        res_list = []
        for i in range(self.num_kernels):
            res_list.append(self.kernels[i](x))
        res = torch.stack(res_list, dim=-1).mean(-1)
        return res


# Custom implementation of InceptionBlock for univariate time series
class InceptionBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, num_kernels=4):
        super(InceptionBlock1D, self).__init__()
        self.kernels = nn.ModuleList([
            nn.Conv1d(in_channels, out_channels // num_kernels, kernel_size=2 * i + 1, padding=i)
            for i in range(num_kernels)
        ])
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        return self.bn(F.gelu(torch.cat([kernel(x) for kernel in self.kernels], dim=1)))

class InceptionNet1D(nn.Module):
    def __init__(self, in_channels, num_blocks=3, out_channels=32, pool_size=2):
        super(InceptionNet1D, self).__init__()

        self.inception_blocks = nn.ModuleList([
            InceptionBlock1D(
                in_channels if i == 0 else out_channels,
                out_channels
            ) for i in range(num_blocks)
        ])

        self.pool = nn.MaxPool1d(kernel_size=pool_size)
        self.adaptive_pool = nn.AdaptiveAvgPool1d(1)
        self.final_linear = nn.Linear(out_channels, 1)

    def forward(self, x):
        # x shape: (batch_size, time_steps, features)
        x = x.transpose(1, 2)  # Change to (batch_size, features, time_steps) for Conv1D

        for inception_block in self.inception_blocks:
            x = inception_block(x)
            x = self.pool(x)

        x = self.adaptive_pool(x).squeeze(-1)  # Global average pooling
        x = self.final_linear(x)
        return x.squeeze(-1)  # Remove last dimension to get scalar

# Test the model
def test_inception_net_1d():
    # Set random seed for reproducibility
    torch.manual_seed(42)

    # Define model parameters
    time_steps, features = 64, 16
    batch_size = 32

    # Create the model
    model = InceptionNet1D(in_channels=features)

    # Generate random input data
    input_tensor = torch.randn(batch_size, time_steps, features)

    # Perform a forward pass
    output = model(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters())}")

if __name__ == "__main__":
    test_inception_net_1d()
