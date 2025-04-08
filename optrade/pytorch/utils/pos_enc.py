import torch
import torch.nn as nn
import torch.nn.init as init


class PositionalEncoding(nn.Module):
    def __init__(self, patch_dim: int = 16, d_model: int = 128, num_patches: int = 64):
        super(PositionalEncoding, self).__init__()
        self.projection = nn.Linear(patch_dim, d_model)  # P x D projection matrix
        self.pos_encoding = nn.Parameter(
            torch.empty(num_patches, d_model)
        )  # N x D positional encoding matrix

        # Weight initialization
        init.xavier_uniform_(self.projection.weight)
        init.uniform_(self.pos_encoding, -0.02, 0.02)

    def forward(self, x):
        """
        Args:
            x: input tensor of shape (B, M, N, P) where B = batch_size, M = num_channels, N = num_patches,
               P = patch_dim.
        Returns:
            x: tensor of shape (B, M, N, D) where D = d_model.
        """

        B, M, N, P = x.shape
        x = x.view(
            B * M, N, P
        )  # Reshape the tensor to (B * M, N, P). We process each channel independently.
        x = self.projection(x) + self.pos_encoding.unsqueeze(0)
        x = x.view(B, M, N, -1)  # Reshape the tensor to (B, M, N, D).

        return x


# # Test PatchtstOG vs ours
# # Hyperparameters
# batch_size, num_channels, num_patches, patch_dim = 3, 4, 17, 6
# d_model = 5
# x = torch.randn(batch_size, num_channels, num_patches, patch_dim)

# # #==========================================MANUAL======================================================

# # Define parameters
# W_P_og = nn.Linear(patch_dim, d_model)
# W_pos_og = torch.empty((num_patches, d_model))

# # Initalization for W_P and W_pos
# nn.init.xavier_uniform_(W_P_og.weight)
# nn.init.uniform_(W_pos_og, -0.02, 0.02)


# # Fix random parameters
# # Print out parameters (out_features, in_features) format
# print(f"OG weight shape {W_P_og.weight.data.shape}")
# print(f"OG bias shape {W_P_og.bias.data.shape}")

# W_P_og.weight.data = torch.randn(d_model, patch_dim) * 5
# W_P_og.bias.data = torch.randn(d_model) * 17.3

# print(f"OG pos enc shape {W_pos_og.data.shape}")
# W_pos_og.data = torch.randn(num_patches, d_model) * (-2.4)


# #==========================================OURS======================================================


# # Initialize matrices to the same values
# positional_encoding = PositionalEncoding(patch_dim=patch_dim, d_model=d_model, num_patches=num_patches)

# print(f"(Before update) Ours weight shape {positional_encoding.projection.weight.data.shape}, "
#       f"bias shape {positional_encoding.projection.bias.data.shape}, "
#       f"and pos enc shape {positional_encoding.pos_encoding.data.shape}")

# positional_encoding.projection.weight.data = W_P_og.weight.data
# positional_encoding.projection.bias.data = W_P_og.bias.data
# positional_encoding.pos_encoding.data = W_pos_og.data

# print(f"(After update) Ours weight shape {positional_encoding.projection.weight.data.shape}, "
#       f"bias shape {positional_encoding.projection.bias.data.shape}, "
#       f"and pos enc shape {positional_encoding.pos_encoding.data.shape}")


# # OG
# # Project + Positional Encoding
# print(f"x shape: {x.shape}")
# print(f"W_P_og shape: {W_P_og.weight.data.shape}")
# u = W_P_og(x)                                                          # x: [bs x nvars x patch_num x d_model]
# u = torch.reshape(u, (u.shape[0]*u.shape[1],u.shape[2],u.shape[3]))      # u: [bs * nvars x patch_num x d_model]
# u = u + W_pos_og                                        # u: [bs * nvars x patch_num x d_model]

# # OURS
# print(u.shape)

# v = positional_encoding(x)
# print(v.shape)
# assert torch.allclose(u, v.view(batch_size*num_channels, num_patches, -1)), "Error: Positional Encoding not equal to OG"
