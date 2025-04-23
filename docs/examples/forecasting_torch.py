# Step 1: Initialize the experiment with offline logging
from optrade.exp.forecasting import Experiment
exp = Experiment(logging="offline")

# Set device to GPU if available, otherwise CPU
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define feature sets for the model
core_feats = ["option_returns", "option_volume", "stock_lob_imbalance"]  # Core features
tte_feats = ["sqrt"]  # Time-to-expiration features
datetime_feats = ["sin_minute_of_day"]  # Temporal features
input_channels = core_feats + tte_feats + datetime_feats  # Combined input features
target_channels = ["option_returns"]  # Target variable

# Step 2: Initialize data loaders with specified configuration
exp.init_loaders(
    root="TSLA",                       # Ticker symbol
    start_date="20210601",             # Full dataset start date
    end_date="20211231",               # Full dataset end date
    contract_stride=5,                 # Sample contracts every 5 days
    interval_min=5,                    # 5-minute intervals
    right="C",                         # Call options
    target_tte=30,                     # Target 30 days to expiration
    tte_tolerance=(15, 45),            # Accept options with 15-45 days to expiration
    moneyness="ATM",                   # At-the-money options
    train_split=0.5,                   # 50% of data for training
    val_split=0.25,                    # 25% of data for validation (remaining 25% for testing)
    seq_len=12,                        # Input sequence length (12 x 5min = 1 hour lookback)
    pred_len=4,                        # Prediction length (4 x 5min = 20 minute forecast)
    scaling=True,                      # Normalize all features
    core_feats=core_feats,
    tte_feats=tte_feats,
    datetime_feats=datetime_feats,
    target_channels=target_channels,
    # DataLoader settings
    num_workers=0,                     # Single-process (development safe)
    prefetch_factor=None,              # No prefetching batches
    persistent_workers=False,          # Kill workers between epochs
)

# Step 3: Define model architecture
from optrade.models.pytorch.patchtst import Model as PatchTST
model = PatchTST(
    num_enc_layers=2,                  # Number of Transformer encoder layers
    d_model=32,                        # Model dimension (embedding size)
    d_ff=64,                           # Feed-forward network dimension
    num_heads=2,                       # Number of self-attention heads
    seq_len=12,                        # Input sequence length (must match data config)
    pred_len=4,                        # Prediction length (must match data config)
    patch_dim=2,                       # Patch dimension
    stride=2,                          # Patch stride
    input_channels=input_channels,
    target_channels=target_channels,
).to(device)

# Define optimization method and objetive (loss) function
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)  # Adam optimizer
criterion = torch.nn.MSELoss()                             # Mean Squared Error loss

# Step 4: Train the model
model = exp.train(
    model=model,
    device=device,
    optimizer=optimizer,
    criterion=criterion,
    num_epochs=5,                      # Number of training epochs
    early_stopping=True,               # Enable early stopping
    patience=20,                       # Number of epochs before early stopping
)

# Step 5: Evaluate model on test set
exp.test(
    model=model,
    criterion=criterion,
    metrics=["mse"],                  # Metrics to compute
    device=device,                     # Computing device (CPU/GPU)
)
exp.save_logs() # Save experiment logs to disk
