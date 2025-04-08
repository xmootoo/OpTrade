# OpTrade

[![Documentation Status](https://readthedocs.org/projects/optrade/badge/?version=latest)](https://optrade.readthedocs.io/en/latest/?badge=latest)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue.svg)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

<div align="center">
  <h3>
    <a href="https://optrade.readthedocs.io/">📚 Documentation</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://github.com/yourusername/optrade">💻 GitHub</a> &nbsp;&nbsp;|&nbsp;&nbsp;
    <a href="https://pypi.org/project/optrade/">📦 PyPI</a>
  </h3>
</div>
OpTrade is a complete toolkit for quantitative research and development of options trading strategies. By abstracting away the complexity of data handling and experimental setup, researchers and traders can focus on what matters most: developing and testing alpha-generating ideas.


<p align="center">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="optrade/assets/optrade_dark.png">
   <source media="(prefers-color-scheme: light)" srcset="optrade/assets/optrade_light.png">
   <img alt="OpTrade Framework" src="optrade/assets/optrade_light.png">
 </picture>
</p>

## Installation
The recommended way to install OptTrade is via pip:

```bash
pip install optrade
```


### Example (Single Contract)
```py
# Step 1: Find and initialize the optimal contract
from optrade.data import Contract

contract = Contract.find_optimal(
    root="AAPL",
    start_date="20230103",  # First trading day of 2023
    target_tte=30,          # Desired expiration: 30 days
    tte_tolerance=(20, 40), # Min 20, max 40 days expiration
    interval_min=1,         # Data requested at 1-min level
    moneyness="ATM",        # At-the-money option
)

# Step 2: Load market data (NBBO quotes and OHLCV)
df = contract.load_data()

# Step 3: Transform raw data into ML-ready features
from optrade.data.features import transform_features

data = transform_features(
    df=df,
    core_feats=[
        "option_returns",     # Option price returns
        "stock_returns",      # Underlying stock returns
        "moneyness",          # Log(S/K)
        "option_lob_imbalance", # Order book imbalance
        "stock_quote_spread", # Bid-ask spread normalized
    ],
    tte_feats=["sqrt", "exp_decay"],  # Time-to-expiration features
    datetime_feats=["minute_of_day", "hour_of_week"],  # Time features
    strike=contract.strike,
    exp=contract.exp,
)

# Step 4: Create dataset for time series forecasting
from optrade.data.forecasting import ForecastingDataset
from torch.utils.data import DataLoader

torch_dataset = ForecastingDataset(
    data=data,
    seq_len=100,        # 100-minute lookback window
    pred_len=10,        # 10-minute forecast horizon
    target_channels=["option_returns"],  # Forecast option returns
)

torch_loader = DataLoader(torch_dataset)
```

## Overview

🔄 **Data Pipeline**
OpTrade integrates with ThetaData's API for affordable options and security data access (down to 1-min resolution). The framework processes NBBO quotes and OHLCVC metrics through a contract selection system optimizing for moneyness, expiration windows, and volatility-scaled strikes.

🌐 **Market Environments**
Built-in market environments enable precise universe selection through multifaceted filtering. OpTrade supports composition by major indices, fundamental-based screening (e.g., PE ratio, market cap), and Fama-French model categorization.

🧪 **Experimental Pipeline**
The experimentation framework supports PyTorch and scikit-learn for options forecasting with online Neptune logging, hyperparameter tuning, and model version control, supporting both online and offline experiment tracking.

🧮 **Featurization**
OpTrade provides option market features including mid-price derivations, order book imbalance metrics, quote spreads, and moneyness calculations. Time-to-expiration transformations capture theta decay effects, while datetime features extract cyclical market patterns for intraday seasonality.

🤖 **Models**
OpTrade includes several off-the-shelf PyTorch and scikit-learn models, including state-of-the-art architectures for time series forecasting alongside tried and true machine learning methods.

## Advanced Usage
### Multiple Contracts

When modeling multiple contracts, you can use the `ContractDataset` class
to find a set of optimal contracts with similar parameters and then use the `get_forecasting_dataset` function to load and transform the data for all contracts:
```py
# Step 1: Find a set of optimal contracts from total_start_date to total_end_date
from optrade.data.contracts import ContractDataset

contract_dataset = ContractDataset(
    root="AMZN",
    total_start_date="20220101",
    total_end_date="20220301",
    contract_stride=1,
    interval_min=1,
    right="P",
    target_tte=3,
    tte_tolerance=(1,10),
    moneyness="ITM",
    strike_band=0.05,
    volatility_scaled=True,
    volatility_scalar=0.1,
    hist_vol=0.1117,
)
contract_dataset.generate()

# Step 2: Load market data and transform features for all contracts then put into a concatenated torch dataset
from optrade.data.forecasting import get_forecasting_dataset
from torch.utils.data import DataLoader

torch_dataset = get_forecasting_dataset(
    contract_dataset=contract_dataset,
    core_feats=["option_returns"],
    tte_feats=["sqrt"],
    datetime_feats=["sin_minute_of_day"],
    tte_tolerance=(25, 35),
    seq_len=100,
    pred_len=10,
    verbose=True
)
torch_loader = DataLoader(torch_dataset)
```

### Forecasting (PyTorch)
When running forecasting experiments, you can use the `Experiment` class from `optrade.exp.forecasting` which supports PyTorch deep learning (DL) models. Several state-of-the-art models are available in the `optrade.pytorch.models`, allowing you to easily experiment with different modern DL architectures:

```py
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
from optrade.pytorch.models.patchtst import Model as PatchTST
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
    metrics=["loss"],                  # Metrics to compute
    device=device,                     # Computing device (CPU/GPU)
)
exp.save_logs() # Save experiment logs to disk
```

### Universe
When modeling a universe of securities, you can use the `Universe` class to filter by parameters such as fundamentals (e.g., P/E ratio), volatility, and Fama-French factor exposures. Here's an example:

```py
from optrade.data.universe import Universe

# Step 1: Initialize Universe
universe = Universe(
    dow_jones=True,                # Use Dow Jones as the starting universe
    start_date="20210101",
    end_date="20211001",

    # Filters
    debt_to_equity="low",          # Low debt ratio (bottom third)
    market_cap="high",             # Large-cap (top third)
    investment_beta="aggressive",  # Aggressive investment strategy (Fama-French exposure)
)

# Step 2: Fetch constituents from Wikipedia
universe.set_candidate_roots()

# Step 3: Get fundamental data via yfinance & compute Fama-French exposures
universe.get_fundamentals()
print(f"Universe: {universe.roots}")

# Step 4: Apply filters (low debt, high market cap, aggressive investment beta)
universe.filter_universe()
print(f"Filtered universe: {universe.roots}")

# Step 5: Download options data for filtered universe
universe.download(
    contract_stride=3,          # Sample contracts every 3 days
    interval_min=1,             # Data requested at 1-min level
    right="C",                  # Calls options only
    target_tte=30,              # Desired expiration: 30 days
    tte_tolerance=(20, 40),     # Min 20, max 40 days expiration
    moneyness="ATM",            # At-the-money option
    train_split=0.5,            # 50% training
    val_split=0.3,              # 30% validation and (hence 20% test)
)

# Step 6: Select a stock the universe and create PyTorch dataloders
root = universe.roots[0]
print(f"Loading data for root: {root}")

loaders = universe.get_forecasting_loaders(
    offline=True,               # Use cached data
    root=root,                  # Stock symbol
    tte_tolerance=(20, 40),     # DTE range
    seq_len=30,                 # 30-min lookback
    pred_len=5,                 # 5-min forecast
    core_feats=["option_mid_price"],  # Feature
    target_channels=["option_mid_price"],  # Target
    dtype="float32",            # Precision
    scaling=False,              # No normalization
)

# Display dataset sizes for each split
print(f"Train loader: {len(loaders[0].dataset)} samples")
print(f"Validation loader: {len(loaders[1].dataset)} samples")
print(f"Test loader: {len(loaders[2].dataset)} samples")
```

## Contact
For queries, please contact: `xmootoo at gmail dot com`.
