from optrade.data.universe import Universe
from rich.console import Console

# Step 1: Initialize Universe
ctx = Console()
universe = Universe(
    dow_jones=True,                # Use Dow Jones as the starting universe
    start_date="20210101",
    end_date="20211001",

    # Filters
    pe_ratio="low",          # Low debt ratio (bottom third)
    market_cap="high",             # Large-cap (top third)
    investment_beta="aggressive",  # Aggressive investment strategy (Fama-French exposure)
    verbose=True,
    dev_mode=True
)

# Step 2: Fetch constituents from Wikipedia
universe.set_roots()

# Step 3: Get market data via yfinance & compute Fama-French exposures
universe.get_market_metrics()

# Step 4: Apply filters (low debt, high market cap, aggressive investment beta)
universe.filter()

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












# # Step 6: Select a stock the universe and create PyTorch dataloders
# root = universe.roots[0]
# print(f"Loading data for root: {root}")

# loaders = universe.get_forecasting_loaders(
#     offline=True,               # Use cached data
#     root=root,                  # Stock symbol
#     tte_tolerance=(20, 40),     # DTE range
#     seq_len=30,                 # 30-min lookback
#     pred_len=5,                 # 5-min forecast
#     core_feats=["option_mid_price"],  # Feature
#     target_channels=["option_mid_price"],  # Target
#     dtype="float32",            # Precision
#     scaling=False,              # No normalization
# )

# # Display dataset sizes for each split
# print(f"Train loader: {len(loaders[0].dataset)} samples")
# print(f"Validation loader: {len(loaders[1].dataset)} samples")
# print(f"Test loader: {len(loaders[2].dataset)} samples")
