from optrade.data.universe import Universe

# Create a Universe instance
universe = Universe(
    dow_jones=True,
    debt_to_equity="low",
    market_cap="high",
    start_date="20210101",
    end_date="20211001",
    save_dir="test",
    verbose=True,
)
universe.set_candidate_roots() # Fetch index constituents
universe.get_fundamentals() # Fetch fundamental data
print(f"Universe: {universe.roots}")

# Filter the universe for stocks with low debt-to-equity and high market cap
universe.filter_universe()
print(f"Filtered universe: {universe.roots}")

# Download contracts and raw data for the filtered universe. This may take a while
universe.download(
    contract_stride=3,
    interval_min=1,
    right="C",
    target_tte=30,
    tte_tolerance=(20, 40),
    moneyness="ATM",
    train_split=0.5,
    val_split=0.3,
)

# Select a root for forecasting
root = universe.roots[0]
print(f"Loading data for root: {root}")
loaders = universe.get_forecasting_loaders(
    offline=True,
    root=root,
    tte_tolerance=(20, 40),
    seq_len=30, # 30-minute lookback window
    pred_len=5, # 5-minute forecast horizon
    core_feats=["option_mid_price"],
    target_channels=["option_mid_price"],
    dtype="float32",
    scaling=False,
)

print(f"Train loader: {len(loaders[0].dataset)} samples")
print(f"Validation loader: {len(loaders[1].dataset)} samples")
print(f"Test loader: {len(loaders[2].dataset)} samples")
