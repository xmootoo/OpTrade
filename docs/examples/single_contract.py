# Step 1: Find and initialize the optimal contract
from optrade.data.contracts import Contract
from rich.console import Console


ctx = Console()
ctx.log("Searching for an ATM call option available January 3, 2023 with approximately 30 days to expiration...")
contract = Contract.find_optimal(
    root="AAPL",
    right="C",              # Call option
    start_date="20230103",  # First trading day of 2023
    target_tte=30,          # Desired expiration: 30 days
    tte_tolerance=(20, 40), # Min 20, max 40 days expiration
    interval_min=15,         # Data requested at 15-min level
    moneyness="ATM",        # At-the-money option
    verbose=True,
)
ctx.log(f"Optimal contract found: {contract}")

# Step 2: Load market data (NBBO quotes and OHLCV)
ctx.log("Loading market data from ThetaData API...")
df = contract.load_data()
print(df.head())

# Step 3: Transform raw data into ML-ready features
from optrade.data.features import transform_features
from rich.table import Table
from rich import box

data = transform_features(
    df=df,
    core_feats=[
        "option_returns",     # Option price returns
        "stock_returns",      # Underlying stock returns
        "moneyness",          # Log(S/K)
        "option_lob_imbalance", # Order book imbalance
        "stock_quote_spread", # Bid-ask spread normalized
    ],
    tte_feats=["sqrt"],  # Time-to-expiration features
    datetime_feats=["minute_of_day"],  # Time features
    vol_feats=["rolling_volatility"], # Rolling volatility window and short-to-long volatility ratio
    rolling_volatility_range=[60], # 60min rolling volatility windows
    strike=contract.strike,
    exp=contract.exp,
    root=contract.root,
    right=contract.right,
)


table = Table(title="Transformed Features", box=box.SIMPLE)

# Add column headers
for col in data.columns:
    table.add_column(col, justify="center", style="cyan", no_wrap=True)

# Add top 10 rows only
for i, row in data.head(10).iterrows():
    table.add_row(*[str(item) for item in row.values])
ctx.log(table)


# Step 4: Create dataset for time series forecasting
from optrade.data.forecasting import ForecastingDataset
from torch.utils.data import DataLoader

ctx.log("Converting data to PyTorch dataset with lookback window size of 20 data points (300min) and forecast horizon of 5 data points (25min)")
ctx.log("Using all features as inputs, with target channel set to option returns")
torch_dataset = ForecastingDataset(
    data=data,
    seq_len=20,        # 300-minute lookback window
    pred_len=5,        # 25-minute forecast horizon
    target_channels=["option_returns"],  # Forecast option returns
)

torch_loader = DataLoader(torch_dataset, batch_size=32)

for batch in torch_loader:
    x, y = batch
    ctx.log("Grabbing a single example:")
    ctx.log(f"Input shape: {x.shape}, Target shape: {y.shape}")
    break
