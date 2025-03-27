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
