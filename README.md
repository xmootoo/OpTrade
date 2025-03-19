# OpTrade

OpTrade is a complete toolkit for quantitative research and development of options trading strategies. By abstracting away the complexity of data handling and experimental setup, researchers and traders can focus on what matters most: developing and testing alpha-generating ideas.

<p align="center">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="optrade/assets/optrade_dark.png">
   <source media="(prefers-color-scheme: light)" srcset="optrade/assets/optrade_light.png">
   <img alt="OpTrade Framework" src="optrade/assets/optrade_light.png">
 </picture>
</p>

<!-- ## Overview

The framework focuses on two primary use cases:

1. **Alpha Generation**: Discovering and forecasting alpha term structures to analyze market dynamics across various options contracts
2. **Trading Strategy Development**: Translating these insights into actionable trading signals (planned for future implementation)
 -->

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
).values

# Step 4: Create dataset for time series forecasting
from optrade.data.forecasting import ForecastingDataset
from torch.utils.data import DataLoader

torch_dataset = ForecastingDataset(
    data=data,
    seq_len=100,        # 100-minute lookback window
    pred_len=10,        # 10-minute forecast horizon
    target_channels=[0],  # Forecast option returns (first column)
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
### Multi-contract Data
```py
# Step 1: Find optimal contracts within date range
from optrade.data import ContractDataset

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
    target_band=0.05,
    volatility_scaled=True,
    volatility_scalar=0.1,
    hist_vol=0.1117,
)
contract_dataset.generate()

# Step 2: Transform market data into ML-ready dataset
from optrade.data.torch import get_forecasting_dataset
from torch.utils.data import DataLoader

torch_dataset = get_forecasting_dataset(
    contracts=contract_dataset,
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




<!--
## What is an Alpha Term Structure?

An alpha term structure represents how excess returns (alpha) are expected to evolve over different time horizons. It is defined as:

$$
\mathbf{r} = (r_1, r_2, \dots, r_H)^T
$$

Where:
- $r_t$ is the expected excess return of an option contract at time $t$
- The vector captures returns across multiple future time points

This structure helps traders:
- Determine optimal entry/exit points
- Develop time-specific trading strategies
- Manage risk (e.g., adjust positions)
- Select appropriate option expiration dates -->

## Documentation
This project includes extensive documentation that is essential for understanding the framework. Users are strongly encouraged to review these documents before usage.

| Document | Description |
|----------|-------------|
| [DATA.md](DATA.md) | Information on the comprehensive data pipeline |
| [FEATURES.md](FEATURES.md) | Details on the selection of important predictors for option forecasting |



<!-- ### Dependencies
- Python ≥ 3.11
- Additional dependencies listed in `requirements.txt` -->
<!--
### Using conda (recommended)
```bash
# Create and activate conda environment
conda create -n venv python=3.11
conda activate venv

# Install requirements
cd <project_root_directory> # Go to project root directory
pip install -r requirements.txt
pip install -e .
```

### Using pip
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
cd <project_root_directory> # Go to project root directory
pip install -r requirements.txt
pip install -e .
``` -->


## Contact
For queries, please contact: `xmootoo at gmail dot com`.
