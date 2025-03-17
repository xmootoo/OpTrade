# OpTrade

OpTrade provides a complete toolkit for quantitative research and development of options trading strategies. By abstracting away the complexity of data handling and experimental setup, researchers and traders can focus on what matters most: developing and testing alpha-generating ideas.

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

## Key Features
🔄 **Data Pipeline**
Our data pipeline integrates with ThetaData's API, providing cost-effective access to minute-level options and security data. The framework processes both NBBO quotes and OHLCVC metrics through an intelligent contract selection system that optimizes for user-defined parameters such as moneyness, expiration windows, and volatility-scaled strike selection.

🌐 **Market Environments**
Custom market environments enable precise universe selection through multifaceted filtering of securities. The framework supports composition by major indices (S&P 500, NASDAQ 100, Dow Jones), factor-based screening (e.g., volatility, PE ratios, beta, market cap), and Fama-French model categorization.

🧪 **Experimental Pipeline**
The experimentation framework offers modern PyTorch and scikit-learn models for options forecasting with integrated Neptune logging, flexible hyperparameter tuning, and robust model version control. It manages the complete model lifecycle from training through evaluation with support for both online and offline experiment tracking.

🧮 **Featurization**
Several option market features are available, including mid-price derivations, order book imbalance metrics, quote spreads, and moneyness calculations. Time-to-expiration transformations capture theta decay effects through multiple mathematical representations, while specialized datetime features extract cyclical market patterns to model intraday seasonality and weekly option expiration effects.

🤖 **Models**
OpTrade includes state-of-the-art PyTorch deep learning architectures for time series forecasting alongside traditional machine learning models from scikit-learn, enabling researchers to leverage both cutting-edge DL approaches and proven quantitative techniques.


## Example Usage
### Single Contract
```py
# Step 1: Find and initialize the optimal contract
from optrade.data import Contract

contract = Contract.find_optimal(
    root="AAPL",
    start_date="20230103",  # First trading day of 2023
    target_tte=30,          # ~30 days to expiration
    tte_tolerance=(20, 40), # Min 20, max 40 days expiration
    interval_min=1,
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
)

# Step 4: Create dataset for time series forecasting
from optrade.data.forecasting import ForecastingDataset

dataset = ForecastingDataset(
    data=data,
    seq_len=100,        # 100-minute lookback window
    pred_len=10,        # 10-minute forecast horizon
    target_channels=[0],  # Forecast option returns (first column)
)
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

## Installation

### Dependencies
- Python ≥ 3.11
- Additional dependencies listed in `requirements.txt`

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
```


## Contact
For queries, please contact: `xmootoo at gmail dot com`.
