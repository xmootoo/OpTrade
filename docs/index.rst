.. OpTrade documentation master file, created by
   sphinx-quickstart on Thu Mar 20 10:13:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpTrade
=====================

OpTrade is a complete toolkit for quantitative research and development of options trading strategies. By abstracting away the complexity of data handling and experimental setup, researchers and traders can focus on what matters most: developing and testing alpha-generating ideas.

.. image:: _static/optrade_light.png
    :alt: OpTrade Framework
    :align: center


Installation
--------------
The recommended way to install OptTrade is via pip::

        pip install optrade

Example (Single Contract)
--------------

.. code-block:: python

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

Overview
--------------

🔄 **Data Pipeline**
OpTrade integrates with ThetaData's API for affordable options and security data access (down to 1-min resolution). The framework processes NBBO quotes and OHLCVC metrics through a contract selection system optimizing for moneyness, expiration windows, and volatility-scaled strikes.

🌐 **Market Environments**
Built-in market environments enable precise universe selection through multifaceted filtering. OpTrade supports composition by major indices, fundamental-based screening (e.g., PE ratio, market cap), and Fama-French model categorization.

🧪 **Experimental Pipeline**
The experimentation framework supports PyTorch and scikit-learn for options forecasting with online Neptune logging, hyperparameter tuning, and model version control, supporting both online and offline experiment tracking.

🧮 **Featurization**
OpTrade provides option market features including mid-price derivations, order book imbalance metrics, quote spreads, and moneyness calculations. Time-to-expiration transformations capture theta decay effects, while datetime features extract cyclical market patterns for intraday seasonality.

🤖 **Models**
OpTrade includes several off-the-shelf PyTorch and scikit-learn models, including state-of-the-art architectures for time series forecasting alongside tried and true machine learning methods

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   modules
   contributing


Contact
--------------
For queries, please contact: `xmootoo at gmail dot com`.

Indices and tables
==================
* :ref:`modindex`
* :ref:`search`
