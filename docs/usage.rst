Usage
=====

Example (Single Contract)
----------

Here's a simple example of how to use OpTrade:

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



Example (Multiple Contracts)
--------------

When modeling multiple contracts, you can use the `optrade.data.contracts.ContractDataset` class
to find a set of optimal contracts with similar parameters and then use the `optrade.data.forecasting.get_forecasting_dataset`
function to load and transform the data for all contracts.

.. code-block:: python

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


Example (Model Training)
--------------

Coming soon.
