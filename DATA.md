# OpTrade Data Pipeline
This document explains the data collection, preprocessing, and loading pipeline for the options trading project. It covers how to obtain financial data for both options and their underlying assets, clean and process this data, and prepare it for machine learning models.

## Table of Contents
- [Overview](#overview)
- [Contract Model](#contract-model)
  - [Finding Optimal Contracts](#finding-optimal-contracts)
- [Contract Dataset](#contract-dataset)
  - [Dataset Generation](#dataset-generation)
- [Feature Engineering](#feature-engineering)
  - [Time-to-Expiration Features](#time-to-expiration-features)
  - [Datetime Features](#datetime-features)
  - [Compiling All Features](#compiling-all-features)
- [Data Loading](#data-loading)
  - [Combined Dataset](#combined-dataset)
  - [Forecasting Datasets](#forecasting-datasets)
- [ThetaData API](#thetadata-api)
  - [Request Parameters (Options)](#request-parameters-options)
  - [Quote Data](#quote-data)
  - [OHLCVC Data](#ohlcvc-data)
  - [Underlying Asset Data](#underlying-asset-data)
  - [Request Parameters (Stocks)](#request-parameters-stocks)

## Overview

The data pipeline is designed to:

1. Define option contracts based on parameters such as time-to-expiration, moneyness, etc.
2. Collect historical data for these contracts and their underlying stocks
3. Clean and preprocess the data
4. Generate features for machine learning models
5. Create appropriate PyTorch datasets and data loaders

## Contract Model

The foundation of our data pipeline is the [`Contract`](optrade/data/thetadata/contracts.py#L8) model, which inherits from PyDantic's `BaseModel`, and represents core parameters of the options contract:

```python
class Contract(BaseModel):
    """
    A Pydantic model representing an options contract with methods for optimal contract selection.

    The Contract class defines the structure of an options contract including the underlying security,
    dates, strike price, and other key parameters. It inherits from Pydantic's BaseModel for automatic
    validation and serialization.
    """

    root: str = Field(default="AAPL", description="Root symbol of the underlying security")
    start_date: str = Field(default="20241107", description="Start date in YYYYMMDD format")
    exp: str = Field(default="20241206", description="Expiration date in YYYYMMDD format")
    strike: int = Field(default=225, description="Strike price")
    interval_min: int = Field(default=1, description="Interval in minutes")
    right: str = Field(default="C", description="Option type (C for call, P for put)")
  
    ...
```

### Finding Optimal Contracts
The [`Contract`](optrade/data/thetadata/contracts.py) model includes a [`find_optimal`](optrade/data/thetadata/contracts.py#L25) method to identify appropriate contracts based on desired characteristics:

```python
@classmethod
def find_optimal(
    cls,
    root: str,
    start_date: str,
    interval_min: int,
    right: str,
    target_tte: int,
    tte_tolerance: Tuple[int, int],
    moneyness: str,
    target_band: float,
    hist_vol: Optional[float] = None,
    volatility_scaled: bool = True,
    volatility_scalar: float = 1.0,
) -> "Contract":
    """
    Find the optimal contract based on parameters like time-to-expiration,
    moneyness, and volatility considerations.
    """
    # Find the best expiration date based on target TTE
    exp, * = find_optimal_exp(...)
    
    # Find the best strike price based on moneyness and volatility
    strike = find_optimal_strike(...)
    
    # Return the constructed Contract instance
    return cls(...)
```
[`find_optimal`](optrade/data/thetadata/contracts.py#L25) first identifies valid expiration dates based on target time-to-expiration (TTE) referenced against ThetaData's [list of expirations](https://http-docs.thetadata.us/operations/get-v2-list-expirations.html) for a given root symbol. The optimal expiration date is then selected based on the target TTE and a tolerance window. After finding the expiration, the method then determines appropriate strike prices based on moneyness criteria (e.g., OTM, ATM, ITM) and a target band, referenced against ThetaData's [list of strikes](https://http-docs.thetadata.us/operations/get-v2-list-strikes.html) for a given root symbol. This target band can be set as a percentage difference of the underlying asset's price at the start of the contract using the `target_band` parameter, or it can be dynamically calculated based on the historical volatility by setting `volatility_scaled=True` and providing values for the `hist_vol` and `volatility_scalar` parameters.

## Contract Datasets

The [`ContractDataset`](optrade/src/preprocessing/data/datasets.py#L13) class generates datasets of contracts based on specified parameters:

```python
class ContractDataset:
    """
    A dataset containing options contracts generated with consistent parameters.

    Contracts are generated by starting from total_start_date and advancing by
    contract_stride days until reaching the last valid date that allows for
    contracts within the specified time-to-expiration tolerance.
    """

    def __init__(
        self,
        root: str = "AAPL",
        total_start_date: str = "20231107",
        total_end_date: str = "20241114",
        contract_stride: int = 5,
        interval_min: int = 1,
        right: str = "C",
        target_tte: int = 30,
        tte_tolerance: Tuple[int, int] = (25, 35),
        moneyness: str = "OTM",
        target_band: float = 0.05,
        volatility_scaled: bool = True,
        volatility_scalar: float = 1.0,
        hist_vol: Optional[float] = None,
        verbose: bool = False,
    ) -> None:
      ...
```

### Contract Dataset Generation
The ContractDataset class contains the [`generate_contracts`](optrade/src/preprocessing/data/datasets.py#L76) method which creates a series of contracts by starting from the initial date and advancing by a specified stride (in days), calling `Contract.find_optimal` at each selected date. This creates a series of contracts that meet the specified criteria and handles special cases like weekends and market holidays.

```python
def generate_contracts(self) -> "ContractDataset":
    """
    Generate all contracts in the dataset based on configuration parameters.
    """
    ...
    return self
```

## Feature Engineering

Our data pipeline includes specialized feature engineering components designed to capture temporal patterns in options data. The feature engineering process is built around three key modules:

### Time-to-Expiration Features

The [`get_tte_features`](optrade/src/preprocessing/features/tte_features.py) provides various transformations of time-to-expiration data to capture theta decay effects:

```python
def get_tte_features(
    df: pd.DataFrame,
    feats: List=["linear", "inverse", "sqrt", "inverse_sqrt", "exp_decay"],
    exp: str="20250117",
) -> pd.DataFrame:
    # Implementation...
```
This function offers several transformations:
- `linear`: Raw time-to-expiration in minutes
- `inverse`: 1/TTE (provides higher sensitivity as expiration approaches)
- `sqrt`: Square root of TTE (moderates the time decay effect)
- `inverse_sqrt`: 1/√TTE (combines benefits of inverse and square root)
- `exp_decay`: Exponential decay relative to contract length

### Datetime Features
The [`get_datetime_features`](optrade/src/preprocessing/features/datetime_features.py) function extracts cyclical patterns from timestamp data:

```python
def get_datetime_features(
    df: pd.DataFrame,
    feats: List=[
        "minuteofday",
        "sin_timeofday",
        "cos_timeofday",
        "dayofweek",
    ],
    dt_col: Optional[str] = "datetime",
    market_open_time: str = "09:30:00",
    market_close_time: str = "16:00:00",
) -> pd.DataFrame:
    # Implementation...
```
This function provides the following features:
 - `minuteofday`: Minutes since market open (0-389 for standard session)
 - `sin_timeofday` and `cos_timeofday`: Sine and cosine transformations of time of day, creating continuous circular features to capture intraday patterns
 - `dayofweek`: Day of the week (0=Monday, 4=Friday)

### Compiling All Features
The [`get_features`](optrade/src/preprocessing/features/get_features.py) function combines core data features with the engineered TTE and datetime features:
```python
def get_features(
    df: pd.DataFrame,
    core_feats: list,
    tte_feats: list,
    datetime_feats: list,
) -> pd.DataFrame:
    # Implementation...
```
which applies both TTE and datetime feature generators to the input DataFrame, derived from the `get_tte_features` and `get_datetime_features` functions, with selected features indicated by `tte_feats` and `datetime_feats` parameters. It also inclues
`core_feats` which comprise the core features of the options data and underlying stock data, including NBBO quotes and OHLCVC metrics detailed in section [ThetaData API](#thetadata-api).

## Data Loading
### Combined Dataset
Data loading is handled mainly by the [`get_combined_dataset`](optrade/src/preprocessing/data/dataloading.py#219) function, which creates a unified PyTorch dataset from multiple option contracts:

```python
def get_combined_dataset(
    contracts: ContractDataset,
    core_feats: list,
    tte_feats: list,
    datetime_feats: list,
    tte_tolerance: Tuple[int, int],
    clean_up: bool = True,
    offline: bool = False,
    intraday: bool = False,
    target_channels: list=[0],
    seq_len: int=100,
    pred_len: int=10,
    dtype: str="float64",
) -> Dataset:
    # Function implementation...
```
This function iterates through each contract in a `ContractDataset` object, and attempts to load data for each contract using the `get_data` function, with error handling for various data validation issues. Since the start of each option contract is not provided by ThetaData, it is not uncommon to request to the underlying data and option data for the same start date, only to have the option data start later because the contract wasn't issued at the start date provided. In this case, the function will adjust the start date of the contract to the first available option data. 


### PyTorch ForecastingDataset
Within `get_combined_dataset`, each contract's data is transformed into a `ForecastingDataset`, which inherits from PyTorch's `Dataset` class:

```python
class ForecastingDataset(Dataset):
    """

    Args:
        data (torch.Tensor): Tensor of shape (num_time_steps, num_features)
        seq_len (int): Length of the lookback window
        pred_len (int): Length of the forecast window
        target_channels (list): Channels to forecast. By default target_channels=[0], which corresponds to the
                                option midprice returns. If None, all channels will be returned for the target.
        dtype (str): Desired data type of the tensor.
    """

    def __init__(self, data, seq_len, pred_len, target_channels=[0], dtype="float32"):
        ...

    def __len__(self):
        return self.data.shape[0] - self.seq_len - self.pred_len

    def __getitem__(self, idx):
        input = self.data[idx:idx+self.seq_len]

        if self.target_channels:
            target = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len][self.target_channels]
        else:
            target = self.data[idx+self.seq_len:idx+self.seq_len+self.pred_len]

        return input, target
```
By creating a `ForecastingDataset` instance for each contract, we can now concatenate all of the datasets together to form a single dataset object, with temporal separation provided between each contract. The concatenation is handled naturally by the PyTorch's `ConcatDataset` class, which is the final output of `get_combined_dataset`.

## ThetaData API
We utilize the [ThetaData API](https://http-docs.thetadata.us/) to obtain high-frequency options data consolidated by the Options Price Reporting Authority (OPRA). The data includes quotes and OHLC metrics at 1-minute intervals during regular trading hours (9:30 AM - 4:00 PM EST). Note that this requires an active subscription to both the option and stock VALUE packages, although not free, are (relatively) cheap with respect to other financial market data providers. To run any of the data scripts found in [`optrade/data/thetadata/`](optrade/data/thetadata/), a ThetaData terminal must running.

### Request Parameters (Options)
Our implementation uses modified versions of the ThetaData API parameters for improved usability:
- `root`: Underlying security's root symbol
- `start_date`, `end_date`: Date range in YYYYMMDD format
- `exp`: Option expiration date (YYYYMMDD)
- `strike`: Strike price in dollars (converted from ThetaData's cents representation)
- `interval_min`: Sampling interval in minutes (converted from ThetaData's millisecond requirement)
- `right`: Option type ('C' for call, 'P' for put)

### Quote Data (NBBO)
The National Best Bid and Offer (NBBO) data represents the optimal available bid and ask prices across all exchanges at each interval:
- `datetime`: Timestamp in 'YYYY-MM-DD HH:MM:SS' format (converted from ThetaData's `ms_of_day` and `date` fields)
- Bid Information:
 - `bid`, `bid_size`: Best bid price and size
 - `bid_exchange`: Exchange identifier
 - `bid_condition`: Quote condition code
- Ask Information:
 - `ask`, `ask_size`: Best ask price and size
 - `ask_exchange`: Exchange identifier
 - `ask_condition`: Quote condition code

Note: If `exp` < `end_date`, data will be provided until the option expires (i.e. `exp`). For more details on the quote data format, see [`hist/options/quote`](https://http-docs.thetadata.us/operations/get-hist-option-quote.html).

### OHLCVC Data
Trade-based statistics filtered according to Securities Information Processor (SIP) rules to exclude misleading trades:
- `open`, `high`, `low`, `close`: Price metrics for the interval
- `volume`: Contract volume from eligible trades
- `count`: Total number of eligible trades

Note: For shorter intervals (e.g., `interval_min`=1), a significant portion of OHLCVC data may contain zeroes, i.e. no eligible trades ocurred within the time period due to low liquidity. Increasing `interval_min` will reduce this issue, but will not eliminate it (even for more liquid options). For more details on the OHLCVC data format, see [`hist/options/ohlc`](https://http-docs.thetadata.us/operations/get-hist-option-ohlc.html). To acquire both quote data and OHLCVC data for options, see [`optrade/data/thetadata/options.py`](optrade/data/thetadata/options.py).

### Underlying Asset Data
For the underlying securities, we collect analogous data through UTP and CTA feeds at 1-minute intervals.

### Request Parameters (Stocks)
- `root`: Security's root symbol
- `start_date`, `end_date`: Date range in YYYYMMDD format
- `interval_min`: Sampling interval in minutes

The underlying data includes both NBBO quotes and OHLCVC metrics in the same format as the options data. To acquire data for the underlying,
see `optrade/data/thetadata/stocks.py`. For more details on the data format, see [`hist/stocks/quote`](https://http-docs.thetadata.us/operations/get-v2-hist-stock-quote.html)
and [`hist/stocks/ohlc`](https://http-docs.thetadata.us/operations/get-v2-hist-stock-ohlc.html).

