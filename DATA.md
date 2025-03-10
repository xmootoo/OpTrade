# Data Pipeline
This document details the data pipeline for options forecasting and trading, which includes information on how to generate option contracts based on user-input parameters (e.g., time-to-expiration, moneyness), request historical data for these contracts and their underlying stocks, and then clean, process, and transform this data into features suitable for machine learning models.

## Table of Contents
- [Data Sources (ThetaData API)](#thetadata-api)
  - [Option Data](#option-data) 
  - [Underlying Security Data](#underlying-asset-data)
  - [Data Usage Notes](#data-usage-notes)
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


## Data Sources  (ThetaData API)
### Option Data
We utilize the [ThetaData API](https://http-docs.thetadata.us/) to obtain options data consolidated by the Options Price Reporting Authority (OPRA). The data includes quotes and OHLC metrics at 1-minute intervals during regular trading hours (9:30 AM - 4:00 PM EST). Note that this requires an active subscription to both the option and stock VALUE packages, although not free, are (relatively) cheap in comparison to other financial market data providers. To run any of the data scripts found in [`optrade/data/thetadata/`](optrade/data/thetadata/), a ThetaData terminal must running. The function [`get_option_data`](optrade/data/thetadata/options.py): 

```py
def get_option_data(
    root: str="AAPL", # Underlying security's root symbol
    start_date: str="20241107",
    end_date: str="20241107",
    exp: Optional[str]="20250117", # Epiration date
    strike: int=225, # Strike in dollars
    interval_min: int=1, # Interval resolution in minutes
    right: str="C", # Right: "C" for call and "P" for put
    ...,
) -> pd.DataFrame:

    """
     historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    options across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).
    """
    ...
```
returns a Pandas `DataFrame` including the National Best Bid and Offer (NBBO) data representing the optimal available bid and ask prices across all exchanges at each interval:
- `datetime`: Timestamp in 'YYYY-MM-DD HH:MM:SS' format (converted from ThetaData's `ms_of_day` and `date` fields)
- Bid Information:
 - `bid`, `bid_size`: Best bid price and size
 - `bid_exchange`: Exchange identifier
 - `bid_condition`: Quote condition code
- Ask Information:
 - `ask`, `ask_size`: Best ask price and size
 - `ask_exchange`: Exchange identifier
 - `ask_condition`: Quote condition code

in addition to OHLCVC transaction data (filtered according to Securities Information Processor (SIP) rules):
- `open`, `high`, `low`, `close`: Price metrics for the interval
- `volume`: Contract volume from eligible trades
- `count`: Total number of eligible trades

For more details see [`ThetaData (Option Quotes)`](https://http-docs.thetadata.us/operations/get-hist-option-quote.html) and [`Thetaata (Option OHLCVC)`](https://http-docs.thetadata.us/operations/get-hist-option-ohlc.html).


### Underlying Security Data
For the underlying security, the [`get_stock_data`](optrade/data/thetadata/stocks.py) function returns analogous data through UTP and CTA feeds with the lowest resolution at 1-minute intervals using the ThetaData API:
```py
def get_stock_data(
    root: str="AAPL",
    start_date: str="20231107",
    end_date: str="20231107",
    interval_min: int=1,
    save_dir: str="../historical_data/stocks",
    clean_up: bool=False,
    offline: bool=False,
) -> pd.DataFrame:

    """
    Requests historical quote-level data (NBBO) and OHLC (Open High Low Close) from ThetaData API for
    stocks across multiple exchanges, aggregated by interval_min (lowest resolution: 1min).
```
This returns both NBBO quotes and OHLCVC metrics in the same format as the options data. For more details see [`ThetaData (Stock Quotes)`](https://http-docs.thetadata.us/operations/get-v2-hist-stock-quote.html)
and [`ThetaData (Stock OHLCVC)`](https://http-docs.thetadata.us/operations/get-v2-hist-stock-ohlc.html).


### Data Usage Notes & Information
* If `exp` < `end_date`, data will be provided until the option expires (i.e. final date is `exp`). 
* If `start_date` is requested before the option contract begins, the function will throw a `DataValidationError` (this is our validation, not ThetaData API's). This error will notify the user that their requested `start_date` is too early and will provide the actual start date of the contract. Note that ThetaData API does not explicitly provide contract inception dates; instead, it simply returns data from the earliest available date for that contract, requiring users to determine the true contract start date through trial and error.
* All individual equity options (e.g., AAPL, MSFT) are **American options**, allowing for exercise at any time prior to expiration. However, all index options (e.g., SPX, VIX) are **European options**, which can only be exercised at expiration.
* For shorter intervals (e.g., `interval_min`=1), a significant portion of OHLCVC data may contain zeroes, i.e. no eligible trades ocurred within the time period due to low liquidity. Increasing `interval_min` will reduce this issue, but will not eliminate it (even for more liquid options).


## Contract Model
The foundation of our data pipeline is the [`Contract`](optrade/data/thetadata/contracts.py#L8) model, which inherits from PyDantic's `BaseModel`, and represents core parameters of the options contract. This serves as the essential building block for researchers and traders to select individual contracts and generate datasets containing multiple contracts with similar parameters.

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
The [`find_optimal`](optrade/data/thetadata/contracts.py#L25) method identifies valid expiration dates based on target time-to-expiration (TTE) referenced against ThetaData's [list of expirations](https://http-docs.thetadata.us/operations/get-v2-list-expirations.html) for a given security. The optimal expiration date is selected based on the target TTE and a tolerance window. After finding the expiration, the method determines appropriate strike prices based on moneyness criteria (e.g., OTM, ATM, ITM) and a target band, referenced against ThetaData's [list of strikes](https://http-docs.thetadata.us/operations/get-v2-list-strikes.html) for the security. This target band can be set as a percentage difference of the underlying asset's price at the start of the contract using the `target_band` parameter, or it can be dynamically calculated based on the historical volatility by setting `volatility_scaled=True` and providing values for the `hist_vol` and `volatility_scalar` parameters.

## Contract Datasets
The [`ContractDataset`](optrade/src/preprocessing/data/datasets.py#L13) class generates a dataset of `Contract` models based on specified parameters:

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
The `ContractDataset` class contains the [`generate_contracts`](optrade/src/preprocessing/data/datasets.py#L76) method which creates a series of `Contract` modles by starting from the initial date and advancing by a specified stride (in days), calling `Contract.find_optimal` at each selected date. This creates a series of contracts that meet the specified criteria and handles special cases such as weekends and market holidays.

```python
def generate_contracts(self) -> "ContractDataset":
    """
    Generate all contracts in the dataset based on configuration parameters.
    """
    ...
    return self
```

## Feature Engineering
Our data pipeline includes specialized feature engineering designed to capture important predictive. The feature engineering process is detailed in [`FEATURES.md`](FEATURES.md)

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
This function iterates through each contract in a `ContractDataset` object, and attempts to load data for each contract using the [`get_data`](optrade/data/thetadata/get_data.py) function, with error handling for various data validation issues. Since the start of each option contract is not provided by ThetaData, it is not uncommon to request to the underlying data and option data for the same start date, only to have the option data start later because the contract wasn't issued at the start date provided. In this case, the function will adjust the start date of the contract to the first available option data. 


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



