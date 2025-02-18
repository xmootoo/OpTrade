# OpTrade
OpTrade is a framework designed for high-frequency forecasting of alpha term structures in American options markets. The framework leverages state-of-the-art deep learning architectures specialized for time series forecasting. This project has two objectives: $(\textbf{I})$ discovering alpha term structures to analyze market microstructure dynamics across various options contracts via forecasting, and $(\textbf{II})$ translating these insights into actionable trading signals.
Currently, the project is focused on completing objective $(\textbf{I})$, with objective $(\textbf{II})$ planned for implementation upon successful completion of the microstructure analysis framework.

## Table of Contents
1. [Market Data](#market-data)
2. [Installation](#installation)
3. [Contact](#contact)

## Market Data
We utilize the [ThetaData API](https://http-docs.thetadata.us/) to obtain high-frequency options data consolidated by the Options Price Reporting Authority (OPRA).
The data includes quotes and OHLC metrics at 1-minute intervals during regular trading hours (9:30 AM - 4:00 PM EST). Note that this requires an
active subscription to both the option and stock VALUE packages, although not free, are (relatively) cheap with respect to other financial market data providers.
To run any of the data scripts found in `optrade/data/thetadata/`, a ThetaData terminal must running.

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



Note: For shorter intervals (e.g., `interval_min`=1), a significant portion of OHLCVC data may contain zeroes, i.e. no eligible trades ocurred within the time period. Increasing `interval_min` will reduce this issue, but will not eliminate it (even for more liquid options). For more details on the OHLCVC data format, see [`hist/options/ohlc`](https://http-docs.thetadata.us/operations/get-hist-option-ohlc.html).
To acquire both quote data and OHLCVC data for options, see `optrade/data/thetadata/options.py`.

## Underlying Asset Data
For the underlying securities, we collect analogous data through UTP and CTA feeds at 1-minute intervals.

### Request Parameters (Stocks)
- `root`: Security's root symbol
- `start_date`, `end_date`: Date range in YYYYMMDD format
- `interval_min`: Sampling interval in minutes

The underlying data includes both NBBO quotes and OHLCVC metrics in the same format as the options data. To acquire data for the underlying,
see `optrade/data/thetadata/stocks.py`. For more details on the data format, see [`hist/stocks/quote`](https://http-docs.thetadata.us/operations/get-v2-hist-stock-quote.html)
and [`hist/stocks/ohlc`](https://http-docs.thetadata.us/operations/get-v2-hist-stock-ohlc.html).


## Installation
### Dependencies
- Python $\geq$ 3.11
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
