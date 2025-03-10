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