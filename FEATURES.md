# Features
This documentation covers all features for individual option contracts, including the mathematics and engineering behind each feature. For a single option contract, the feature engineering process generates a matrix $\mathbf{X} \in \mathbb{R}^{M \times T}$ where $M \in \mathbb{N}$ is the number of features, and $T \in \mathbb{N}$ is the number of time points in the time series.
Each row of this matrix represents a specific feature tracked over time, while each column captures the complete feature state at a particular moment. 

This structure allows time series models to learn both temporal dynamics and spatial (feature-wise) relationships simultaneously. In contrast to "compressed" feature representations (e.g., averaged or aggregated features), we provide a richer view of the data, which is more suitable and beneficial for modern deep learning approaches.

## Table of Contents 
- [Usage](#usage)
- [Core Features](#core-features)
- [Time-to-Expiration Features](#tte-features)
- [Datetime Features](#datetime-features)

## Usage
Featurization is handled by [`get_features`](optrade/src/preprocessing/features/get_features.py), which includes several different subsets of features to choose from:
```py
def get_features(
    df: pd.DataFrame,
    core_feats: List[str],
    tte_feats: List[str],
    datetime_feats: List[str],
    strike: Optional[int]=None,
) -> pd.DataFrame:

    """
    Selects features from a DataFrame based on a list of feature names (i.e. columns).
    """
    ...
```
This function processes the output dataframe from the [`get_data`](optrade/data/thetadata/get_data.py) function (detailed in `DATA.md`) and extracts specific feature sets based on the following parameter lists:

- `core_feats (List[str])`: Core market features including standard NBBO and OHLCVC data, plus derived predictors critical for securities forecasting such as mid-price, returns, and order book metrics.

- `tte_feats (List[str])`: Time-to-expiration features that capture various transformations of option contract maturity, helping models learn theta decay patterns.

- `datetime_feats (List[str])`: Temporal context features like minute of day or day of week that help capture market seasonality effects (opening/closing patterns, lunch hour dips, etc.) independent of option-specific data.

The function returns a processed dataframe containing only the requested feature sets, ready for model training or prediction. The full list of available parameters include

## Core Features
To learn about the source data, please visit `DATA.md`, which details ThetaData's API and data preparation. 

#### Basic Features (NBBO and OHLCVC)
Cleaned NBBO and OHLCVC features for both the option and the underlying include:
- `datetime`
- `{asset}_bid_size`
- `{asset}_bid_exchange`
- `{asset}_bid`
- `{asset}_bid_condition`
- `{asset}_ask_size`
- `{asset}_ask_exchange`
- `{asset}_ask`
- `{asset}_ask_condition`
- `{asset}_open`
- `{asset}_high`
- `{asset}_low`
- `{asset}_close`
- `{asset}_volume`
- `{asset}_count`

where `{asset}` can be either "option" or "stock".

#### Custom Features
Using our primary source data, we can also can construct the following features:

- `mid_price`: The average of the bid and ask prices, which is the commonly used measure of market value for the option and stock:

$$
\begin{align*}
    \text{option\_mid\_price} &= V_t = \frac{V_t^b+V_t^a}{2} \\
    \text{stock\_mid\_price} &= S_t = \frac{S_t^b+S_t^a}{2} 
\end{align*}
$$

- `returns`: The relative change in mid price from the previous period, which is more stable (i.e. stationary) and easier to predict than raw prices.

$$
\begin{align*}
    \text{option\_returns} &= \frac{V_{t} - V_{t-1}}{V_{t-1}} \\
    \text{stock\_returns} &= \frac{S_{t} - S_{t-1}}{S_{t-1}}
\end{align*}
$$

If `option_returns` or `stock_returns` is included in the list of features, by convention we remove the first market open, due to the presence of 0% returns.

- `lob_imbalance`: The Limit Order Book (LOB) imbalance measures buying or selling pressure, indicating potential price movement direction.

$$
\begin{align*}
    \text{option\_lob\_imbalance} &= \frac{Q_t^{b,V} - Q_t^{a,V}}{Q_t^{b,V} + Q_t^{a,V}} \\
    \text{stock\_lob\_imbalance} &= \frac{Q_t^{b,S} - Q_t^{a,S}}{Q_t^{b,S} + Q_t^{a,S}} 
\end{align*}
$$
Where $Q_t^{b,V}$ and $Q_t^{a,V}$ represent the bid and ask quantities for the option at time $t$, and $Q_t^{b,S}$ and $Q_t^{a,S}$ represent the bid and ask quantities for the underlying stock.

- `quote_spread`: The normalized bid-ask spread, which indicates liquidity and trading costs.

$$
\begin{align*}
    \text{option\_quote\_spread} &= \frac{V_t^a - V_t^b}{V_t} \\
    \text{stock\_quote\_spread} &= \frac{S_t^a - S_t^b}{S_t}
\end{align*}
$$

- `distance_to_strike`: The raw difference between the strike price and the current stock price, directly measuring proximity to the exercise threshold.
$$
\begin{align*}
\text{distance\_to\_strike} = K - S_t
\end{align*}
$$
This feature is useful when absolute dollar distances matter for risk management and provides an intuitive measure that directly relates to option pricing models.

- `moneyness`: The logarithm of the ratio between the current stock price and the option's strike price, indicating whether the option is profitable to exercise.
$$
\begin{align*}
\text{moneyness} = \log\left(\frac{S_t}{K}\right)
\end{align*}
$$
Compared to `distance_to_strike`, this feature is normalized and scale-invariant, and therefore better suited for comparing across different stocks, time periods, or price ranges.


## Time-to-Expiration Features
The [`get_tte_features`](optrade/src/preprocessing/features/tte_features.py) is a component function of [`get_features`](optrade/src/preprocessing/features/get_features.py) providing various transformations of time-to-expiration data to capture theta decay effects (the reduction in option value over time):
```py
def get_tte_features(
    df: pd.DataFrame,
    feats: List=[
        "tte",
        "inverse",
        "sqrt",
        "inverse_sqrt",
        "exp_decay"
    ],
    exp: str="20250117",
) -> pd.DataFrame:
    """ Generate Time to Expiration (TTE) features for a given DataFrame."""
    ...
```
These features include:
- `tte`:  Raw time-to-expiration, providing baseline measurement of remaining time.
$$\text{tte} = t_\text{expiration} - t_{\text{current}}$$
where $t_{\text{current}}$ is the current time (measured in units based on `interval_min`). For example, if `interval_min=1`, then `tte` provides the current time-to-expiration in minutes.

- `tte_inverse`: The reciprocal of time-to-expiration, increasing sensitivity as expiration approaches:
$$\text{tte\_inverse} = \frac{1}{\text{tte}}$$
- `tte_sqrt`: The square root of time-to-expiration, moderating decay and aligning with Black-Scholes:
$$\text{tte\_sqrt} = \sqrt{\text{tte}}$$
- `tte_inverse_sqrt`: Inverse square root of time-to-expiration, balanced approach to capture accelerating decay.
$$\text{tte\_inverse\_sqrt} = \frac{1}{\sqrt{\text{tte}}}$$
which is more sensitive near the expiration than `tte_inverse`.

- `tte_exp_decay`: Exponential decay relative to contract length, normalizing decay across options of different durations (useful for modeling multiple contracts):
$$\text{tte\_exp\_decay} = \exp\left( -\frac{\text{tte}}{\text{contract\_length}} \right) $$

where $\text{contract\_length}$ is the total duration from issuance to expiration, resulting in a normalized value $0 \leq (\frac{\text{tte}}{\text{contract\_length}}) \leq 1$.


## Datetime Features
`datetime_feats` provides temporal context features that capture market seasonality effects independent of option-specific data. These features help models learn intraday patterns and weekly option expiration cycles, which are critical for short-term options forecasting.

The datetime features are handled by the [`get_datetime_features`](optrade/src/preprocessing/features/datetime_features.py) component function:

```py
def get_datetime_features(
    df: pd.DataFrame,
    feats: List=[
        "minute_of_day",
        "sin_minute_of_day",
        "cos_minute_of_day",
        "day_of_week",
        "hour_of_week",
        "sin_hour_of_week",
        "cos_hour_of_week",
    ],
    dt_col: Optional[str] = "datetime",
    market_open_time: str = "09:30:00",
    market_close_time: str = "16:00:00",
) -> pd.DataFrame:
    """Generates optimized datetime features for short-term options forecasting."""
    ...
```
This function processes temporal data to extract cyclic patterns and market-specific time features:
- `minute_of_day`: Position within the trading day (e.g., 0-389 minutes for `interval_min=1`), where 0 represents market open.
$$\text{minute\_of\_day} = \text{current\_time\_in\_minutes} - \text{market\_open\_in\_minutes}$$

- `sin_minute_of_day` and `cos_minute_of_day`: Sine and cosine transformations of time of day, providing continuous circular features that capture daily cyclical patterns:
$$\text{normalized\_time} = 2\pi \times \frac{\text{minute\_of\_day}}{\text{trading\_minutes\_per\_day}}$$
$$\text{sin\_minute\_of\_day} = \sin(\text{normalized\_time})$$
$$\text{cos\_minute\_of\_day} = \cos(\text{normalized\_time})$$

These transformations help capture recurring patterns at specific times of the trading day (e.g., opening/closing volatility, lunch hour dips).
- `day_of_week`: Trading day number (0=Monday, 4=Friday), capturing weekly expiration effects.

- `hour_of_week`: Hour position as a proportion of the trading week (0.0-1.0), providing a continuous measure of weekly progress:
$$\text{hour\_of\_week} = \frac{\text{total\_trading\_hours\_elapsed\_this\_week}}{\text{total\_trading\_hours\_per\_week}}$$

- `sin_hour_of_week` and `cos_hour_of_week`: Sine and cosine transformations of the hour of week, providing continuous circular features that capture weekly cyclical patterns:
$$\text{normalized\_week\_time} = 2\pi \times \text{hour\_of\_week}$$
$$\text{sin\_hour\_of\_week} = \sin(\text{normalized\_week\_time})$$
$$\text{cos\_hour\_of\_week} = \cos(\text{normalized\_week\_time})$$

These weekly transformations are particularly important for capturing option expiration cycles and day-of-week effects that are common in derivatives markets. All datetime features are prefixed with `dt_` in the resulting DataFrame to distinguish them from other feature types.

## Future Implementations
We plan to implement the following features in the future:
* Implied Volatility
* The Greeks ($\Delta, \Gamma, \Theta, \nu, \rho,$)