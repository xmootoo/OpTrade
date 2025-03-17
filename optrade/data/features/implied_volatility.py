import QuantLib as ql
import pandas as pd
# Derived From: https://www.pyquantnews.com/the-pyquant-newsletter/how-easily-solve-volatility-for-american-options

# TODO: Test and implement greeks. IV is too much, due to numerical optimization issues.
def calculate_greeks(
    stock_price: float,
    strike: float,
    risk_free_rate: float,
    dividend_yield: float,
    volatility: float,  # Use historical vol if IV is problematic
    tte: float,  # Time to expiry in days
):
    """Calculate option Greeks using QuantLib"""

    # Set up QuantLib environment
    today = ql.Date().todaysDate()
    ql.Settings.instance().evaluationDate = today
    day_count = ql.Actual365Fixed()

    # Convert tte from days to years
    tte_years = tte / 365.0

    # Set up option
    expiration_date = today + ql.Period(int(tte), ql.Days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)

    # Can use European option for Greeks approximation
    exercise = ql.EuropeanExercise(expiration_date)
    option = ql.VanillaOption(payoff, exercise)

    # Set up market data
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(stock_price))
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, risk_free_rate, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, dividend_yield, day_count)
    )
    vol_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, ql.NullCalendar(), volatility, day_count)
    )

    # Create process and pricing engine
    process = ql.BlackScholesMertonProcess(
        spot_handle, dividend_ts, risk_free_ts, vol_handle
    )
    engine = ql.AnalyticEuropeanEngine(process)
    option.setPricingEngine(engine)

    # Calculate greeks
    results = {
        'price': option.NPV(),
        'delta': option.delta(),
        'gamma': option.gamma(),
        'theta': option.theta(),
        'vega': option.vega()
    }

    return results

def get_implied_volatility(
    df,
    dividend_yield: float=0.0052,
    volatility_guess: float=0.20,
    strike: int=225,
) -> pd.DataFrame:
    """
    Calculate implied volatility for a dataframe of options using lambda function.

    Args:
        df (pandas.DataFrame): DataFrame with columns:
            - "stock_price"
            - "option_price"
            - "risk_free_rate"
            - "tte"
        dividend_yield (float): Fixed dividend yield, typically the last earnings report annual yield.
        volatility_guess (float): Initial volatility guess for all calculations
        strike (int): Option strike price in dollars.

    Returns:
        df (pd.DataFrame): Original dataframe with an additional "implied_volatility" column
    """
    result_df = df.copy()

    # Use fixed strike for all calculations
    result_df["implied_volatility"] = result_df.apply(
        lambda row: single_implied_volatility(
            stock_price=row["stock_mid_price"],
            option_price=row["option_mid_price"],
            risk_free_rate=row["risk_free_rate"],
            dividend_yield=dividend_yield,
            tte=row["tte_linear"]/(24*60), # Convert minutes to days
            strike=strike,
            volatility_guess=volatility_guess
        ),
        axis=1
    )
    return result_df

def single_implied_volatility(
    stock_price: float,
    option_price: float,
    risk_free_rate: float,
    dividend_yield: float,
    tte: float,
    strike: int,
    volatility_guess: float=0.20,
) -> float:
    """
    Calculate the implied volatility of an American option using QuantLib.

    Args:
        stock_price (float): Current stock price in dollars.
        option_price (float): Market price of the option in dollars.
        risk_free_rate (float): Risk-free interest rate (annualized) as decimal from 0 to 1.
        dividend_yield (float): Dividend yield (annualized) as a decimal from 0 to 1.
        tte (float): Time to expiration in days.
        strike (int): Option strike price in dollars.
        volatility_guess (float, optional): Initial volatility guess, defaults to 0.20

    Returns:
        Implied volatility as a float.
    """

    # Set up the environment for valuing the option
    calendar = ql.NullCalendar()
    day_count = ql.Actual360()
    today = ql.Date().todaysDate()

    ql.Settings.instance().evaluationDate = today

    # Create term structures
    risk_free_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, risk_free_rate, day_count)
    )
    dividend_ts = ql.YieldTermStructureHandle(
        ql.FlatForward(today, dividend_yield, day_count)
    )

    # Create spot price handle
    spot_handle = ql.QuoteHandle(ql.SimpleQuote(stock_price))

    # Create volatility handle
    vol_quote = ql.SimpleQuote(volatility_guess)
    volatility_handle = ql.BlackVolTermStructureHandle(
        ql.BlackConstantVol(today, calendar, ql.QuoteHandle(vol_quote), day_count)
    )

    # Create the option and set the exercise type
    expiration_date = today + ql.Period(int(tte), ql.Days)
    payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike)
    exercise = ql.AmericanExercise(today, expiration_date)
    american_option = ql.VanillaOption(payoff, exercise)

    # Set up the process and pricing engine
    bsm_process = ql.BlackScholesMertonProcess(
        spot_handle, dividend_ts, risk_free_ts, volatility_handle
    )
    engine = ql.BinomialVanillaEngine(bsm_process, "crr", 1000)
    american_option.setPricingEngine(engine)

    # Calculate implied volatility
    implied_volatility = american_option.impliedVolatility(
        option_price, bsm_process, 1e-4, 1000, 1e-8, 4.0
    )

    return implied_volatility

# Example usage
if __name__ == "__main__":

    from optrade.data.thetadata.get_data import load_all_data
    from optrade.data.thetadata.contracts import Contract
    from optrade.preprocessing.features.transform_features import transform_features

    contract = Contract(
        root="AAPL",
        start_date="20241107",
        exp="20241220",
        strike=225,
        interval_min=1,
        right="C"
    )

    df = load_all_data(contract=contract, clean_up=False, offline=True)

    core_feats = [
        "option_mid_price",
        "stock_mid_price",
    ]
    tte_feats = ["linear"]
    datetime_feats = ["sin_timeofday", "cos_timeofday", "dayofweek"]


    df = transform_features(
        df=df,
        core_feats=core_feats,
        tte_feats=tte_feats,
        datetime_feats=datetime_feats,
        strike=contract.strike)

    # Add risk_free_rate column
    df["risk_free_rate"] = 0.0525

    # # Sample dataframe
    # sample_data = {
    #     "stock_mid_price": [188.64, 190.25, 187.30],
    #     "option_mid_price": [11.05, 12.50, 10.75],
    #     "risk_free_rate": [0.0525, 0.0530, 0.0520],
    #     "tte": [148*24*60, 92*24*60, 214*24*60], # in minutes
    # }

    # df = pd.DataFrame(sample_data)

    # Calculate with fixed strike
    result_fixed = get_implied_volatility(df=df, strike=contract.strike)
    print("\nResults with strike (190):")
    print(result_fixed[["stock_mid_price", "implied_volatility"]])

# # Example usage:
# if __name__ == "__main__":
#     iv = single_implied_volatility(
#         stock_price=188.64,
#         option_price=11.05,
#         risk_free_rate=0.0525,
#         dividend_yield=0.0052,
#         tte=148,
#         strike=190,
#         volatility_guess=0.20
#     )
#     print(f"Implied Volatility: {iv:.4f}")

# # Define the option parameters and market data
# spot_price = 188.64
# risk_free_rate = 0.0525
# dividend_yield = 0.0052
# volatility = 0.20
# days_to_maturity = 148
# strike_price = 190
# option_price = 11.05


# # Next we set up the environment for valuing the option
# calendar = ql.NullCalendar()
# day_count = ql.Actual360()
# today = ql.Date().todaysDate()
# print(f"Today: {today}")
# ql.Settings.instance().evaluationDate = today

# risk_free_ts = ql.YieldTermStructureHandle(
#     ql.FlatForward(today, risk_free_rate, day_count)
# )
# dividend_ts = ql.YieldTermStructureHandle(
#     ql.FlatForward(today, dividend_yield, day_count)
# )
# spot_handle = ql.QuoteHandle(ql.SimpleQuote(spot_price))

# # Create volatility handle - THIS WAS MISSING
# vol_quote = ql.SimpleQuote(volatility)
# volatility_handle = ql.BlackVolTermStructureHandle(
#     ql.BlackConstantVol(today, calendar, ql.QuoteHandle(vol_quote), day_count)
# )

# # Create the option and set the exercise type
# expiration_date = today + ql.Period(days_to_maturity, ql.Days)
# print(f"Expiration Date: {expiration_date}")
# payoff = ql.PlainVanillaPayoff(ql.Option.Call, strike_price)
# exercise = ql.AmericanExercise(today, expiration_date)
# american_option = ql.VanillaOption(payoff, exercise)

# # Implied volatility calculation
# bsm_process = ql.BlackScholesMertonProcess(
#     spot_handle, dividend_ts, risk_free_ts, volatility_handle
# )
# engine = ql.BinomialVanillaEngine(bsm_process, "crr", 1000)
# american_option.setPricingEngine(engine)

# implied_volatility = american_option.impliedVolatility(
#     option_price, bsm_process, 1e-4, 1000, 1e-8, 4.0
# )

# print(f"Implied Volatility: {implied_volatility:.4f}")
