import yfinance as yf
import pandas as pd
import pandas_datareader.data as web
import statsmodels.api as sm
from datetime import datetime
import warnings
from typing import Dict, Any
warnings.filterwarnings('ignore', message='The argument \'date_parser\' is deprecated')

from optrade.data.thetadata.stocks import get_stock_data_eod

def get_fama_french_factors(
    root: str,
    start_date: str,
    end_date: str,
    mode: str = "3_factor",
) -> Dict[str, Any]:
    """
    Calculate Fama-French factor model exposures for a stock over the specified period.

    Args:
        root (str): Root symbol of the underlying security
        start_date (str): Start date in YYYYMMDD format
        end_date (str): End date in YYYYMMDD format
        mode (str): Mode for the Fama-French model ("3_factor" or "5_factor")

    Returns:
        Dictionary containing the factor betas:
        - market_beta: Market excess return sensitivity
        - size_beta: Small Minus Big (SMB) factor exposure
        - value_beta: High Minus Low (HML) book-to-market factor exposure
        - profitability_beta: Robust Minus Weak (RMW) profitability factor (5-factor only)
        - investment_beta: Conservative Minus Aggressive (CMA) investment factor (5-factor only)
        - r_squared: Proportion of return variation explained by the factors
    """

    # Suppress the date_parser deprecation warning
    warnings.filterwarnings('ignore', message='The argument \'date_parser\' is deprecated')

    # Shift the start_date by -1 day to get returns for the current day
    ff_start_date = datetime.strptime(start_date, "%Y%m%d")
    ff_end_date = datetime.strptime(end_date, "%Y%m%d")

    print(f"New start date: {start_date}")

    # Get stock data
    stock_data = get_stock_data_eod(
        root=root,
        start_date=start_date,
        end_date=end_date,
        clean_up=True,
        offline=False,
    )

    # Calculate daily returns
    stock_data["returns"] = stock_data["close"].pct_change().dropna()
    stock_data["Date"] = stock_data["datetime"].dt.date

    # Drop NaN
    stock_data = stock_data.dropna()

    # Drop all other columns besides Date and returns
    stock_data = stock_data[["Date", "returns"]]

    # Get Fama-French factor data based on mode
    if mode == "3_factor":
        ff_data = web.DataReader("F-F_Research_Data_Factors_daily", "famafrench",
                                start=ff_start_date, end=ff_end_date)[0]
        factor_columns = ["Mkt-RF", "SMB", "HML"]
    elif mode == "5_factor":  # 5_factor
        ff_data = web.DataReader("F-F_Research_Data_5_Factors_2x3_daily", "famafrench",
                                start=ff_start_date, end=ff_end_date)[0]
        factor_columns = ["Mkt-RF", "SMB", "HML", "RMW", "CMA"]
    else:
        raise ValueError(f"Invalid mode: {mode}. Choose '3_factor' or '5_factor'.")

    # Convert percentages to decimals
    ff_data = ff_data / 100

    # Truncate ff_data to the same date range as stock_data["Date"]
    valid_dates = pd.DatetimeIndex(stock_data["Date"])
    ff_data = ff_data.loc[ff_data.index.intersection(valid_dates)]

    # Reset index to make Date a column
    ff_data_reset = ff_data.reset_index()
    ff_data_reset['Date'] = ff_data_reset['Date'].dt.date

    # Merge stock_data with ff_data on Date
    aligned_data = pd.merge(stock_data, ff_data_reset, on="Date", how="inner")

    # Linear regression
    X = aligned_data[factor_columns]
    X = sm.add_constant(X)
    y = aligned_data["returns"] - aligned_data['RF']  # Excess return

    # Ensure y is 1-dimensional
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]

    # Run regression
    model = sm.OLS(y, X).fit()

    # Prepare results
    result = {
        'market_beta': model.params.get('Mkt-RF', None),
        'size_beta': model.params.get('SMB', None),
        'value_beta': model.params.get('HML', None),
        'r_squared': model.rsquared
    }

    # Add additional factors for 5-factor model
    if mode == "5_factor":
        result['profitability_beta'] = model.params.get('RMW', None)
        result['investment_beta'] = model.params.get('CMA', None)

    return result

if __name__ == "__main__":

    def test_fama_french_5factor():
        """
        Test the Fama-French 5-factor model exposure calculation.
        """
        # Choose a well-known stock
        symbol = "AAPL"

        # Set test period (1 year)
        start_date = "20230101"  # YYYYMMDD format
        end_date = "20231231"    # YYYYMMDD format

        print(f"Testing Fama-French 5-factor exposures for {symbol} from {start_date} to {end_date}")

        # Calculate Fama-French exposures directly using the symbol
        result = get_fama_french_factors(symbol, start_date, end_date, mode="5_factor")

        # Print results
        print("\nFama-French 5-Factor Exposures:")
        print(f"Market Beta: {result['market_beta']:.4f}" if result['market_beta'] is not None else "Market Beta: None")
        print(f"Size Beta (SMB): {result['size_beta']:.4f}" if result['size_beta'] is not None else "Size Beta: None")
        print(f"Value Beta (HML): {result['value_beta']:.4f}" if result['value_beta'] is not None else "Value Beta: None")
        print(f"Profitability Beta (RMW): {result['profitability_beta']:.4f}" if result['profitability_beta'] is not None else "Profitability Beta: None")
        print(f"Investment Beta (CMA): {result['investment_beta']:.4f}" if result['investment_beta'] is not None else "Investment Beta: None")
        print(f"R-squared: {result['r_squared']:.4f}" if result['r_squared'] is not None else "R-squared: None")

        # Interpret results
        if result['market_beta'] is not None:
            print("\nInterpretation:")

            # Market beta interpretation
            if result['market_beta'] > 1.1:
                print("- High market sensitivity (beta > 1.1)")
            elif result['market_beta'] < 0.9:
                print("- Low market sensitivity (beta < 0.9)")
            else:
                print("- Market sensitivity close to market (beta ≈ 1)")

            # Size factor interpretation
            if result['size_beta'] > 0.2:
                print("- Behaves more like small-cap stocks (positive SMB exposure)")
            elif result['size_beta'] < -0.2:
                print("- Behaves more like large-cap stocks (negative SMB exposure)")
            else:
                print("- Neutral size exposure")

            # Value factor interpretation
            if result['value_beta'] > 0.2:
                print("- Behaves more like value stocks (positive HML exposure)")
            elif result['value_beta'] < -0.2:
                print("- Behaves more like growth stocks (negative HML exposure)")
            else:
                print("- Neutral value/growth exposure")

            # Profitability factor interpretation
            if result['profitability_beta'] > 0.2:
                print("- Associated with robust profitability (positive RMW exposure)")
            elif result['profitability_beta'] < -0.2:
                print("- Associated with weak profitability (negative RMW exposure)")
            else:
                print("- Neutral profitability exposure")

            # Investment factor interpretation
            if result['investment_beta'] > 0.2:
                print("- Associated with conservative investment (positive CMA exposure)")
            elif result['investment_beta'] < -0.2:
                print("- Associated with aggressive investment (negative CMA exposure)")
            else:
                print("- Neutral investment exposure")

        return result

    if __name__ == "__main__":
        test_fama_french_5factor()
