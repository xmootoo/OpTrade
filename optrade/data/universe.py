from typing import List, Optional, Dict, Tuple, Union

import pandas as pd
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
from rich.console import Console
from rich.table import Table
from optrade.data.contracts import ContractDataset

from optrade.data.thetadata import load_stock_data_eod
from optrade.analysis.factors import (
    get_factor_exposures,
    factor_categorization,
)
from optrade.utils.stock_categories import (
    ThreeFactorLevel,
    FiveFactorLevel,
    SectorType,
    IndustryType,
)
from optrade.data.contracts import get_contract_datasets
from optrade.data.forecasting import (
    get_forecasting_dataset,
    get_forecasting_loaders,
)

class Universe:
    def __init__(
        self,
        start_date: str,
        end_date: str,
        sp_500: bool = False,
        nasdaq_100: bool = False,
        dow_jones: bool = False,
        candidate_roots: Optional[List[str]] = None,
        volatility: Optional[str] = None,
        pe_ratio: Optional[str] = None,
        debt_to_equity: Optional[str] = None,
        beta: Optional[str] = None,
        market_cap: Optional[str] = None,
        sector: Optional[str] = None,
        industry: Optional[str] = None,
        dividend_yield: Optional[str] = None,
        earnings_volatility: Optional[str] = None,
        market_beta: Optional[str] = None,
        size_beta: Optional[str] = None,
        value_beta: Optional[str] = None,
        profitability_beta: Optional[str] = None,
        investment_beta: Optional[str] = None,
        momentum_beta: Optional[str] = None,
        all_metrics: bool = False,
        save_dir: Optional[str] = None,
        verbose: bool = False,
        dev_mode: bool = False,
    ) -> None:
        """A class for defining the universe of stocks and options for data retrieval and analysis.

        This class contains parameters for filtering stocks based on various factors
        and selecting options contracts based on specific criteria.

        Attributes:
            start_date (str, optional): Start date for data retrieval in YYYYMMDD format.
            end_date (str, optional): End date for data retrieval in YYYYMMDD format.
            sp_500 (bool): If True, use S&P 500 stocks as the candidate universe. Default is False.
            nasdaq_100 (bool): If True, use NASDAQ 100 stocks as the candidate universe. Default is False.
            dow_jones (bool): If True, use Dow Jones Industrial Average stocks as the candidate universe. Default is False.
            candidate_roots (list, optional): Candidate root symbols to be filtered by other parameters.
                Used only if no collection (sp_500, nasdaq_100, etc.) is selected.
            volatility (str, optional): The volatility of the stock.
                Options: 'low', 'medium', 'high'. Based on the terciles of volatility from the candidate universe.
            pe_ratio (str, optional): The P/E ratio of the stock.
                Options: 'low', 'medium', 'high'. Based on the terciles of P/E ratio from the candidate universe.
            debt_to_equity (str, optional): The debt to equity ratio of the stock.
                Options: 'low', 'medium', 'high'. Based on the terciles of debt to equity from the candidate universe.
            beta (str, optional): The beta of the stock.
                Options: 'low', 'medium', 'high'. Based on the terciles of beta from the candidate universe.
            market_cap (str, optional): The market cap of the stock.
                Options: 'low', 'medium', 'high'. Based on the terciles of market cap from the candidate universe.
            sector (str, optional): The sector of the stock.
                Options: 'tech', 'healthcare', 'financial', 'consumer_cyclical',
                'consumer_defensive', 'industrial', 'energy', 'materials', 'utilities',
                'real_estate', 'communication'.
            industry (str, optional): The industry of the stock matching Yahoo Finance classifications.
            dividend_yield (str, optional): The dividend yield of the stock.
                Options: 'low', 'medium', 'high'. Based on the terciles of dividend yield from the candidate universe.
            earnings_volatility (str, optional): The earnings volatility of the stock.
                Options: 'low', 'medium', 'high'. Based on the terciles of earnings volatility from the candidate universe.
            market_beta (str, optional): The market beta of the stock.
                Options: 'high', 'low', 'neutral'. Based on the absolute thresholds of < 0.9 and > 1.1.
            size_beta (str, optional): The size beta of the stock.
                Options: 'small_cap', 'large_cap', 'neutral'. Based on 30th and 70th percentiles of beta from the candidate universe.
            value_beta (str, optional): The value beta of the stock.
                Options: 'value', 'growth', 'neutral'. Based on 30th and 70th percentiles of beta from the candidate universe.
            profitability_beta (str, optional): The profitability beta of the stock.
                Options: 'robust', 'weak', 'neutral'. Based on 30th and 70th percentiles of beta from the candidate universe.
            investment_beta (str, optional): The investment beta of the stock.
                Options: 'conservative', 'aggressive', 'neutral'. Based on 30th and 70th percentiles of beta from the candidate universe.
            momentum_beta: (str, optional): The momentum beta of the stock used in Carhart 4-Factor model.
                Options: 'high', 'low', 'neutral'. Based on 30th and 70th percentiles of beta from the candidate universe.
            all_metrics (bool): If True, computes all metrics to the candidate universe. Default is False.
            save_dir (str, optional): Directory to save the contract datasets and raw data.
            verbose (bool): Whether to print verbose output. Default is False.
            dev_mode (bool): If True, enables development mode specific data directory management. Default is False.
        """

        # Date range for data retrieval
        self.start_date = start_date
        self.end_date = end_date

        # Stock collections (only one can be true at a time)
        self.sp_500 = sp_500
        self.nasdaq_100 = nasdaq_100
        self.dow_jones = dow_jones

        # Ensure only one index collection is selected
        assert (
            sum([sp_500, nasdaq_100, dow_jones]) <= 1
        ), "Please select only one index collection at time from sp_500, nasdaq_100, or dow_jones."

        # Candidate roots (used if no indices are selected)
        self.candidate_roots = candidate_roots

        # Factor filters
        self.volatility = volatility
        self.pe_ratio = pe_ratio
        self.debt_to_equity = debt_to_equity
        self.beta = beta
        self.market_cap = market_cap
        self.sector = sector
        self.industry = industry
        self.dividend_yield = dividend_yield
        self.earnings_volatility = earnings_volatility

        # Fama French Factors
        self.market_beta = market_beta
        self.size_beta = size_beta
        self.value_beta = value_beta
        self.profitability_beta = profitability_beta
        self.investment_beta = investment_beta
        self.momentum_beta = momentum_beta

        # Check if any of the ff factors are set
        if any([profitability_beta, investment_beta]):
            self.factor_mode = "ff5"
        elif any([momentum_beta]):
            self.factor_mode = "c4"
        elif any([market_beta, size_beta, value_beta]):
            self.factor_mode = "ff3"
        else:
            self.factor_mode = None

        # Directory and logging
        self.all_metrics = all_metrics
        self.save_dir = save_dir
        self.verbose = verbose
        self.dev_mode = dev_mode
        self.ctx = Console()

        # Set all metrics to not None
        if all_metrics:
            # Set all fundamental metrics to 'all' (or other appropriate default)
            self.volatility = "all" if self.volatility is None else self.volatility
            self.pe_ratio = "all" if self.pe_ratio is None else self.pe_ratio
            self.debt_to_equity = (
                "all" if self.debt_to_equity is None else self.debt_to_equity
            )
            self.beta = "all" if self.beta is None else self.beta
            self.market_cap = "all" if self.market_cap is None else self.market_cap
            self.dividend_yield = (
                "all" if self.dividend_yield is None else self.dividend_yield
            )
            self.earnings_volatility = (
                "all" if self.earnings_volatility is None else self.earnings_volatility
            )

            # Set all Fama French factors to 'all' (or other appropriate default)
            self.market_beta = "all" if self.market_beta is None else self.market_beta
            self.size_beta = "all" if self.size_beta is None else self.size_beta
            self.value_beta = "all" if self.value_beta is None else self.value_beta
            self.profitability_beta = (
                "all" if self.profitability_beta is None else self.profitability_beta
            )
            self.investment_beta = (
                "all" if self.investment_beta is None else self.investment_beta
            )
            self.momentum_beta = (
                "all" if self.momentum_beta is None else self.momentum_beta
            )

            # Factor mode should be set to the most comprehensive option
            self.factor_mode = "c4"  # Carhart 4-factor model includes all factor

    def set_roots(self) -> None:
        """
        Fetches constituents of a specified index using public data on Wikipedia and updates candidate_roots.
        """
        if self.sp_500:
            sp_data = pd.read_html(
                "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
            )[0]
            self.roots = sp_data["Symbol"].str.replace(".", "-").tolist()
        elif self.nasdaq_100:
            nasdaq_tables = pd.read_html("https://en.wikipedia.org/wiki/Nasdaq-100")
            for i, table in enumerate(nasdaq_tables):
                if "Symbol" in table.columns:
                    self.roots = table["Symbol"].tolist()
                    break
                elif "Ticker" in table.columns:
                    self.roots = table["Ticker"].tolist()
                    break

            # If we couldn't find the right table, raise an error
            if not self.roots:
                raise ValueError(
                    "Could not find NASDAQ-100 constituents table with expected columns"
                )
        elif self.dow_jones:
            # From the output, we can see Table 2 has the right structure
            dj_tables = pd.read_html(
                "https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average"
            )
            # Table 2 has 'Symbol' column
            self.roots = dj_tables[2]["Symbol"].tolist()
        else:
            self.roots = self.candidate_roots

        if self.verbose:
            self.ctx.log(f"Universe roots set to: {self.roots}")

    def get_market_metrics(self, remove_roots: bool = False) -> None:
        """
        Retrieves market metrics data for each stock in candidate_roots from various sources.
        Only includes metrics that are specified in the filter criteria.
        """
        self.market_metrics = dict()

        # Assert that roots is an attribute of self and is not empty
        assert (
            hasattr(self, "roots") and self.roots
        ), "No roots available. Run set_roots() first."

        for root in self.roots:
            market_metric_data = {}
            info = yf.Ticker(root).info

            # Only calculate volatility if the filter is set
            if self.volatility is not None:
                try:
                    volatility = load_stock_data_eod(
                        root=root,
                        start_date=self.start_date,
                        end_date=self.end_date,
                        clean_up=True,
                        offline=False,
                    )["close"].pct_change().std() * (252**0.5)
                    market_metric_data["volatility"] = volatility
                except:
                    volatility = None

            # Add other metrics only if their corresponding filter is set
            factor_map = {
                "pe_ratio": self.pe_ratio is not None and info.get("trailingPE"),
                "debt_to_equity": self.debt_to_equity is not None
                and info.get("debtToEquity"),
                "beta": self.beta is not None and info.get("beta"),
                "market_cap": self.market_cap is not None and info.get("marketCap"),
                "dividend_yield": self.dividend_yield is not None
                and info.get("dividendYield"),
            }

            # Add each metric to market_metric_data only if it should be included
            for key, value in factor_map.items():
                if (
                    value
                ):  # This will be false if either the filter is None or the value is None/falsy
                    market_metric_data[key] = value

            # If any values are None (i.e. the stock doesn't have that metric), don't add it to the dictionary
            if all(market_metric_data.values()):
                self.market_metrics[root] = market_metric_data
                if self.verbose:
                    self.ctx.log(f"Market metric data for {root}: {market_metric_data}")
            else:
                if self.verbose:
                    # Find out which metrics are missing
                    missing_metrics = [
                        key
                        for key, value in market_metric_data.items()
                        if value is None
                    ]
                    self.ctx.log(
                        f"Missing market metric data for {root}: {missing_metrics}"
                    )
                if remove_roots:
                    self.roots.remove(root)

    def get_factor_exposures(self, remove_roots: bool = False) -> None:
        """
        Computes and categorizes Fama-French factor exposures for each stock in the universe,
        using Kenneth French's data library and fitting the specified factor mode (ff3, c4, or ff5)
        with linear regression.
        """

        # Assert that roots is an attribute of self and is not empty
        assert (
            hasattr(self, "roots") and self.roots
        ), "No roots available. Run set_roots() first."

        assert (
            hasattr(self, "factor_mode") and self.factor_mode
        ), "No Factor mode specified. Set factor_mode to 'ff3', 'c4', or 'ff5'."

        self.factor_exposures = dict()

        # Fetch Fama-French factors for each root
        for root in self.roots:
            try:
                self.factor_exposures[root] = get_factor_exposures(
                    root=root,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    mode=self.factor_mode,
                )
            except:
                if self.verbose:
                    self.ctx.log(f"Could not fetch Fama-French factors for {root}.")
                if remove_roots:
                    self.roots.remove(root)

        factor_categories = factor_categorization(
            factors=self.factor_exposures, mode=self.factor_mode
        )

        # Update the market metrics with factor categories
        for root in self.roots:

            # Check if the root exists in market_metrics and factor_exposures
            if root in self.market_metrics and root in self.factor_exposures:
                self.market_metrics[root].update(factor_categories[root])
            else:
                continue

            if self.verbose:
                table = Table(title=f"Factor exposures for {root}")
                table.add_column("Factor")
                table.add_column("Value")
                for factor, value in self.factor_exposures[root].items():
                    table.add_row(factor, f"{float(value):.4f}")
                self.ctx.print(table)

    # Helper function to get percentiles for a given metric
    def get_percentiles(self, metric, bins=3):
        values = [
            self.market_metrics[root].get(metric)
            for root in self.roots
            if root in self.market_metrics and metric in self.market_metrics[root]
        ]
        values = [v for v in values if v is not None]

        if not values:
            return []

        if bins == 3:  # Terciles
            return [np.percentile(values, 33.33), np.percentile(values, 66.67)]
        elif bins == 5:  # Quintiles
            return [
                np.percentile(values, 20),
                np.percentile(values, 40),
                np.percentile(values, 60),
                np.percentile(values, 80),
            ]

    # Filter function for three-level factors
    def filter_three_level(
        self, filtered_roots: List[str], metric: str, level_value: Union[str, None]
    ) -> List[str]:
        if level_value is None:
            return filtered_roots

        percentiles = self.get_percentiles(metric, bins=3)
        if not percentiles:
            return filtered_roots

        result = []
        for root in filtered_roots:
            if (
                root not in self.market_metrics
                or metric not in self.market_metrics[root]
            ):
                continue

            value = self.market_metrics[root][metric]
            if level_value == "low" and value <= percentiles[0]:
                result.append(root)
            elif level_value == "medium" and percentiles[0] < value <= percentiles[1]:
                result.append(root)
            elif level_value == "high" and value > percentiles[1]:
                result.append(root)

        return result

    # Filter function for five-level factors
    def filter_five_level(
        self, filtered_roots: List[str], metric: str, level_value: Union[str, None]
    ) -> List[str]:
        if level_value is None:
            return filtered_roots

        percentiles = self.get_percentiles(metric, bins=5)
        if not percentiles:
            return filtered_roots

        result = []
        for root in filtered_roots:
            if (
                root not in self.market_metrics
                or metric not in self.market_metrics[root]
            ):
                continue

            value = self.market_metrics[root][metric]
            if level_value == "very_low" and value <= percentiles[0]:
                result.append(root)
            elif level_value == "low" and percentiles[0] < value <= percentiles[1]:
                result.append(root)
            elif level_value == "medium" and percentiles[1] < value <= percentiles[2]:
                result.append(root)
            elif level_value == "high" and percentiles[2] < value <= percentiles[3]:
                result.append(root)
            elif level_value == "very_high" and value > percentiles[3]:
                result.append(root)

        return result

    # Filter function for categorical factors
    def filter_categorical(
        self, filtered_roots: List[str], metric: str, category_value: Union[str, None]
    ) -> List[str]:
        if category_value is None:
            return filtered_roots

        result = []
        for root in filtered_roots:
            if (
                root in self.market_metrics
                and metric in self.market_metrics[root]
                and str(self.market_metrics[root][metric]).lower()
                == str(category_value).lower()
            ):
                result.append(root)

        return result

    def filter(self) -> None:
        """
        Filters the universe of stocks based on the specified criteria.
        - For ThreeFactorLevel: 'low' (0-33%), 'medium' (33-66%), 'high' (66-100%)
        - For FiveFactorLevel: 'very_low' (0-20%), 'low' (20-40%), 'medium' (40-60%), 'high' (60-80%), 'very_high' (80-100%)
        """

        if not self.market_metrics:
            self.ctx.log(
                "No market metric data available. Run get_market_metrics() first."
            )
            return

        if self.all_metrics:
            self.roots = self.candidate_roots
            return

        # Create a copy of roots to filter
        starting_roots = self.roots.copy()
        filtered_roots = []

        # Apply all filters
        filtered_roots.append(
            self.filter_three_level(starting_roots, "volatility", self.volatility)
        )

        filtered_roots.append(
            self.filter_three_level(starting_roots, "pe_ratio", self.pe_ratio)
        )
        filtered_roots.append(
            self.filter_three_level(
                starting_roots, "debt_to_equity", self.debt_to_equity
            )
        )
        filtered_roots.append(
            self.filter_three_level(starting_roots, "beta", self.beta)
        )
        filtered_roots.append(
            self.filter_three_level(starting_roots, "market_cap", self.market_cap)
        )
        filtered_roots.append(
            self.filter_three_level(
                starting_roots, "dividend_yield", self.dividend_yield
            )
        )

        # Categorical factors
        filtered_roots.append(
            self.filter_categorical(starting_roots, "sector", self.sector)
        )
        filtered_roots.append(
            self.filter_categorical(starting_roots, "industry", self.industry)
        )

        # Fama French factors
        if self.factor_mode is not None:
            if self.verbose:
                with self.ctx.status(
                    f"Computing factor exposures using {self.factor_mode} factors..."
                ):
                    self.get_factor_exposures()
            else:
                self.get_factor_exposures()

            # Define a function for direct FF categorical filtering
            def filter_direct(metric, desired_category):
                if desired_category is None:
                    return starting_roots

                result = []
                for root in starting_roots:
                    if self.market_metrics[root][metric] == desired_category:
                        result.append(root)

                return result

            # Apply factor filters directly using the specified categories
            filtered_roots.append(filter_direct("market_beta", self.market_beta))
            filtered_roots.append(filter_direct("size_beta", self.size_beta))
            filtered_roots.append(filter_direct("value_beta", self.value_beta))

            if self.factor_mode == "c4":
                filtered_roots.append(
                    filter_direct("momentum_beta", self.momentum_beta)
                )
            elif self.factor_mode == "ff5":
                filtered_roots.append(
                    filter_direct("profitability_beta", self.profitability_beta)
                )
                filtered_roots.append(
                    filter_direct("investment_beta", self.investment_beta)
                )

        # Take the intersection of all filtered roots
        self.roots = list(set.intersection(*map(set, filtered_roots)))
        assert (
            len(self.roots) != 0
        ), "No stocks left after filtering. Use less filtering or increase the size of your candidate universe."

        if self.verbose:
            self.ctx.log(f"Candidate universe: {starting_roots}")
            self.ctx.log(f"Filtered universe: {self.roots}")

    def download(
        self,
        contract_stride: int,
        interval_min: int,
        right: str,
        target_tte: int,
        tte_tolerance: Tuple[int, int],
        moneyness: str,
        train_split: float,
        val_split: float,
        strike_band: Optional[float] = 0.05,
        volatility_type: Optional[str] = "period",
        volatility_scaled: bool = False,
        volatility_scalar: Optional[float] = None,
    ) -> None:
        """
        Downloads options contract datasets and market data for the filtered universe of stocks. To be used in conjunction with
        offline=True when calling get_forecasting_loaders() for higher efficiency during model training.

        Args:
            contract_stride (int): Number of days between consecutive contracts.
            interval_min (int): Interval in minutes for the options data.
            right (str): Type of contract ('C' for call or 'P' and for put).
            target_tte (int): Target time to expiration in days.
            tte_tolerance (Tuple[int, int]): Lower and upper bounds for the time to expiration.
            moneyness (str): Moneyness of the option. Options: "ATM", "ITM", or "OTM".
            strike_band (float): Strike band for the option.
            train_split (float): Proportion of contracts to use for training.
            val_split (float): Proportion of contracts to use for validation.
            volatility_type (str, optional): Type of volatility to use for scaling. Options: "daily", "period", or "annualized".
            volatility_scaled (bool, optional): Whether to scale the volatility.
            volatility_scalar (float, optional): Scalar to multiply the volatility by.
            dev_mode (bool, optional): Whether to use development mode.

        Returns:
            None
        """

        self.contract_paths = dict()

        # Assert that roots is an attribute of self and is not empty
        assert (
            hasattr(self, "roots") and self.roots
        ), "No roots available. Run set_roots() first."

        # Download all data for the filtered universe
        for root in self.roots:

            # Download contracts
            if self.verbose:
                self.ctx.log(f"Downloading data for root: {root}")
            train_contract_dataset, val_contract_dataset, test_contract_dataset = (
                get_contract_datasets(
                    root=root,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    contract_stride=contract_stride,
                    interval_min=interval_min,
                    right=right,
                    target_tte=target_tte,
                    tte_tolerance=tte_tolerance,
                    moneyness=moneyness,
                    strike_band=strike_band,
                    volatility_type=volatility_type,
                    volatility_scaled=volatility_scaled,
                    volatility_scalar=volatility_scalar,
                    train_split=train_split,
                    val_split=val_split,
                    clean_up=False,  # Set to False to download data
                    offline=False,  # Set to False to download data
                    save_dir=self.save_dir,
                    verbose=self.verbose,
                    dev_mode=self.dev_mode,
                )
            )

            # Download raw data
            updated_train_contract_dataset = get_forecasting_dataset(
                contract_dataset=train_contract_dataset,
                tte_tolerance=tte_tolerance,
                download_only=True,
                modify_contracts=True,
                verbose=self.verbose,
                save_dir=self.save_dir,
                dev_mode=self.dev_mode,
            )
            updated_val_contract_dataset = get_forecasting_dataset(
                contract_dataset=val_contract_dataset,
                tte_tolerance=tte_tolerance,
                download_only=True,
                modify_contracts=True,
                verbose=self.verbose,
                save_dir=self.save_dir,
                dev_mode=self.dev_mode,
            )
            updated_test_contract_dataset = get_forecasting_dataset(
                contract_dataset=test_contract_dataset,
                tte_tolerance=tte_tolerance,
                download_only=True,
                modify_contracts=True,
                verbose=self.verbose,
                save_dir=self.save_dir,
                dev_mode=self.dev_mode,
            )

            self.contract_paths[root] = {
                "train": updated_train_contract_dataset.filepath,
                "val": updated_val_contract_dataset.filepath,
                "test": updated_test_contract_dataset.filepath,
            }

    def get_forecasting_loaders(
        self,
        root: str,
        tte_tolerance: Tuple[int, int],
        seq_len: int,
        pred_len: int,
        scaling: bool = False,
        dtype: str = "float32",
        core_feats: List[str] = ["option_returns"],
        tte_feats: Optional[List[str]] = None,
        datetime_feats: Optional[List[str]] = None,
        keep_datetime: bool = False,
        target_channels: Optional[List[str]] = None,
        target_type: str = "multistep",
        offline: bool = False,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = torch.cuda.is_available(),
        persistent_workers: bool = True,
    ) -> Union[
        Tuple[DataLoader, DataLoader, DataLoader],
        Tuple[DataLoader, DataLoader, DataLoader, StandardScaler],
    ]:
        """

        Args:
            root (str): Root symbol of the stock.
            contract_stride (int): Number of days between consecutive contracts.
            interval_min (int): Interval in minutes for the options data.
            right (str): Type of contract ('C' for call or 'P' and for put).
            target_tte (int): Target time to expiration in days.
            tte_tolerance (Tuple[int, int]): Lower and upper bounds for the time to expiration.
            moneyness (str): Moneyness of the option. Options: "ATM", "ITM", or "OTM".
            seq_len (int): Sequence length for the input data.
            pred_len (int): Prediction length for the target data.
            dtype_str (str): Data type for the input and target data.
            train_split (float): Proportion of contracts to use for training.
            val_split (float): Proportion of contracts to use for validation.
            scaling (bool): Whether to scale the data.
            dtype (str): Data type for the input and target data.
            core_feats (List[str]): Core features to include in the input data.
            tte_feats (List[str], optional): Time-to-expiration features to include in the input data.
            datetime_feats (List[str], optional): Datetime features to include in the input data.
            keep_datetime (bool, optional): Whether to keep the datetime features in the input data.
            target_channels (List[str], optional): Target channels to include in the target data.
            target_type (str, optional): Type of forecasting target. Options: "multistep" (float), "average" (float), or "average_direction" (binary).
            strike_band (float, optional): Strike band for the option.
            volatility_type (str, optional): Type of volatility to use for scaling. Options: "daily", "period", or "annualized".
            volatility_scaled (bool, optional): Whether to scale the volatility.
            volatility_scalar (float, optional): Scalar to multiply the volatility by.
            offline (bool, optional): Whether to use offline data for faster training.
            batch_size (int, optional): Batch size for the data loader.
            shuffle (bool, optional): Whether to shuffle the data.
            drop_last (bool, optional): Whether to drop the last incomplete batch.
            num_workers (int, optional): Number of workers for the data loader.
            prefetch_factor (int, optional): Prefetch factor for the data loader.
            pin_memory (bool, optional): Whether to pin memory for the data loader.
            persistent_workers (bool, optional): Whether to use persistent workers for the data loader.
            dev_mode (bool, optional): Whether to use development mode.

        Returns:
            Tuple[DataLoader, DataLoader, DataLoader]: Train, validation, and test data loaders if scaling=False.
            Tuple[DataLoader, DataLoader, DataLoader, StandardScaler]: Train, validation, and test data loaders, and the scaler if scaling=True.
        """

        loaders = get_forecasting_loaders(
            train_contract_dataset=ContractDataset.load(
                self.contract_paths[root]["train"]
            ),
            val_contract_dataset=ContractDataset.load(self.contract_paths[root]["val"]),
            test_contract_dataset=ContractDataset.load(
                self.contract_paths[root]["test"]
            ),
            tte_tolerance=tte_tolerance,
            core_feats=core_feats,
            tte_feats=tte_feats,
            datetime_feats=datetime_feats,
            keep_datetime=keep_datetime,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers,
            clean_up=False,
            offline=offline,
            save_dir=self.save_dir,
            verbose=self.verbose,
            scaling=scaling,
            intraday=False,  # Not implemented yet
            target_channels=target_channels,
            target_type=target_type,
            seq_len=seq_len,
            pred_len=pred_len,
            dtype=dtype,
            dev_mode=self.dev_mode,
        )

        return loaders


if __name__ == "__main__":
    ctx = Console()

    # Create a Universe instance
    universe = Universe(
        dow_jones=True,
        debt_to_equity="low",
        momentum_beta="low",
        market_cap="high",
        # all_metrics=True,
        start_date="20210101",
        end_date="20211231",
        dev_mode=True,
        verbose=True,
    )
    universe.set_roots()  # Fetch index constituents
    universe.get_market_metrics()  # Fetch market metric data
    universe.get_factor_exposures()  # Fetch Factor Model factors
    universe.filter()  # Filter the universe based on the specified criteria
    universe.download(
        contract_stride=2,
        interval_min=15,
        right="C",
        target_tte=30,
        tte_tolerance=(20, 40),
        moneyness="ATM",
        train_split=0.5,
        val_split=0.3,
    )  # Download contracts and raw data

    # Display market metrics dictionary with rich
    table = Table(title="Market Metrics")
    table.add_column("Root")
    table.add_column("Metrics")
    for root, metrics in universe.market_metrics.items():
        metrics_str = ", ".join([f"{k}: {v}" for k, v in metrics.items()])
        table.add_row(root, metrics_str)
    ctx.print(table)

    # Save universe market_metrics to JSON file
    import json

    with open("universe_market_metrics.json", "w") as f:
        json.dump(universe.market_metrics, f, indent=4)

    # Uncomment the following lines to filter the universe based on specific criteria

    # Filter the universe for stocks with low debt-to-equity and high market cap
    # universe.filter()

    # # Set parameters
    # contract_stride = 2
    # interval_min = 1
    # right = "C"
    # target_tte = 30
    # tte_tolerance = (20, 40)
    # moneyness = "ATM"
    # train_split = 0.5
    # val_split = 0.3
    # volatility_scaled = False

    # # Download contracts and raw data for the filtered universe
    # universe.roots = ["AMZN"]
    # root = universe.roots[0]
    # universe.download(
    #     contract_stride=3,
    #     interval_min=1,
    #     right="C",
    #     target_tte=30,
    #     tte_tolerance=(15, 45),
    #     moneyness="ATM",
    #     train_split=0.5,
    #     val_split=0.3,
    #     dev_mode=True,
    # )

    # # Select a root for forecasting
    # loaders = universe.get_forecasting_loaders(
    #     offline=True,
    #     root=root,
    #     tte_tolerance=tte_tolerance,
    #     seq_len=30,  # 30-minute lookback window
    #     pred_len=5,  # 5-minute forecast horizon
    #     core_feats=["option_returns"],
    #     target_channels=["option_returns"],
    #     target_type="multistep",
    #     keep_datetime=True,
    #     dtype="float32",
    #     scaling=False,
    #     dev_mode=True,
    # )

    # print(f"Train loader: {len(loaders[0].dataset)} samples")
    # print(f"Validation loader: {len(loaders[1].dataset)} samples")
    # print(f"Test loader: {len(loaders[2].dataset)} samples")
