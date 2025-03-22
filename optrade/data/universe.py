from typing import List, Optional, Dict, Any, Tuple, Union
from pydantic import BaseModel, Field, model_validator
import pandas as pd
import numpy as np
import yfinance as yf
import torch
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler

from optrade.data.thetadata import load_stock_data_eod
from optrade.utils.market_metrics import (
    get_fama_french_factors,
    factor_categorization,
)
from optrade.utils.stock_categories import (
    ThreeFactorLevel,
    FiveFactorLevel,
    SectorType,
    IndustryType,
)
from optrade.data.forecasting import (
    get_contract_datasets,
    get_forecasting_dataset,
    get_forecasting_loaders,
)
from optrade.data.contracts import ContractDataset


class Universe(BaseModel):
    model_config = {"use_enum_values": True, "arbitrary_types_allowed": True}

    # Date range for data retrieval
    start_date: str = Field(
        default=None, description="Start date for data retrieval in YYYYMMDD format"
    )
    end_date: str = Field(
        default=None, description="End date for data retrieval in YYYYMMDD format"
    )

    # Stock collections (only one can be true at a time)
    sp_500: bool = Field(
        default=False,
        description="If True, use S&P 500 stocks as the candidate universe",
    )
    nasdaq_100: bool = Field(
        default=False,
        description="If True, use NASDAQ 100 stocks as the candidate universe",
    )
    dow_jones: bool = Field(
        default=False,
        description="If True, use Dow Jones Industrial Average stocks as the candidate universe",
    )
    russell_2000: bool = Field(
        default=False,
        description="If True, use Russell 2000 stocks as the candidate universe",
    )

    # Candidate roots (used if no collection is selected)
    candidate_roots: List[str] = Field(
        default=None,
        description="Candidate root symbols, to be filtered by other parameters. Used only if no collection (sp_500, nasdaq_100, etc.) is selected",
    )

    # Factor filters
    volatility: Optional[FiveFactorLevel] = Field(
        default=None,
        description="The volatility of the stock. Options: 'very_low' (bottom 20%), 'low' (20-40%), 'medium' (40-60%), 'high' (60-80%), 'very_high' (top 20%)",
    )
    pe_ratio: Optional[ThreeFactorLevel] = Field(
        default=None,
        description="The P/E ratio of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)",
    )
    debt_to_equity: Optional[ThreeFactorLevel] = Field(
        default=None,
        description="The debt to equity ratio of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)",
    )
    beta: Optional[ThreeFactorLevel] = Field(
        default=None,
        description="The beta of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)",
    )
    market_cap: Optional[ThreeFactorLevel] = Field(
        default=None,
        description="The market cap of the stock. Options: 'low' (small cap), 'medium' (mid cap), 'high' (large cap)",
    )
    sector: Optional[SectorType] = Field(
        default=None,
        description="The sector of the stock. Options: 'tech', 'healthcare', 'financial', 'consumer_cyclical', 'consumer_defensive', 'industrial', 'energy', 'materials', 'utilities', 'real_estate', 'communication'",
    )
    industry: Optional[IndustryType] = Field(
        default=None,
        description="The industry of the stock. See IndustryType enum for complete list of options matching Yahoo Finance classifications",
    )
    dividend_yield: Optional[ThreeFactorLevel] = Field(
        default=None,
        description="The dividend yield of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)",
    )
    earnings_volatility: Optional[ThreeFactorLevel] = Field(
        default=None,
        description="The earnings volatility of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)",
    )

    # Fama French Factors
    market_beta: Optional[str] = Field(
        default=None,
        description="The market beta of the stock. Options: 'high', 'low', 'neutral'",
    )
    size_beta: Optional[str] = Field(
        default=None,
        description="The size beta of the stock. Options: 'small_cap', 'large_cap', 'neutral'",
    )
    value_beta: Optional[str] = Field(
        default=None,
        description="The value beta of the stock. Options: 'value', 'growth', 'neutral'",
    )
    profitability_beta: Optional[str] = Field(
        default=None,
        description="The profitability beta of the stock. Options: 'robust', 'weak', 'neutral'",
    )
    investment_beta: Optional[str] = Field(
        default=None,
        description="The investment beta of the stock. Options: 'conservative', 'aggressive', 'neutral'",
    )
    ff_mode: str = Field(
        default=None,
        description="Mode for the Fama-French model ('3_factor' or '5_factor'). If None, no FF factors will be used for filtering.",
    )

    # Dataset factors
    core_feats: List[str] = Field(
        default=["option_returns"],
        description="Core features to include in the dataset",
    )
    tte_feats: List[str] = Field(
        default=["tte"],
        description="Time to expiration features to include in the dataset",
    )
    datetime_feats: List[str] = Field(
        default=None, description="Datetime features to include in the dataset"
    )
    contract_stride: int = Field(
        default=1, description="Number of contracts to skip between each contract"
    )
    interval_min: int = Field(default=1, description="Interval in minutes for the data")
    right: str = Field(
        default="C", description="Option type: 'C' for call, 'P' for put"
    )
    target_tte: int = Field(default=30, description="Target time to expiration in days")
    tte_tolerance: Tuple[int, int] = Field(
        default=(25, 35), description="Tolerance range for time to expiration"
    )
    moneyness: str = Field(
        default="ATM", description="Moneyness of the option: 'ATM', 'OTM', 'ITM'"
    )
    target_band: float = Field(
        default=0.05, description="Target band for strike selection"
    )
    volatility_type: str = Field(
        default="period",
        description="Type of volatility to use: 'period', 'annualized'",
    )
    volatility_scaled: bool = Field(
        default=False, description="Whether to used volatility-based strike selection"
    )
    volatility_scalar: float = Field(
        default=1.0,
        description="Scalar used for volatility-based strike selection, i.e. number of SDs of the current price",
    )
    train_split: float = Field(default=0.7, description="Train split ratio")
    val_split: float = Field(default=0.15, description="Validation split ratio")
    save_dir: Optional[str] = Field(
        default=None, description="Directory to save the contract datasets"
    )
    verbose: bool = Field(default=False, description="Whether to print verbose output")

    # Attributes used to store information (do not initialize)
    fundamentals: Dict[str, Dict[str, Any]] = Field(
        default_factory=dict, description="Fundamental data for each stock"
    )
    roots: List[str] = Field(
        default_factory=list,
        description="Root symbols of the stocks in the candidate universe",
    )
    contracts: Dict[str, Dict[str, ContractDataset]] = Field(
        default_factory=dict, description="Contract datasets for each stock"
    )

    @model_validator(mode="after")
    def validate_universe_selection(self) -> "Universe":
        """
        Validates that only one index collection is selected at a time and
        ensures candidate_roots is provided if no index is selected.
        """
        # Identify which index collections are selected
        collections = ["sp_500", "nasdaq_100", "dow_jones"]
        selected = [coll for coll in collections if getattr(self, coll, False)]

        # Check if more than one collection is selected
        if len(selected) > 1:
            raise ValueError(
                f"Only one collection can be selected at a time. You selected: {', '.join(selected)}"
            )

        # If no collection is selected, ensure candidate_roots is provided
        if len(selected) == 0 and not self.candidate_roots:
            raise ValueError(
                "Either select a collection (sp_500, nasdaq_100, etc.) or provide candidate_roots"
            )

        return self

    def set_candidate_roots(self) -> None:
        """
        Fetches constituents of a specified index using public data sources and updates candidate_roots.
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

    def get_fundamentals(self) -> None:
        """
        Retrieves fundamental data for each stock in candidate_roots using yfinance.
        Only includes metrics that are specified in the filter criteria.
        """
        self.fundamentals = dict()

        for root in self.roots:
            fundamental_data = {}
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
                    fundamental_data["volatility"] = volatility
                except:
                    volatility = None

            # Add other metrics only if their corresponding filter is set
            factor_map = {
                "pe_ratio": self.pe_ratio is not None and info.get("trailingPE"),
                "debt_to_equity": self.debt_to_equity is not None
                and info.get("debtToEquity"),
                "beta": self.beta is not None and info.get("beta"),
                "market_cap": self.market_cap is not None and info.get("marketCap"),
                "sector": self.sector is not None and info.get("sector").lower(),
                "industry": self.industry is not None and info.get("industry").lower(),
                "dividend_yield": self.dividend_yield is not None
                and info.get("dividendYield") * 100,
            }

            # Add each metric to fundamental_data only if it should be included
            for key, value in factor_map.items():
                if (
                    value
                ):  # This will be false if either the filter is None or the value is None/falsy
                    fundamental_data[key] = value

            # If any values are None (i.e. the stock doesn't have that metric), don't add it to the dictionary
            if all(fundamental_data.values()):
                self.fundamentals[root] = fundamental_data
            else:
                self.roots.remove(root)

    def get_fama_french_factors(self) -> None:
        for root in self.roots:
            try:
                ff_factors = get_fama_french_factors(
                    root=root,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    mode=self.ff_mode,
                )
                ff_factor_categories = factor_categorization(
                    ff_factors, mode=self.ff_mode
                )
                self.fundamentals[root].update(ff_factor_categories)
            except:
                self.roots.remove(root)

    def filter_universe(self) -> None:
        """
        Filters the universe of stocks based on the specified criteria.
        - For ThreeFactorLevel: 'low' (0-33%), 'medium' (33-66%), 'high' (66-100%)
        - For FiveFactorLevel: 'very_low' (0-20%), 'low' (20-40%), 'medium' (40-60%), 'high' (60-80%), 'very_high' (80-100%)
        """

        # Store fundamentals in a df
        fundamentals_df = pd.DataFrame(self.fundamentals).T

        if not self.fundamentals:
            print("No fundamental data available. Run get_fundamentals() first.")
            return

        # Create a copy of roots to filter
        filtered_roots = self.roots.copy()

        # Helper function to get percentiles for a given metric
        def get_percentiles(metric, bins=3):
            values = [
                self.fundamentals[root].get(metric)
                for root in self.roots
                if root in self.fundamentals and metric in self.fundamentals[root]
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
        def filter_three_level(metric, level_value):
            if level_value is None:
                return filtered_roots

            percentiles = get_percentiles(metric, bins=3)
            if not percentiles:
                return filtered_roots

            result = []
            for root in filtered_roots:
                if (
                    root not in self.fundamentals
                    or metric not in self.fundamentals[root]
                ):
                    continue

                value = self.fundamentals[root][metric]
                if level_value == "low" and value <= percentiles[0]:
                    result.append(root)
                elif (
                    level_value == "medium" and percentiles[0] < value <= percentiles[1]
                ):
                    result.append(root)
                elif level_value == "high" and value > percentiles[1]:
                    result.append(root)

            return result

        # Filter function for five-level factors
        def filter_five_level(metric, level_value):
            if level_value is None:
                return filtered_roots

            percentiles = get_percentiles(metric, bins=5)
            if not percentiles:
                return filtered_roots

            result = []
            for root in filtered_roots:
                if (
                    root not in self.fundamentals
                    or metric not in self.fundamentals[root]
                ):
                    continue

                value = self.fundamentals[root][metric]
                if level_value == "very_low" and value <= percentiles[0]:
                    result.append(root)
                elif level_value == "low" and percentiles[0] < value <= percentiles[1]:
                    result.append(root)
                elif (
                    level_value == "medium" and percentiles[1] < value <= percentiles[2]
                ):
                    result.append(root)
                elif level_value == "high" and percentiles[2] < value <= percentiles[3]:
                    result.append(root)
                elif level_value == "very_high" and value > percentiles[3]:
                    result.append(root)

            return result

        # Filter function for categorical factors
        def filter_categorical(metric, category_value):
            if category_value is None:
                return filtered_roots

            result = []
            for root in filtered_roots:
                if (
                    root in self.fundamentals
                    and metric in self.fundamentals[root]
                    and str(self.fundamentals[root][metric]).lower()
                    == str(category_value).lower()
                ):
                    result.append(root)

            return result

        # Apply all filters
        # Five-level factors
        filtered_roots = filter_five_level("volatility", self.volatility)

        # Three-level factors
        filtered_roots = filter_three_level("pe_ratio", self.pe_ratio)
        filtered_roots = filter_three_level("debt_to_equity", self.debt_to_equity)
        filtered_roots = filter_three_level("beta", self.beta)
        filtered_roots = filter_three_level("market_cap", self.market_cap)
        filtered_roots = filter_three_level("dividend_yield", self.dividend_yield)

        # Categorical factors
        filtered_roots = filter_categorical("sector", self.sector)
        filtered_roots = filter_categorical("industry", self.industry)

        # Fama French factors
        if self.ff_mode is not None:
            self.get_fama_french_factors()

            # Define a function for direct FF categorical filtering
            def filter_ff_direct(metric, desired_category):
                if desired_category is None:
                    return filtered_roots

                result = []
                for root in filtered_roots:
                    if self.fundamentals[root][metric] == desired_category:
                        result.append(root)

                return result

            # Apply FF factor filters directly using the specified categories
            filtered_roots = filter_ff_direct("market_beta", self.market_beta)
            filtered_roots = filter_ff_direct("size_beta", self.size_beta)
            filtered_roots = filter_ff_direct("value_beta", self.value_beta)

            if self.ff_mode == "5_factor":
                filtered_roots = filter_ff_direct(
                    "profitability_beta", self.profitability_beta
                )
                filtered_roots = filter_ff_direct(
                    "investment_beta", self.investment_beta
                )

        # Update roots to the filtered list
        self.roots = filtered_roots

    def download(self) -> None:
        # Download all data for the filtered universe
        for root in self.roots:

            # Download contracts
            train_contracts, val_contracts, test_contracts = get_contract_datasets(
                root=root,
                start_date=self.start_date,
                end_date=self.end_date,
                contract_stride=self.contract_stride,
                interval_min=self.interval_min,
                right=self.right,
                target_tte=self.target_tte,
                tte_tolerance=self.tte_tolerance,
                moneyness=self.moneyness,
                target_band=self.target_band,
                volatility_type=self.volatility_type,
                volatility_scaled=self.volatility_scaled,
                volatility_scalar=self.volatility_scalar,
                train_split=self.train_split,
                val_split=self.val_split,
                clean_up=False,  # Set to False to download data
                offline=False,  # Set to False to download data
                save_dir=self.save_dir,
                verbose=self.verbose,
            )

            self.contracts[root] = {
                "train": train_contracts,
                "val": val_contracts,
                "test": test_contracts,
            }

            # Download raw data
            get_forecasting_dataset(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                contracts=train_contracts,
                tte_tolerance=self.tte_tolerance,
                download_only=True,
            )
            get_forecasting_dataset(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                contracts=val_contracts,
                tte_tolerance=self.tte_tolerance,
                download_only=True,
            )
            get_forecasting_dataset(
                seq_len=self.seq_len,
                pred_len=self.pred_len,
                contracts=test_contracts,
                tte_tolerance=self.tte_tolerance,
                download_only=True,
            )

    def get_forecasting_loaders(
        self,
        root: str,
        offline: bool = False,
        batch_size: int = 32,
        shuffle: bool = True,
        drop_last: bool = False,
        num_workers: int = 4,
        prefetch_factor: int = 2,
        pin_memory: bool = torch.cuda.is_available(),
    ) -> Union[
        Tuple[DataLoader, DataLoader, DataLoader],
        Tuple[DataLoader, DataLoader, DataLoader, StandardScaler],
    ]:

        loaders = get_forecasting_loaders(
            root=root,
            start_date=self.start_date,
            end_date=self.end_date,
            contract_stride=self.contract_stride,
            interval_min=self.interval_min,
            right=self.right,
            target_tte=self.target_tte,
            tte_tolerance=self.tte_tolerance,
            moneyness=self.moneyness,
            target_band=self.target_band,
            volatility_type=self.volatility_type,
            volatility_scaled=self.volatility_scaled,
            volatility_scalar=self.volatility_scalar,
            train_split=self.train_split,
            val_split=self.val_split,
            core_feats=self.core_feats,
            tte_feats=self.tte_feats,
            datetime_feats=self.datetime_feats,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=drop_last,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=pin_memory,
            clean_up=False,
            offline=offline,
            save_dir=self.save_dir,
            verbose=self.verbose,
            scaling=self.scaling,
            intraday=self.intraday,
            target_channels=self.target_channels,
            seq_len=self.seq_len,
            pred_len=self.pred_len,
            dtype=self.dtype,
        )

        return loaders


if __name__ == "__main__":
    # Create a Universe instance
    save_dir = "temp"
    universe = Universe(
        # sp_500=True,
        dow_jones=True,
        # volatility="medium",
        # pe_ratio="low",
        # debt_to_equity="low",
        # beta="medium",
        # market_cap="low",
        # sector="technology",
        # industry="technology",
        # dividend_yield="low",
        # earnings_volatility="low",
        start_date="20210101",
        end_date="20211001",
        # market_beta="neutral",
        ff_mode="5_factor",
        size_beta="large_cap",
        # volatility="high"
        save_dir=save_dir,
    )

    # Fetch the S&P 500 constituents
    universe.set_candidate_roots()
    universe.get_fundamentals()

    # print(universe.fundamentals["NVDA"],
    # universe.fundamentals["AAPL"],
    # universe.fundamentals["MMM"],
    # universe.fundamentals["CAT"],
    # universe.fundamentals["DIS"]
    # )
    print(f"Universe: {universe.roots}")
    # Filter the universe
    universe.filter_universe()
    print(f"Filtered universe: {universe.roots}")

    universe.roots = ["AAPL"]

    universe.download()
    loaders = universe.get_forecasting_loaders(root="AAPL", offline=True)
