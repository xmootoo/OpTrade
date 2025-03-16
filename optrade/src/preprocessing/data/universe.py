from typing import List, Optional, Dict, Any, Union, Literal
from pydantic import BaseModel, Field, model_validator
import pandas as pd
import numpy as np
from enum import Enum
import yfinance as yf

# Define enums for categorical factors
class ThreeFactorLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"

class FiveFactorLevel(str, Enum):
    VERY_LOW = "very_low"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"

class SectorType(str, Enum):
    TECH = "tech"
    HEALTHCARE = "healthcare"
    FINANCIAL = "financial"
    CONSUMER_CYCLICAL = "consumer_cyclical"
    CONSUMER_DEFENSIVE = "consumer_defensive"
    INDUSTRIAL = "industrial"
    ENERGY = "energy"
    MATERIALS = "materials"
    UTILITIES = "utilities"
    REAL_ESTATE = "real_estate"
    COMMUNICATION = "communication"

class IndustryType(str, Enum):
    # Technology Industries
    SEMICONDUCTOR = "semiconductor"
    SOFTWARE_APPLICATION = "software_application"
    SOFTWARE_INFRASTRUCTURE = "software_infrastructure"
    ELECTRONIC_COMPONENTS = "electronic_components"
    COMPUTER_HARDWARE = "computer_hardware"
    CONSUMER_ELECTRONICS = "consumer_electronics"

    # Healthcare Industries
    BIOTECHNOLOGY = "biotechnology"
    PHARMACEUTICAL_MANUFACTURERS = "pharmaceutical_manufacturers"
    MEDICAL_DEVICES = "medical_devices"
    MEDICAL_CARE_FACILITIES = "medical_care_facilities"
    HEALTH_INFORMATION_SERVICES = "health_information_services"

    # Financial Industries
    BANKS_REGIONAL = "banks_regional"
    BANKS_DIVERSIFIED = "banks_diversified"
    INSURANCE_PROPERTY_CASUALTY = "insurance_property_casualty"
    INSURANCE_LIFE = "insurance_life"
    ASSET_MANAGEMENT = "asset_management"
    CAPITAL_MARKETS = "capital_markets"

    # Consumer Cyclical Industries
    AUTO_MANUFACTURERS = "auto_manufacturers"
    AUTO_PARTS = "auto_parts"
    APPAREL_RETAIL = "apparel_retail"
    DEPARTMENT_STORES = "department_stores"
    RESTAURANTS = "restaurants"
    LODGING = "lodging"
    TRAVEL_SERVICES = "travel_services"

    # Consumer Defensive Industries
    GROCERY_STORES = "grocery_stores"
    DISCOUNT_STORES = "discount_stores"
    HOUSEHOLD_PRODUCTS = "household_products"
    BEVERAGES_NON_ALCOHOLIC = "beverages_non_alcoholic"
    BEVERAGES_ALCOHOLIC = "beverages_alcoholic"
    TOBACCO = "tobacco"

    # Industrial Industries
    AEROSPACE_DEFENSE = "aerospace_defense"
    FARM_HEAVY_MACHINERY = "farm_heavy_machinery"
    ENGINEERING_CONSTRUCTION = "engineering_construction"
    TOOLS_ACCESSORIES = "tools_accessories"
    BUSINESS_EQUIPMENT = "business_equipment"

    # Energy Industries
    OIL_GAS_INTEGRATED = "oil_gas_integrated"
    OIL_GAS_EXPLORATION_PRODUCTION = "oil_gas_exploration_production"
    OIL_GAS_REFINING_MARKETING = "oil_gas_refining_marketing"
    OIL_GAS_EQUIPMENT_SERVICES = "oil_gas_equipment_services"

    # Materials Industries
    CHEMICALS = "chemicals"
    SPECIALTY_CHEMICALS = "specialty_chemicals"
    STEEL = "steel"
    BUILDING_MATERIALS = "building_materials"
    PAPER_PAPER_PRODUCTS = "paper_paper_products"

    # Utilities Industries
    UTILITIES_REGULATED_ELECTRIC = "utilities_regulated_electric"
    UTILITIES_REGULATED_GAS = "utilities_regulated_gas"
    UTILITIES_RENEWABLE = "utilities_renewable"
    UTILITIES_DIVERSIFIED = "utilities_diversified"

    # Real Estate Industries
    REIT_RESIDENTIAL = "reit_residential"
    REIT_RETAIL = "reit_retail"
    REIT_OFFICE = "reit_office"
    REIT_HEALTHCARE = "reit_healthcare"
    REIT_INDUSTRIAL = "reit_industrial"
    REAL_ESTATE_SERVICES = "real_estate_services"

    # Communication Industries
    TELECOM_SERVICES = "telecom_services"
    ENTERTAINMENT = "entertainment"
    ADVERTISING_AGENCIES = "advertising_agencies"
    ELECTRONIC_GAMING_MULTIMEDIA = "electronic_gaming_multimedia"
    INTERNET_CONTENT_INFORMATION = "internet_content_information"

class ContractUniverse(BaseModel):
    model_config = {"use_enum_values": True}

    # Stock collections (only one can be true at a time)
    sp_500: bool = Field(default=False, description="If True, use S&P 500 stocks as the candidate universe")
    nasdaq_100: bool = Field(default=False, description="If True, use NASDAQ 100 stocks as the candidate universe")
    dow_jones: bool = Field(default=False, description="If True, use Dow Jones Industrial Average stocks as the candidate universe")
    russell_2000: bool = Field(default=False, description="If True, use Russell 2000 stocks as the candidate universe")

    # Candidate roots (used if no collection is selected)
    candidate_roots: List[str] = Field(default=[""], description="Candidate root symbols, to be filtered by other parameters. Used only if no collection (sp_500, nasdaq_100, etc.) is selected")

    # Factor filters
    volatility: Optional[FiveFactorLevel] = Field(default=None, description="The volatility of the stock. Options: 'very_low' (bottom 20%), 'low' (20-40%), 'medium' (40-60%), 'high' (60-80%), 'very_high' (top 20%)")
    pe_ratio: Optional[ThreeFactorLevel] = Field(default=None, description="The P/E ratio of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)")
    debt_to_equity: Optional[ThreeFactorLevel] = Field(default=None, description="The debt to equity ratio of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)")
    beta: Optional[ThreeFactorLevel] = Field(default=None, description="The beta of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)")
    market_cap: Optional[ThreeFactorLevel] = Field(default=None, description="The market cap of the stock. Options: 'low' (small cap), 'medium' (mid cap), 'high' (large cap)")
    sector: Optional[SectorType] = Field(default=None, description="The sector of the stock. Options: 'tech', 'healthcare', 'financial', 'consumer_cyclical', 'consumer_defensive', 'industrial', 'energy', 'materials', 'utilities', 'real_estate', 'communication'")
    industry: Optional[IndustryType] = Field(default=None, description="The industry of the stock. See IndustryType enum for complete list of options matching Yahoo Finance classifications")
    dividend_yield: Optional[ThreeFactorLevel] = Field(default=None, description="The dividend yield of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)")
    earnings_volatility: Optional[ThreeFactorLevel] = Field(default=None, description="The earnings volatility of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)")

    # Fama French Factors
    market_beta: Optional[ThreeFactorLevel] = Field(default=None, description="The market beta of the stock. Options: 'low' (bottom 33%), 'medium' (middle 33%), 'high' (top 33%)")
    size_beta: Optional[ThreeFactorLevel] = Field(default=None, description="The size beta of the stock. Options: 'low' (small cap exposure), 'medium' (neutral), 'high' (large cap exposure)")
    value_beta: Optional[ThreeFactorLevel] = Field(default=None, description="The value beta of the stock. Options: 'low' (growth exposure), 'medium' (neutral), 'high' (value exposure)")
    profitability_beta: Optional[ThreeFactorLevel] = Field(default=None, description="The profitability beta of the stock. Options: 'low' (weak profitability), 'medium' (neutral), 'high' (robust profitability)")
    investment_beta: Optional[ThreeFactorLevel] = Field(default=None, description="The investment beta of the stock. Options: 'low' (aggressive investment), 'medium' (neutral), 'high' (conservative investment)")

    @model_validator(mode='after')
    def validate_universe_selection(self) -> 'ContractUniverse':
        """
        Validates that only one index collection is selected at a time and
        ensures candidate_roots is provided if no index is selected.
        """
        # Identify which index collections are selected
        collections = ['sp_500', 'nasdaq_100', 'dow_jones']
        selected = [coll for coll in collections if getattr(self, coll, False)]

        # Check if more than one collection is selected
        if len(selected) > 1:
            raise ValueError(f"Only one collection can be selected at a time. You selected: {', '.join(selected)}")

        # If no collection is selected, ensure candidate_roots is provided
        if len(selected) == 0 and not self.candidate_roots:
            raise ValueError("Either select a collection (sp_500, nasdaq_100, etc.) or provide candidate_roots")

        return self

    # def set_candidate_roots(self) -> None:
    #     """
    #     Fetches constituents of a specified index using public data sources.
    #     Returns:
    #         List[str]: List of ticker symbols for the specified index
    #     """
    #     if self.sp_500:
    #         sp_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    #         self.candidate_roots = sp_data['Symbol'].str.replace('.', '-').tolist()
    #     elif self.nasdaq_100:
    #         ndx_data = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')[1]
    #         self.candidate_roots = ndx_data['Ticker'].tolist()
    #     elif self.dow_jones:
    #         dj_data = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')[1]
    #         self.candidate_roots = dj_data['Symbol'].tolist()
    #     else:
    #         pass

    def set_candidate_roots(self) -> None:
        """
        Fetches constituents of a specified index using public data sources and updates candidate_roots.
        """
        if self.sp_500:
            sp_data = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
            self.candidate_roots = sp_data['Symbol'].str.replace('.', '-').tolist()
        elif self.nasdaq_100:
            nasdaq_tables = pd.read_html('https://en.wikipedia.org/wiki/Nasdaq-100')
            for i, table in enumerate(nasdaq_tables):
                if 'Symbol' in table.columns:
                    self.candidate_roots = table['Symbol'].tolist()
                    break
                elif 'Ticker' in table.columns:
                    self.candidate_roots = table['Ticker'].tolist()
                    break

            # If we couldn't find the right table, raise an error
            if not self.candidate_roots:
                raise ValueError("Could not find NASDAQ-100 constituents table with expected columns")
        elif self.dow_jones:
                # From the output, we can see Table 2 has the right structure
                dj_tables = pd.read_html('https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average')
                # Table 2 has 'Symbol' column
                self.candidate_roots = dj_tables[2]['Symbol'].tolist()
        else:
            pass  # No index selected, keep candidate_roots as is


if __name__ == "__main__":
    # Create a ContractUniverse instance
    universe = ContractUniverse(dow_jones=True)

    # Fetch the S&P 500 constituents
    universe.set_candidate_roots()

    # Print the candidate roots
    print(universe.candidate_roots)
    print(len(universe.candidate_roots))
