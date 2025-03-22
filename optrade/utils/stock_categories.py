from enum import Enum


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
    TECH = "technology"
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
