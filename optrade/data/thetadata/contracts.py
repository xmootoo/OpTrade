from pydantic import BaseModel, Field
from typing import Tuple, Optional

# Custom modules
from optrade.data.thetadata.expirations import find_optimal_exp
from optrade.data.thetadata.strikes import find_optimal_strike

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

    @classmethod
    def find_optimal(
        cls,
        root: str = "AAPL",
        start_date: str = "20241107",
        interval_min: int = 1,
        right: str = "C",
        target_tte: int = 30,
        tte_tolerance: Tuple[int, int] = (25, 35),
        moneyness: str = "OTM",
        target_band: float = 0.05,
        hist_vol: Optional[float] = None,
        volatility_scaled: bool = True,
        volatility_scalar: float = 1.0,
    ) -> "Contract":
        """Find the optimal contract for a given security, start date, and approximate TTE."""
        exp, _ = find_optimal_exp(
            root=root,
            start_date=start_date,
            target_tte=target_tte,
            tte_tolerance=tte_tolerance,
            clean_up=True,
        )

        strike = find_optimal_strike(
            root=root,
            start_date=start_date,
            exp=exp,
            right=right,
            interval_min=interval_min,
            moneyness=moneyness,
            target_band=target_band,
            hist_vol=hist_vol,
            volatility_scaled=volatility_scaled,
            volatility_scalar=volatility_scalar,
            clean_up=True,
        )

        return cls(
            root=root,
            start_date=start_date,
            exp=exp,
            strike=strike,
            interval_min=interval_min,
            right=right
        )

if __name__ == "__main__":
    from rich.console import Console

    console = Console()

    contract = Contract.find_optimal(
        root="AAPL",
        start_date="20241107",
        interval_min=1,
        right="C",
        target_tte=30,
        tte_tolerance=(25, 35),
        moneyness="OTM",
        target_band=0.05,
        volatility_scaled=True,
        volatility_scalar=1.0,
        hist_vol=1.0
    )
    console.log(contract)
