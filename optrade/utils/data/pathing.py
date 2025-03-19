from pathlib import Path
from typing import Tuple, Optional


def set_contract_dir(
    SCRIPT_DIR: Path,
    root: str = "AAPL",
    start_date: str = "20231107",
    end_date: str = "20241114",
    contract_stride: int = 5,
    interval_min: int = 1,
    right: str = "C",
    target_tte: int = 30,
    tte_tolerance: Tuple[int, int] = (25, 35),
    moneyness: str = "OTM",
    target_band: float = 0.05,
    volatility_scaled: bool = True,
    volatility_scalar: float = 1.0,
    hist_vol: Optional[float] = None,
    save_dir: Optional[str] = None
) -> Path:
    # Directory setup
    if save_dir is None:
        save_dir = SCRIPT_DIR.parent / "historical_data" / "contracts"
    else:
        save_dir = Path(save_dir) / "contracts"

    # Create a structured path based on key parameters
    contract_dir = (
        save_dir /
        root /
        f"{start_date}_{end_date}" /
        right /
        f"contract_stride_{contract_stride}" /
        f"interval_{interval_min}" /
        f"target_tte_{target_tte}" /
        f"moneyness_{moneyness}"
    )

    # Add volatility info to path if volatility_scaled is True
    if volatility_scaled:
        contract_dir = contract_dir / f"histvol_{hist_vol}_volscalar_{volatility_scalar}"
    else:
        contract_dir = contract_dir / f"target_band_{str(target_band).replace('.', 'p')}"

    return contract_dir
