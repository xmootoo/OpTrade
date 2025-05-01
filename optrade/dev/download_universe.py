from pathlib import Path
import json
import yaml
from typing import Dict, List, Any
from rich.console import Console

# Custom modules
from optrade.data.universe import Universe
from optrade.dev.utils.ablations import generate_ablation_combinations

SCRIPT_DIR = Path(__file__).resolve().parent
UNIVERSE_DOWNLOAD_PARAMETERS = [
    "contracts.stride",
    "contracts.interval_min",
    "contracts.right",
    "contracts.target_tte",
    "contracts.tte_tolerance",
    "contracts.moneyness",
    "data.train_split",
    "data.val_split",
    "contracts.strike_band",
    "contracts.volatility_type",
    "contracts.volatility_scaled",
    "contracts.volatility_scalar",
]


def filter_ablation_config(
    ablation_config: Dict[str, List[Any]], prefixes: List[str]
) -> Dict[str, List[Any]]:
    """
    Filter ablation configuration by prefixes and group them
    Args:
        ablation_config: The full ablation configuration
        prefixes: List of prefixes to filter by (e.g. ["data", "contracts"])
    Returns:
        Dictionary with filtered and grouped ablation parameters
    """
    # ctx = Console()
    filtered_config = ablation_config.copy()
    keys_to_delete = []

    for key in filtered_config:
        # Check if the key starts with any of the prefixes
        if not any(key.startswith(prefix) for prefix in prefixes):
            keys_to_delete.append(key)

    # Delete keys outside the loop to avoid modification during iteration
    for key in keys_to_delete:
        # ctx.log(f"Removing {key} from filtered config")
        del filtered_config[key]

    return filtered_config


def run_universe(parent_id: str, download: bool = False, filter: bool = False) -> None:
    """Set up universe, filter it, and optionally download data with ablation parameters

    Args:
        parent_id: The ID of the parent experiment
        download: Whether to download data or not
        filter: Whether to filter the universe or not

    Returns:
        None
    """

    ctx = Console()

    # Set up paths
    parent_path = (
        SCRIPT_DIR.parents[0] / "jobs" / "parent" / parent_id
    )
    universe_path = parent_path / "universe.yaml"

    # Load universe configuration
    with open(universe_path, "r") as file:
        universe_config = yaml.safe_load(file)

    # Initialize universe
    universe = Universe(
        start_date=universe_config["start_date"],
        end_date=universe_config["end_date"],
        sp_500=universe_config.get("sp_500", False),
        nasdaq_100=universe_config.get("nasdaq_100", False),
        dow_jones=universe_config.get("dow_jones", False),
        candidate_roots=universe_config.get("candidate_roots", None),
        volatility=universe_config.get("volatility", None),
        pe_ratio=universe_config.get("pe_ratio", None),
        debt_to_equity=universe_config.get("debt_to_equity", None),
        beta=universe_config.get("beta", None),
        market_cap=universe_config.get("market_cap", None),
        sector=universe_config.get("sector", None),
        industry=universe_config.get("industry", None),
        dividend_yield=universe_config.get("dividend_yield", None),
        earnings_volatility=universe_config.get("earnings_volatility", None),
        market_beta=universe_config.get("market_beta", None),
        size_beta=universe_config.get("size_beta", None),
        value_beta=universe_config.get("value_beta", None),
        profitability_beta=universe_config.get("profitability_beta", None),
        investment_beta=universe_config.get("investment_beta", None),
        momentum_beta=universe_config.get("momentum_beta", None),
        all_metrics=universe_config.get("all_metrics", False),
        save_dir=universe_config.get("save_dir", None),
        verbose=universe_config.get("verbose", False),
        dev_mode=universe_config.get("dev_mode", False),
    )

    # Set up universe
    ctx.log("Setting candidate roots...")
    universe.set_roots()
    ctx.log("Getting market metrics...")
    universe.get_market_metrics()
    ctx.log("Getting factor exposures...")
    universe.get_factor_exposures()
    ctx.log(f"Roots: {universe.roots}")

    # Filter the universe
    if filter:
        universe.filter()

    # Save market metrics
    market_metrics_path = parent_path / "market_metrics.json"
    ctx.log(f"Saving market metrics to {market_metrics_path}")
    with open(market_metrics_path, "w") as f:
        json.dump(universe.market_metrics, f, indent=4)

    if download:
        # Load ablation configuration
        ablation_path = parent_path / "ablation.yaml"
        with open(ablation_path, "r") as file:
            ablation_full_config = yaml.safe_load(file)["ablations"]

        ctx.log(f"Ablation full config: {ablation_full_config}")
        ctx.log(f"Keys: {list(ablation_full_config.keys())}")

        # Filter only by "contracts" and "data", as they are the only ones needed for downloading
        filtered_ablation_config = filter_ablation_config(
            ablation_full_config, UNIVERSE_DOWNLOAD_PARAMETERS
        )
        ablation_combinations = generate_ablation_combinations(filtered_ablation_config)

        for i, ablation in enumerate(ablation_combinations):
            ctx.log(f"Downloading data combination {i+1}/{len(ablation_combinations)}")

            # Download data with these parameters
            universe.download(
                contract_stride=ablation.get("contracts.stride"),
                interval_min=ablation.get("contracts.interval_min"),
                right=ablation.get("contracts.right"),
                target_tte=ablation.get("contracts.target_tte"),
                tte_tolerance=ablation.get("contracts.tte_tolerance"),
                moneyness=ablation.get("contracts.moneyness"),
                train_split=ablation.get("data.train_split"),
                val_split=ablation.get("data.val_split"),
                strike_band=ablation.get("contracts.strike_band", 0.05),
                volatility_type=ablation.get("contracts.volatility_type", "period"),
                volatility_scaled=ablation.get("contracts.volatility_scaled", False),
                volatility_scalar=ablation.get("contracts.volatility_scalar", 1.0),
            )

        ctx.log("Download complete!")


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("parent_id", type=str, help="Parent experiment ID")
    parser.add_argument("--d", action="store_true", help="Download data")
    parser.add_argument("--f", action="store_true", help="Filter universe")

    args = parser.parse_args()

    run_universe(
        parent_id=args.parent_id,
        download=args.d,
        filter=args.f,
    )
