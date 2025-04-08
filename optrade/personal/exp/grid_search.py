from pathlib import Path
import itertools
import yaml
import sys
from typing import Dict, List, Any, Tuple
from rich.console import Console
import argparse

from optrade.main import run_job

SCRIPT_DIR = Path(__file__).resolve().parent


def load_ablation_config(file_path: Path) -> Tuple[Dict[str, List[Any]], str]:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    ablation_config = config.get("ablations", {})
    cc_base = config.get("cc", None)
    return ablation_config, cc_base


def generate_ablation_combinations(
    ablation_config: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    keys, values = zip(*ablation_config.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]


def grid_search(job_name: str) -> None:
    # Adjust the base_path to use the absolute path
    base_path = SCRIPT_DIR.parents[1] / "jobs"
    ablation_path = base_path / job_name / "ablation.yaml"

    print(f"Ablation path: {ablation_path}")

    # Rich
    console = Console()

    # Load the ablation configuration
    ablation_config, cc_base = load_ablation_config(ablation_path)

    # Generate all combinations of ablations
    ablation_combinations = generate_ablation_combinations(ablation_config)

    for i, ablation in enumerate(ablation_combinations):
        console.log(f"Running ablation {i} (locally): {ablation}")
        run_job(job_name=job_name, ablation=ablation, ablation_id=i)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "job_name", type=str, default="forecasting/test", help="Name of the ablation"
    )
    args = parser.parse_args()
    grid_search(args.job_name)
