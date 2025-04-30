from pathlib import Path
import itertools
import yaml
from typing import Dict, List, Any, Tuple
from rich.console import Console
import argparse

from optrade.config.config import load_config
from optrade.main import run_job
from optrade.dev.utils.ablations import load_ablation_config, generate_ablation_combinations

SCRIPT_DIR = Path(__file__).resolve().parent

def grid_search(job_name: str) -> None:
    # Adjust the base_path to use the absolute path
    base_path = SCRIPT_DIR.parents[0] / "jobs" # <- Might have to change due to refactor
    ablation_path = base_path / job_name / "ablation.yaml"

    # Rich
    ctx = Console()

    # Load the ablation configuration
    ablation_config = load_ablation_config(ablation_path)

    # Load args and check if 'parent' experiment is provided
    args_path = base_path / job_name / "args.yaml"
    args = load_config(args_path)

    if args.exp.parent is not None:
        with ctx.status(f"Adjusting ablation config according to parent: {args.exp.parent}..."):
            parent_path = base_path / "parent" / args.exp.parent / "ablation.yaml"
            parent_ablation_config = load_ablation_config(parent_path)

            original_ablation_config = ablation_config.copy()

            # Merge parent ablation config with current ablation config
            for key, value in parent_ablation_config.items():
                ablation_config[key] = value

    ctx.log(f"Original ablation config: {original_ablation_config}")
    ctx.log(f"Updated ablation config: {ablation_config}")

    # Generate all combinations of ablations
    ablation_combinations = generate_ablation_combinations(ablation_config)

    # Run each ablation
    for i, ablation in enumerate(ablation_combinations):
        ctx.log(f"Running ablation {i}: {ablation}")
        run_job(job_name=job_name, ablation=ablation, ablation_id=i)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "job_name", type=str, default="forecasting/test", help="Name of the ablation"
    )
    args = parser.parse_args()
    grid_search(args.job_name)
