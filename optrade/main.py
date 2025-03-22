import os
import json
import argparse
import sys
from datetime import datetime
from optrade.exp.exp import Experiment
from optrade.config.config import load_config
import torch
from dotenv import load_dotenv
from rich.console import Console
from rich.pretty import pprint
import warnings
from pydantic import BaseModel
from typing import Dict, Any

warnings.filterwarnings(
    "ignore", message="h5py not installed, hdf5 features will not be supported."
)


def update_global_config(
    ablation_config: Dict[str, Any], global_config: BaseModel, ablation_id: int
) -> BaseModel:
    """
    Updates the global config for ablation studies and hyperparameter tuning. For example,
    if the ablation_config is {'global_config.sl.lr': 0.01}, then the global_config.sl.lr will be
    updated to 0.01.
    """

    for key, value in ablation_config.items():
        parts = key.split(".")
        if len(parts) == 2:
            sub_model, param = parts
            if hasattr(global_config, sub_model):
                sub_config = getattr(global_config, sub_model)
                if hasattr(sub_config, param):
                    setattr(sub_config, param, value)
                else:
                    print(f"Warning: {sub_model} does not have attribute {param}")
            else:
                print(f"Warning: global_config does not have attribute {sub_model}")
        elif len(parts) == 3:
            model, sub_model, param = parts
            if model == "global" and hasattr(global_config, sub_model):
                sub_config = getattr(global_config, sub_model)
                if hasattr(sub_config, param):
                    setattr(sub_config, param, value)
                else:
                    print(f"Warning: {sub_model} does not have attribute {param}")
            else:
                print(f"Warning: Invalid key format or 'global' not specified: {key}")
        else:
            print(f"Warning: Invalid key format: {key}")

    global_config.exp.ablation_id = ablation_id

    return global_config


def main(job_name="test", ablation=None, ablation_id=1):

    # Rich console
    load_dotenv()
    console = Console()

    # Load experimental configuration
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    args_path = os.path.join(base_dir, "optrade", "jobs", job_name, "args.yaml")
    args = load_config(args_path)
    if ablation is not None:
        args = update_global_config(ablation, args, ablation_id)
    console.print("Pydantic Configuration Loaded Successfully:", style="bold green")

    # Run experiment on multiple seeds (optional)
    seed_list = args.exp.seed_list

    for i in range(len(seed_list)):
        args.exp.seed = seed_list[i]
        args.exp.time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Print args
        pprint(args, expand_all=True)

        # Initialize experiment
        exp = Experiment(args)

        console.log("Using single device")
        exp.run()


if __name__ == "__main__":
    # Non-hyperparameter tuning
    warnings.filterwarnings(
        "ignore", message="h5py not installed, hdf5 features will not be supported."
    )
    parser = argparse.ArgumentParser(description="Run experiment")
    parser.add_argument("job_name", type=str, default="test", help="Name of the job")
    parser.add_argument("--venv_test", action="store_true", help="Flag for venv test")
    parser.add_argument(
        "--ablation", type=str, default=None, help="Ablation study configuration"
    )
    parser.add_argument(
        "--ablation_id", type=int, default=1, help="Ablation study configuration"
    )

    args = parser.parse_args()

    if args.venv_test:
        print("Running in venv test mode")
        # Add any venv test specific code here
    else:
        if args.ablation is not None:
            try:
                ablation = json.loads(args.ablation)
                main(
                    job_name=args.job_name,
                    ablation=ablation,
                    ablation_id=args.ablation_id,
                )
            except json.JSONDecodeError:
                print("Error: Invalid JSON string for ablation configuration")
                sys.exit(1)
        else:
            main(args.job_name)
