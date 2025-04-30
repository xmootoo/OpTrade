import itertools
import yaml
from typing import Dict, List, Any, Tuple
from pathlib import Path

def load_ablation_config(file_path: Path) -> Dict[str, List[Any]]:
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    ablation_config = config.get("ablations", {})
    return ablation_config


def generate_ablation_combinations(
    ablation_config: Dict[str, List[Any]],
) -> List[Dict[str, Any]]:
    keys, values = zip(*ablation_config.items())
    return [dict(zip(keys, v)) for v in itertools.product(*values)]