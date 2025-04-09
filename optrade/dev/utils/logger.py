import os
import json
import shutil
import neptune
from pydantic import BaseModel
from rich.console import Console
from datetime import datetime
from typing import Any, Dict

# Import yaml
import yaml

from optrade.config.config import Global


def offline_to_neptune():
    console = Console()
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)

    if not os.listdir(log_dir):
        console.log("[bold red]No logs to upload[/bold red]")
        return

    for exp in os.listdir(log_dir):
        exp_dir = os.path.join(log_dir, exp)
        if not os.path.isdir(exp_dir):
            console.log(f"[yellow]Skipping {exp_dir} as it's not a directory[/yellow]")
            continue

        exp_contents = os.listdir(exp_dir)
        if not exp_contents:
            console.log(f"[yellow]No contents found in {exp_dir}. Skipping...[/yellow]")
            continue

        processed_successfully = False
        if any(os.path.isfile(os.path.join(exp_dir, item)) for item in exp_contents):
            processed_successfully = process_experiment(exp_dir, console)
        else:
            processed_successfully = True
            for run in exp_contents:
                run_dir = os.path.join(exp_dir, run)
                if os.path.isdir(run_dir):
                    if not process_experiment(run_dir, console):
                        processed_successfully = False

        if processed_successfully:
            move_to_old_logs(exp_dir, console)
        else:
            console.log(
                f"[yellow]Skipping move of {exp} due to processing errors[/yellow]"
            )


def process_experiment(directory, console):
    try:
        json_file = find_json_file(directory, console)
        if not json_file:
            return False

        logs = read_json_file(directory, json_file, console)
        if not logs:
            return False

        neptune_logger = initialize_neptune_logger(logs, console)
        if not neptune_logger:
            return False

        upload_model_weights(directory, logs, neptune_logger, console)
        log_metrics_and_parameters(logs, neptune_logger, console)

        neptune_logger.stop()
        console.log(f"[green]Successfully processed and uploaded {directory}[/green]")
        return True

    except Exception as e:
        console.log(
            f"[bold red]Unexpected error processing {directory}: {str(e)}[/bold red]"
        )
        return False


def find_json_file(directory, console):
    try:
        return next(f for f in os.listdir(directory) if f.endswith(".json"))
    except StopIteration:
        console.log(f"[blue]No JSON file found in {directory}. Skipping...[/blue]")
        return None


def read_json_file(directory, json_file, console):
    try:
        with open(os.path.join(directory, json_file), "r") as f:
            return json.load(f)
    except json.JSONDecodeError:
        console.log(
            f"[bold red]Error: Unable to parse JSON file in {directory}[/bold red]"
        )
    except IOError:
        console.log(
            f"[bold red]Error: Unable to read JSON file in {directory}[/bold red]"
        )
    return None


def initialize_neptune_logger(logs, console):
    try:
        api_token = os.environ.get("NEPTUNE_API_TOKEN", "")
        return neptune.init_run(
            project=logs["parameters/exp/project_name"],
            api_token=api_token,
            custom_run_id=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        )
    except Exception as e:
        console.log(f"[bold red]Error initializing Neptune logger: {str(e)}[/bold red]")
        return None


def upload_model_weights(directory, logs, neptune_logger, console):
    try:
        model_weights = next(f for f in os.listdir(directory) if f.endswith(".pth"))
        model_id = logs["parameters/exp/model_id"]
        neptune_logger[f"model_checkpoints/{model_id}"].upload(
            os.path.join(directory, model_weights)
        )
    except StopIteration:
        console.log(
            f"[blue]No .pth file found in {directory}. Skipping model weight upload...[/blue]"
        )
    except Exception as e:
        console.log(f"[bold red]Error uploading model weights: {str(e)}[/bold red]")


def log_metrics_and_parameters(logs, neptune_logger, console):
    for key, value in logs.items():
        try:
            if isinstance(value, list):
                for v in value:
                    neptune_logger[key].append(v)
            else:
                neptune_logger[key] = value
        except Exception as e:
            console.log(f"[bold red]Error logging {key}: {str(e)}[/bold red]")


def move_to_old_logs(source_dir, console):
    destination_dir = "old_logs"
    destination_path = os.path.join(destination_dir, os.path.basename(source_dir))
    os.makedirs(destination_dir, exist_ok=True)
    console.log(
        f"Attempting to move run folder from {source_dir} to {destination_path}"
    )
    try:
        shutil.move(source_dir, destination_path)
        console.log(
            f"[green]Successfully moved {source_dir} to {destination_dir}[/green]"
        )
    except Exception as e:
        console.log(f"[bold red]Error moving {source_dir}: {str(e)}[/bold red]")

    if os.path.exists(destination_path):
        if not os.path.exists(source_dir):
            console.log(f"[green]Verified: {source_dir} was moved successfully[/green]")
        else:
            console.log(
                f"[yellow]Warning: {source_dir} exists in both source and destination[/yellow]"
            )
    else:
        console.log(
            f"[bold red]Error: {source_dir} was not moved to the destination[/bold red]"
        )


def epoch_logger(args, logger, key, value):
    if args.exp.neptune:
        logger[key].append(value)
    else:
        if key not in logger:
            logger[key] = []
        logger[key].append(value)


def log_pydantic(logger, obj, key="parameters"):
    def log_model(model, current_key):
        for attr_name, attr_value in model.__dict__.items():
            if isinstance(attr_value, BaseModel):
                log_model(attr_value, f"{current_key}/{attr_name}")
            else:
                converted_value = convert_pydantic_types(attr_value)
                logger[f"{current_key}/{attr_name}"] = converted_value

    if isinstance(obj, BaseModel):
        log_model(obj, key)
    elif hasattr(obj, "__dict__"):
        for attr_name, attr_value in obj.__dict__.items():
            if isinstance(attr_value, BaseModel):
                log_model(attr_value, f"{key}/{attr_name}")
            else:
                converted_value = convert_pydantic_types(attr_value)
                logger[f"{key}/{attr_name}"] = converted_value
    else:
        print(f"Unsupported type for Neptune logging: {type(obj)}")


def convert_pydantic_types(value):
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    elif isinstance(value, tuple):
        return str(value)
    elif isinstance(value, dict):
        return {k: convert_pydantic_types(v) for k, v in value.items()}
    else:
        return value


# Converting PyDantic -> YAML for Neptune upload
def represent_none(self, _):
    return self.represent_scalar("tag:yaml.org,2002:null", "")


yaml.SafeDumper.add_representer(type(None), represent_none)


class CleanDumper(yaml.SafeDumper):
    def represent_tuple(self, data):
        return self.represent_list(list(data))


CleanDumper.add_representer(tuple, CleanDumper.represent_tuple)


def global_to_yaml(global_instance: Global, file_path: str) -> str:
    def convert_to_dict(obj: Any) -> Dict:
        if isinstance(obj, BaseModel):
            return {
                k: convert_to_dict(v) for k, v in obj.dict().items() if v is not None
            }
        elif isinstance(obj, (list, tuple)):
            return [convert_to_dict(item) for item in obj]
        elif isinstance(obj, dict):
            return {k: convert_to_dict(v) for k, v in obj.items() if v is not None}
        else:
            return obj

    global_dict = convert_to_dict(global_instance)
    yaml_string = yaml.dump(global_dict, Dumper=CleanDumper, default_flow_style=False)

    if file_path:
        with open(file_path, "w") as file:
            file.write(yaml_string)

    return yaml_string


if __name__ == "__main__":
    offline_to_neptune()
