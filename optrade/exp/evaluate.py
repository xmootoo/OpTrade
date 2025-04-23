import torch
import numpy as np
from typing import List, Union, Tuple
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

def get_metrics(
    metrics: List[str],
    output: Union[np.ndarray, torch.Tensor],
    target: Union[np.ndarray, torch.Tensor],
    target_type: str,
) -> Tuple[np.ndarray, List[str]]:
    """Computes evaluation metrics for regression and classification tasks.

    Args:
        metrics (List[str]): List of metrics to compute. Options include:
            - "mse": Mean Squared Error
            - "rmse": Root Mean Squared Error
            - "mae": Mean Absolute Error
            - "mape": Mean Absolute Percentage Error
            - "r2": R-squared score
            - "accuracy": Accuracy score (for classification)
            - "precision": Precision score (for classification)
            - "recall": Recall score (for classification)
            - "f1": F1 score (for classification)
            - "auc": Area Under the ROC Curve (for classification)
        output (Union[np.ndarray, torch.Tensor]): Model predictions with shape (num_examples, *).
            For multistep: (num_examples, num_target_features, pred_len)
            For average: (num_examples, num_target_features, 1)
            For average_direction: (num_examples, num_target_features, 1) in probability space
        target (Union[np.ndarray, torch.Tensor]): True target values with same shape as output.
        target_type (str): Type of target variable. Options include:
            - "multistep": Multistep regression
            - "average": Average regression
            - "average_direction": Average direction classification

    Returns:
        Tuple[np.ndarray, List[str]]: A tuple containing:
            - results (np.ndarray): Computed metrics.
            - results_key (List[str]): Corresponding metric names in the same order.
    """
    # Convert torch tensors to numpy if needed
    if isinstance(output, torch.Tensor):
        output = output.detach().cpu().numpy()
    if isinstance(target, torch.Tensor):
        target = target.detach().cpu().numpy()

    results = np.zeros(len(metrics))
    results_key = []
    i = 0

    # Regression metrics
    if target_type=="multistep" or target_type=="average":
        output = output.reshape(output.shape[0], -1) # (num_examples, num_target_features * pred_len)
        target = target.reshape(target.shape[0], -1) # (num_examples, num_target_features * pred_len)
        if "mse" in metrics:
            results[i] = mean_squared_error(target, output)
            results_key.append("mse")
            i+=1
        if "rmse" in metrics:
            results[i] = np.sqrt(mean_squared_error(target, output))
            results_key.append("rmse")
            i+=1
        if "mae" in metrics:
            results[i] = mean_absolute_error(target, output)
            results_key.append("mae")
            i+=1
        if "mape" in metrics:
            # Avoid division by zero
            if np.any(target == 0):
                results[i] = float("nan")
            else:
                results[i] = mean_absolute_percentage_error(target, output)
            results_key.append("mape")
            i+=1
        if "r2" in metrics:
            results[i] = r2_score(target, output)
            results_key.append("r2")
            i+=1

    # Classification metrics
    elif target_type=="average_direction":
        output = output.reshape(-1) # (num_examples * num_target_features)
        target = target.reshape(-1) # (num_examples * num_target_features)
        output_class = (output > 0.5).astype(int) # Binary: threshold at 0.5

        if "accuracy" in metrics:
            results[i] = accuracy_score(target, output_class)
            results_key.append("accuracy")
            i+=1

        if "precision" in metrics:
            results[i] = precision_score(target, output_class)
            results_key.append("precision")
            i+=1

        if "recall" in metrics:
            results[i] = recall_score(target, output_class)
            results_key.append("recall")
            i+=1

        if "f1" in metrics:
            results[i] = f1_score(target, output_class)
            results_key.append("f1")
            i+=1

        if "auc" in metrics:
            # AUC requires probability scores, not class predictions
            results[i] = roc_auc_score(target, output)
            results_key.append("auc")
            i+=1
    else:
        raise ValueError(f"Unknown target type: {target_type}. Supported types are 'multistep', 'average', 'average_direction'.")

    return results, results_key


if __name__ == "__main__":
    x = np.random.randn(32, 5, 10)
    y = np.random.randn(32, 5, 10)
    metrics = [
        "mse",
        "rmse",
        "mae",
        "mape",
        "r2",
    ]

    results, results_key = get_metrics(
        metrics,
        y,
        x,
        target_type="multistep",
    )
    print(f"results:{results}")
    print(f"Metric keys: {results_key}")


    # Example for classification
    x = np.random.randn(32, 5, 1)
    # Clip values to [0, 1] for binary classification
    x = np.clip(x, 0, 1)
    y = np.random.randint(0, 2, size=(32, 5, 1))  # Binary classification

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "auc",
    ]

    results, results_key = get_metrics(
        metrics,
        x,
        y,
        target_type="average_direction",
    )

    print(f"results:{results}")
    print(f"Metric keys: {results_key}")
