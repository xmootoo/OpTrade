from typing import Optional, Dict, Any

from sklearn.base import BaseEstimator
from sklearn.ensemble import (
    RandomForestRegressor, GradientBoostingRegressor,
    RandomForestClassifier, GradientBoostingClassifier
)
from sklearn.linear_model import Lasso, Ridge, RidgeClassifier, LogisticRegression
from sklearn.multioutput import MultiOutputRegressor

from xgboost import XGBRegressor, XGBClassifier
from catboost import CatBoostRegressor, CatBoostClassifier

def get_sklearn_model(
    model_id: str,
    target_type: str,
    model_args: Optional[Dict[str, Any]] = None
) -> BaseEstimator:
    """
    Instantiate and return a scikit-learn (or compatible) estimator by ID,
    unpacking all kwargs from a dict.

    Args:
        model_id (str): One of
            "random_forest", "gradient_boosting", "lasso", "linear",
            "xgboost", or "catboost".
        target_type (str):
            - "multistep": forecasting a vector of future values
            - "average": forecasting a single continuous value
            - "average_direction": forecasting a binary label (0 or 1)
        model_args (Optional[Dict[str,Any]]):
            All keyword args to pass to the underlying constructor, e.g.
            {"n_estimators":200, "max_depth":5}.
            If None, defaults to {}.

    Returns:
        BaseEstimator: The configured estimator.  If target_type=="multistep",
        wraps in MultiOutputRegressor.
    """
    args = {} if model_args is None else model_args.copy()
    args.setdefault("random_state", 1995)

    if model_id == "rf":
        Est = (
            RandomForestClassifier
            if target_type == "average_direction"
            else RandomForestRegressor
        )
        base = Est(**args)

    elif model_id == "gb":
        Est = (
            GradientBoostingClassifier
            if target_type == "average_direction"
            else GradientBoostingRegressor
        )
        base = Est(**args)

    elif model_id == "lasso":
        if target_type == "average_direction":
            base = LogisticRegression
        else:
            base = Lasso(**args)

    elif model_id == "ridge":
        if target_type == "average_direction":
            base = RidgeClassifier(**args)
        else:
            base = Ridge(**args)

    elif model_id == "xgb":
        Est = XGBClassifier if target_type == "average_direction" else XGBRegressor
        base = Est(**args)

    elif model_id == "cb":
        # CatBoost expects "iterations" not "n_estimators"
        cb_args = args.copy()
        if "n_estimators" in cb_args:
            cb_args["iterations"] = cb_args.pop("n_estimators")
        # rename random_state → random_seed
        cb_args.setdefault("random_seed", cb_args.pop("random_state"))
        Est = (
            CatBoostClassifier
            if target_type == "average_direction"
            else CatBoostRegressor
        )
        # silence the verbose by default
        cb_args.setdefault("verbose", False)
        base = Est(**cb_args)

    else:
        raise ValueError(f"Unknown model_id: {model_id}")

    if target_type == "multistep" and model_id in ["rf", "gb", "xgb", "cb"]:
        return MultiOutputRegressor(base)

    return base
