import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import os
import yaml
import time
from datetime import datetime
from pydantic import BaseModel, Field, validator
from typing import Tuple, List, Optional

import random
import string

SCRIPT_DIR = Path(__file__).resolve().parent


def generate_random_id(length=10):
    # Seed the random number generator with current time and os-specific random data
    random.seed(int(time.time() * 1000) ^ os.getpid())

    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


class Experiment(BaseModel):
    parent: Optional[str] = Field(
        default=None,
        description="Parent experiment ID for nested experiments. If not None, it will replace all Global values specified in the jobs/parent/<parent_name>/ablation.yaml for experiments with parent=<parent_name>.",
    )
    model_id: str = Field(
        default="PatchTSTBlind",
        description="Model ID. Options: 'PatchTSTOG', 'PatchTSTBlind', 'JEPA', 'DualJEPA'",
    )
    seed_list: List[int] = Field(
        default=[2024],
        description="List of random seeds to run a single experiment on.",
    )
    seed: int = Field(default=2024, description="Random seed")
    log_dir: Path = Field(
        default=SCRIPT_DIR.parent, description="Directory to save the 'logs' folder."
    )
    id: str = Field(
        default_factory=generate_random_id,
        description="Experiment ID, randomly generated 10-character string",
    )
    neptune: bool = Field(
        default=False,
        description="Whether to use Neptune for logging. If False, offline logging (JSON) will be used.",
    )
    api_token: str = Field(
        default=os.environ.get("NEPTUNE_API_TOKEN", ""), description="Neptune API token"
    )
    project_name: str = Field(
        default="xmootoo/soz-localization", description="Neptune project name"
    )
    time: str = Field(
        default=datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
        description="Neptune run ID",
    )
    thresh: float = Field(
        default=0.5, description="The threshold for binary classification"
    )
    mps: bool = Field(
        default=False,
        description="Whether to use MPS for Apple silicon hardware acceleration",
    )
    task: str = Field(
        default="forecasting",
        description="Task type. Options: 'forecasting', 'classification'",
    )
    gpu_id: int = Field(
        default=0, description="GPU ID to use for for single device training"
    )
    ablation_id: int = Field(
        default=1, description="Ablation ID for the base experiment"
    )
    sklearn: bool = Field(
        default=False,
        description="Whether to use sklearn for modeling, otherwise, defaults to PyTorch",
    )


class Data(BaseModel):
    dtype: str = Field(
        default="float32", description="Type of data. Options: 'float32', 'float64'"
    )
    seq_len: int = Field(default=512, description="Sequence length of the input.")
    window_stride: int = Field(
        default=1, description="Window stride for generating windows."
    )
    pred_len: int = Field(
        default=96, description="Prediction length of the forecast window."
    )
    num_classes: int = Field(
        default=2, description="Number of classes for classification tasks."
    )
    num_channels: int = Field(default=321, description="Number of time series channels")
    balance: bool = Field(
        default=True, description="Balance classes within dataset for classification."
    )
    train_split: float = Field(
        default=0.6, description="Portion of data to use for training"
    )
    val_split: float = Field(
        default=0.2, description="Portion of data used for validation"
    )

    # PyTorch Dataloader
    batch_size: int = Field(default=32, description="Batch size for the dataloader")
    drop_last: bool = Field(
        default=False, description="Whether to drop the last batch."
    )
    persistent_workers: bool = Field(
        default=True,
        description="Whether to use persistent workers for the dataloader.",
    )
    shuffle: bool = Field(default=True, description="Whether to shuffle all datasets")
    num_workers: int = Field(
        default=4, description="Number of workers for the dataloader"
    )
    pin_memory: bool = Field(
        default=True, description="Whether to pin memory for the dataloader"
    )
    prefetch_factor: Optional[int] = Field(
        default=2, description="Prefetch factor for the dataloader"
    )

    @validator("prefetch_factor", pre=True)
    def convert_zero_to_none(cls, v):
        if v == 0:
            return None
        return v

    shuffle_test: bool = Field(
        default=False, description="Whether to shuffle the test set"
    )

    # Download Parameters
    clean_up: bool = Field(
        default=False,
        description="Whether to clean up the CSV files after saving data from ThetaData API.",
    )
    offline: bool = Field(
        default=False,
        description="Whether to use offline data instead of calling ThetaData API directly.",
    )
    download_only: bool = Field(
        default=False,
        description="Whether to download data only and not run the experiment.",
    )
    save_dir: Optional[str] = Field(
        default=None, description="Directory to save the data."
    )
    verbose: bool = Field(default=False, description="Verbose print for data loading.")
    dev_mode: bool = Field(
        default=False,
        description="Whether to use development mode (uses specific directory management for saving and loading data).",
    )


class Features(BaseModel):
    target_channels: Optional[List[str]] = Field(
        default=["option_returns"],
        description="Target channels used in the target window.",
    )
    target_type: str = Field(
        default="multistep",
        description="Target type different forecasting tasks. Options: 'multistep', 'average', or 'average_direction'.",
    )
    core: List[str] = Field(
        default=["option_mid_price"],
        description="Core features of option and underlying data",
    )
    tte: List[str] = Field(
        default=["sqrt"],
        description="Time-to-Expiration (TTE) features for option contracts.",
    )
    datetime: List[str] = Field(
        default=["sin_minute_of_day"],
        description="Datetime features to use for the time series.",
    )
    vol: List[str] = Field(
        default=["rolling_volatility"],
        description="Volatility features to use for the time series.",
    )
    rolling_volatility_range: List[int] = Field(
        default=[20, 60],
        description="Rolling volatility features to use for the time series.",
    )
    keep_datetime: bool = Field(
        default=False,
        description="Whether to keep datetime features in the dataset.",
    )
    scaling: bool = Field(
        default=False,
        description="Whether to use z-score normalize each channel using StandardScaler (scikit-learn).",
    )
    intraday: bool = Field(
        default=False,
        description="Whether to use intraday data for the dataset or allow crossover between different days.",
    )


class Contracts(BaseModel):
    # Contract Parameters
    root: str = Field(
        default="AAPL", description="The root symbol of the underlying security"
    )
    start_date: str = Field(
        default="20231107",
        description="Start date of the dataset of multiple option contracts",
    )
    end_date: str = Field(
        default="20241114",
        description="End date of the dataset of multiple option contracts",
    )
    interval_min: int = Field(
        default=1,
        description="Interval length in minutes for requested data. Mininimum is 1minute resolution.",
    )
    right: str = Field(
        default="C",
        description="Right of the option of contract. 'C' for call and 'P' for put.",
    )
    moneyness: str = Field(
        default="ATM",
        description="Moneyness of the option contract. Options: 'ATM', 'ITM', or 'OTM'.",
    )
    strike_band: float = Field(
        default=0.05,
        description="Target band in proportion between 0 and 1 used for fixed strike selection (e.g., strike_band=0.05 uses +/- 5% of current underlying price",
    )
    volatility_scaled: bool = Field(
        default=True,
        description="Whether to select strikes based on scaled historical volatility",
    )
    volatility_scalar: float = Field(
        default=0.1,
        description="Scalar to amplify or attenuate the effect of historical volatility on selecting strikes",
    )
    volatility_type: str = Field(
        default="period",
        description="The time range to calculate historical volatility for volatility-based strike selection",
    )
    target_tte: int = Field(
        default=30,
        description="Desired length of the contract in days. The find_optimal() method of the Contract class will attempt to the contract with the closests TTE",
    )
    tte_tolerance: Tuple[int, int] = Field(
        default=(15, 45),
        description="Tolerance range for selecting a contract if exact target_tte does not exist",
    )
    stride: int = Field(
        default=5,
        description="Stride length used to select multiple contracts at different dates",
    )
    validate: bool = Field(
        default=False,
        description="Whether to validate the contracts in the dataset.",
    )
    modify: bool = Field(
        default=False,
        description="Whether to modify and overwite contracts .pkl files if any contracts are invalid.",
    )


class Conformal(BaseModel):
    conf: bool = Field(
        default=False,
        description="Whether to use conformal prediction. This will split .",
    )
    alpha: float = Field(
        default=0.1,
        description="Significance level (i.e. error rate) for the conformal prediction.",
    )
    corrected: bool = Field(
        default=False,
        description="Whether to use Bonferonni correction for conformal time series forecasting. This will use the corrected_critical_scores.",
    )
    intervals: bool = Field(
        default=True,
        description="Whether to report interval width mean and standard deviation for evaluation metrics.",
    )
    validation_eval: bool = Field(
        default=False,
        description="Whether evaluate the model for coverage on the validation set at each epoch. If False, it only evaluates on the test set.",
    )


class Train(BaseModel):
    early_stopping: bool = Field(
        default=True, description="Whether to use early stopping"
    )
    optimizer: str = Field(
        default="adam",
        description="Optimizer for supervised learning: 'adam' or 'adamw'",
    )
    criterion: str = Field(
        default="MSE",
        description="Criterion for supervised learning: 'MSE', 'SmoothL1', 'CrossEntropy', 'BCE', 'ChannelLossBCE', 'ChannelLossCE'",
    )
    num_enc_layers: int = Field(
        default=3, description="Number of encoder layers in the model"
    )
    dropout: float = Field(
        default=0.05, description="Dropout for some of the linears layers in PatchTSTOG"
    )
    batch_first: bool = Field(
        default=True, description="Whether the first dimension is batch"
    )
    norm_mode: str = Field(
        default="batch1d",
        description="Normalization mode: 'batch1d', 'batch2d', or 'layer'",
    )
    batch_size: int = Field(default=64, description="Batch size")
    revin: bool = Field(
        default=True, description="Whether to use instance normalization with RevIN."
    )
    revout: bool = Field(
        default=True, description="Whether to use add mean and std back after forecast."
    )
    revin_affine: bool = Field(
        default=True,
        description="Whether to use learnable affine parameters for RevIN.",
    )
    eps_revin: float = Field(
        default=1e-5, description="Epsilon value for reversible input"
    )
    lr: float = Field(default=1e-4, description="Learning rate")
    epochs: int = Field(default=100, description="Number of epochs to train")
    scheduler: str = Field(
        default=None, description="Scheduler to use for learning rate annealing"
    )
    weight_decay: float = Field(
        default=1e-6, description="Weight decay for the optimizer"
    )
    dataset_class: str = Field(
        default="forecasting",
        description="Task type: 'forecasting', 'forecasting_og', 'classification', 'JEPA', 'DualJEPA'",
    )
    early_stopping: bool = Field(
        default=False, description="Early stopping for supervised learning."
    )
    early_stopping_patience: int = Field(
        default=10,
        description="Patience for early stopping. Number of epochs to wait before stopping.",
    )
    head_type: str = Field(
        default="linear",
        description="Head type for supervised learning: 'linear' or 'mlp'",
    )  # CyclicalPatchedForecaster (usable)
    return_head: bool = Field(
        default=False, description="Whether to return the head of the model."
    )
    channel_independent: bool = Field(
        default=False,
        description="Whether to use channel independent linear layers for the head.",
    )


class Evaluation(BaseModel):
    metrics: List[str] = Field(
        default=["mse"],
        description="Metrics to use for evaluation. Options (regression): 'mse', 'rmse', 'mae', 'mape', 'r2'. Options (classification): 'accuracy', 'f1', 'precision', 'recall', 'auc'.",
    )
    best_model_metric: str = Field(
        default="mse",
        description="Metric to use for model saving and early stopping. Options: 'loss', 'acc', 'ch_acc'",
    )


class Scheduler(BaseModel):
    warmup_steps: int = Field(
        default=15,
        description="Number of warmup epochs for the scheduler for cosine_warmup",
    )
    start_lr: float = Field(
        default=1e-4,
        description="Starting learning rate for the scheduler for warmup, for cosine_warmup",
    )
    ref_lr: float = Field(
        default=1e-3,
        description="End learning rate for the scheduler after warmp, for cosine_warmup",
    )
    final_lr: float = Field(
        default=1e-6,
        description="Final learning rate by the end of the schedule (starting from ref_lr) for cosine_warmup",
    )
    T_max: int = Field(
        default=100,
        description="Maximum number of epochs for the scheduler for CosineAnnealingLR or cosine_warmup",
    )
    last_epoch: int = Field(
        default=-1,
        description="Last epoch for the scheduler or CosineAnnealingLR or cosine_warmup",
    )
    eta_min: float = Field(
        default=1e-6, description="Minimum learning rate for CosineAnnealingLR"
    )
    pct_start: float = Field(
        default=0.3,
        description="Percentage of the cycle (in number of steps) spent increasing the learning rate for OneCycleLR",
    )
    lradj: str = Field(
        default="type3",
        description="Learning rate adjustment type (ontop of scheduling). Options: 'type3', 'TST'",
    )


class PatchTST(BaseModel):
    patch_dim: int = Field(default=16, description="Patch dimension or patch length.")
    patch_stride: int = Field(
        default=8, description="Patch stride for generating patches.}"
    )
    num_enc_layers: int = Field(
        default=3, description="Number of encoder layers for the PatchTST model."
    )
    d_model: int = Field(
        default=16, description="Model dimension for the PatchTST model."
    )
    d_ff: int = Field(
        default=128, description="FeedForward dimension for the PatchTST model."
    )
    num_heads: int = Field(
        default=4, description="Number of heads for the PatchTST model."
    )
    attn_dropout: float = Field(
        default=0.3,
        description="Dropout rate for attention mechanism in the PatchTST model.",
    )
    ff_dropout: float = Field(
        default=0.3,
        description="Dropout rate for feedforward mechanism in the PatchTST model.",
    )
    pred_dropout: float = Field(
        default=0.0,
        description="Dropout rate for prediction mechanism in the PatchTST model.",
    )
    norm_mode: str = Field(
        default="batch1d", description="Normalization mode for the PatchTST model."
    )


class DLinear(BaseModel):
    moving_avg: int = Field(
        default=25, description="Moving average window for the DLinear model."
    )
    individual: bool = Field(
        default=False,
        description="Whether to use model channels together or separately.",
    )

    # PatchedForecaster
    final_moving_avg: int = Field(
        default=25,
        description="Moving average window for the final forecast in the DLinear model.",
    )


class TSMixer(BaseModel):
    num_enc_layers: int = Field(
        default=2, description="Number of encoder layers for the TSMixer model."
    )
    dropout: float = Field(
        default=0.3, description="Dropout rate for the TSMixer model."
    )


class EMForecaster(BaseModel):
    patch_dim: int = Field(default=16, description="Patch dimension or patch length.")
    patch_stride: int = Field(
        default=8, description="Patch stride for generating patches.}"
    )
    patch_norm: str = Field(
        default="none",
        description="Normalization mode for the PatchedForecaster model. Options: 'none', 'layer'",
    )
    patch_act: str = Field(
        default="gelu",
        description="Activation function for the PatchedForecaster model",
    )
    patch_embed_dim: int = Field(
        default=128, description="Embedding dimension for the PatchedForecaster model"
    )
    pos_enc: str = Field(
        default="none",
        description="Positional encoding for the PatchedForecaster model. Options: 'none', 'learnable'",
    )
    patch_model_id: str = Field(
        default="TSMixer",
        description="Patch model for the EMForecaster class. Options: 'TSMixer', 'DLinear'.",
    )
    d_model: int = Field(
        default=24,
        description="The hidden MLP dimension for TSMixer when using patch_model_id='TSMixer'.",
    )
    dropout: float = Field(
        default=0.3, description="Dropout rate for the EMForecaster model."
    )
    num_enc_layers: int = Field(
        default=2, description="Number of encoder layers for the EMForecaster model."
    )
    moving_avg: int = Field(
        default=25,
        description="Moving average window for the EMForecaster model when patch_model_id='DLinear'.",
    )


class RecurrentModel(BaseModel):
    d_model: int = Field(
        default=16, description="Model dimension for the RecurrentModel."
    )
    bidirectional: bool = Field(
        default=False,
        description="Whether to use bidirectional LSTM (typically for classification).",
    )
    last_state: bool = Field(
        default=True,
        description="Whether to use the last state hidden of the LSTM (typically for classification).",
    )
    avg_state: bool = Field(
        default=False,
        description="Whether to use the average the hidden states of the LSTM.",
    )
    backbone_id: str = Field(
        default="LSTM",
        description="Backbone for the RecurrentModel class. Options: 'LSTM', 'RNN', 'GRU', 'Mamba'.",
    )
    num_enc_layers: int = Field(
        default=1, description="Number of encoder layers for the RecurrentModel."
    )
    patching: bool = Field(
        default=False, description="Whether to use patching for the dataset (LSTM only)"
    )
    patch_dim: int = Field(default=16, description="Patch dimension or patch length.")
    patch_stride: int = Field(
        default=8, description="Patch stride for generating patches.}"
    )

class SklearnConfig(BaseModel):
    tuning_method: str = Field(
        default="grid", description="Tuning method for sklearn models. Options: 'grid', 'random'."
    )
    n_splits: int = Field(
        default=5, description="Number of cross-validation splits for sklearn models (this will be fed into TimeSeriesSplit for temporal separation)."
    )
    verbose: int = Field(
        default=0, description="Verbosity level for sklearn models. 0 = silent, 1 = some output, 2 = more output."
    )
    n_jobs: int = Field(
        default=-1, description="Number of jobs to run in parallel for sklearn models. -1 = all processors."
    )
    n_iter: int = Field(
        default=10, description="Number of iterations for RandomizedSearchCV in hyperparameter tuning sklearn models."
    )

class SklearnRidge(BaseModel):
    """Configuration for sklearn.linear_model.Ridge."""

    alpha: float = Field(
        default=1.0, description="Regularization strength; must be a positive float."
    )
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model."
    )
    copy_X: bool = Field(
        default=True,
        description="If True, X will be copied; else, it may be overwritten.",
    )
    tol: float = Field(default=0.001, description="Solver termination tolerance.")
    max_iter: Optional[int] = Field(
        default=None, description="Maximum number of iterations for the solver."
    )
    solver: str = Field(
        default="auto", description="Solver to use in the computational routines."
    )
    positive: bool = Field(
        default=False, description="When True, constrains coefficients to be positive."
    )
    random_state: Optional[int] = Field(
        default=1995, description="Random number seed for 'sag' and 'saga' solvers."
    )


class SklearnLasso(BaseModel):
    """Configuration for sklearn.linear_model.Lasso."""

    alpha: float = Field(
        default=1.0, description="Constant that multiplies the L1 term."
    )
    fit_intercept: bool = Field(
        default=True, description="Whether to calculate the intercept for this model."
    )
    max_iter: int = Field(
        default=1000, description="Maximum number of iterations for optimization."
    )
    tol: float = Field(default=0.0001, description="Tolerance for the optimization.")
    selection: str = Field(
        default="cyclic",
        description="If 'random', a random coefficient is updated each iteration.",
    )


class SklearnRandomForest(BaseModel):
    """Configuration for sklearn.ensemble.RandomForestRegressor/Classifier."""

    n_estimators: int = Field(default=100, description="Number of trees in the forest.")
    criterion: str = Field(
        default="squared_error",
        description="Function to measure the quality of a split.",
    )
    max_depth: Optional[int] = Field(
        default=None, description="Maximum depth of each tree."
    )
    min_samples_split: int = Field(
        default=2,
        description="Minimum number of samples required to split an internal node.",
    )
    min_samples_leaf: int = Field(
        default=1,
        description="Minimum number of samples required to be at a leaf node.",
    )
    max_features: str = Field(
        default="auto",
        description="Number of features to consider when looking for the best split.",
    )
    bootstrap: bool = Field(
        default=True,
        description="Whether bootstrap samples are used when building trees.",
    )
    n_jobs: Optional[int] = Field(
        default=None, description="Number of jobs to run in parallel."
    )
    random_state: int = Field(
        default=1995, description="Controls randomness of the estimator."
    )


class SklearnGradientBoosting(BaseModel):
    """Configuration for sklearn.ensemble.GradientBoostingRegressor/Classifier."""

    n_estimators: int = Field(
        default=100, description="Number of boosting stages to perform."
    )
    learning_rate: float = Field(
        default=0.1, description="Learning rate shrinks contribution of each tree."
    )
    max_depth: int = Field(
        default=3, description="Maximum depth of the individual regression estimators."
    )
    min_samples_split: int = Field(
        default=2,
        description="Minimum number of samples required to split an internal node.",
    )
    min_samples_leaf: int = Field(
        default=1,
        description="Minimum number of samples required to be at a leaf node.",
    )
    subsample: float = Field(
        default=1.0,
        description="Fraction of samples to be used for fitting the individual base learners.",
    )
    max_features: Optional[str] = Field(
        default=None,
        description="Number of features to consider when looking for the best split.",
    )
    random_state: int = Field(
        default=1995, description="Controls randomness of the estimator."
    )


class SklearnXGBoost(BaseModel):
    """Configuration for xgboost.XGBRegressor/XGBClassifier."""

    n_estimators: int = Field(
        default=100, description="Number of gradient boosted trees."
    )
    learning_rate: float = Field(default=0.1, description="Boosting learning rate.")
    max_depth: int = Field(
        default=6, description="Maximum tree depth for base learners."
    )
    subsample: float = Field(
        default=1.0, description="Subsample ratio of the training instance."
    )
    colsample_bytree: float = Field(
        default=1.0,
        description="Subsample ratio of columns when constructing each tree.",
    )
    gamma: float = Field(
        default=0.0,
        description="Minimum loss reduction required to make a further partition.",
    )
    reg_alpha: float = Field(
        default=0.0, description="L1 regularization term on weights."
    )
    reg_lambda: float = Field(
        default=1.0, description="L2 regularization term on weights."
    )
    random_state: int = Field(default=1995, description="Random number seed.")


class SklearnCatBoost(BaseModel):
    """Configuration for catboost.CatBoostRegressor/CatBoostClassifier."""

    iterations: int = Field(
        default=1000, description="The maximum number of trees that can be built."
    )
    learning_rate: float = Field(default=0.03, description="The learning rate.")
    depth: int = Field(default=6, description="Depth of the tree.")
    l2_leaf_reg: float = Field(
        default=3.0, description="L2 regularization coefficient."
    )
    border_count: int = Field(
        default=254, description="The number of splits for numerical features."
    )
    bagging_temperature: float = Field(
        default=1.0, description="Controls intensity of Bayesian bagging."
    )
    random_strength: float = Field(default=1.0, description="The randomness strength.")
    random_state: int = Field(default=1995, description="Random number seed.")


class Global(BaseModel):
    exp: Experiment = Experiment()
    data: Data = Data()
    contracts: Contracts = Contracts()
    feats: Features = Features()
    train: Train = Train()
    eval: Evaluation = Evaluation()
    scheduler: Scheduler = Scheduler()

    # PyTorch models
    patchtst: PatchTST = PatchTST()
    dlinear: DLinear = DLinear()
    tsmixer: TSMixer = TSMixer()
    conf: Conformal = Conformal()
    emf: EMForecaster = EMForecaster()
    rnn: RecurrentModel = RecurrentModel()

    # Sklearn models
    sklearn: SklearnConfig = SklearnConfig()
    sklearn_ridge: SklearnRidge = SklearnRidge()
    sklearn_lasso: SklearnLasso = SklearnLasso()
    sklearn_rf: SklearnRandomForest = SklearnRandomForest()
    sklearn_gb: SklearnGradientBoosting = SklearnGradientBoosting()
    sklearn_xgb: SklearnXGBoost = SklearnXGBoost()
    sklearn_cb: SklearnCatBoost = SklearnCatBoost()


def load_config(file_path: Path) -> Global:
    print(f"Received file_path in load_config: {file_path}")
    print(f"Absolute file_path in load_config: {file_path.absolute()}")
    with open(file_path, "r") as file:
        config_data = yaml.safe_load(file)
    return Global(**config_data)
