# Step 1: Find a set of optimal contracts from total_start_date to total_end_date
from optrade.data.contracts import ContractDataset

contract_dataset = ContractDataset(
    root="AMZN",
    total_start_date="20220101",
    total_end_date="20220301",
    contract_stride=1,
    interval_min=1,
    right="P",
    target_tte=3,
    tte_tolerance=(1,10),
    moneyness="ITM",
    strike_band=0.05,
    volatility_scaled=True,
    volatility_scalar=0.1,
    hist_vol=0.1117,
)
contract_dataset.generate()

# Step 2: Load market data and transform features for all contracts then put into a concatenated torch dataset
from optrade.data.forecasting import get_forecasting_dataset
from torch.utils.data import DataLoader

torch_dataset = get_forecasting_dataset(
    contract_dataset=contract_dataset,
    core_feats=["option_returns"],
    tte_feats=["sqrt"],
    datetime_feats=["sin_minute_of_day"],
    tte_tolerance=(25, 35),
    seq_len=100,
    pred_len=10,
    verbose=True
)
torch_loader = DataLoader(torch_dataset)
