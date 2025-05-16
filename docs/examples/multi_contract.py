# Step 1: Find a set of optimal contracts from total_start_date to total_end_date
from optrade.data.contracts import ContractDataset
from rich.console import Console
import random
ctx = Console()

ctx.log("Constructing a dataset of put options for Amazon from January 1, 2022 to June 1, 2022...")
contract_dataset = ContractDataset(
    root="AMZN",
    total_start_date="20220101",
    total_end_date="20220601",
    contract_stride=3,
    interval_min=15,
    right="P",
    target_tte=30,
    tte_tolerance=(15,45),
    moneyness="OTM",
    volatility_scaled=True,
    volatility_scalar=0.1,
    hist_vol=0.1117,
)
with ctx.status("Generating contracts..."):
    contract_dataset.generate()
ctx.log(f"Found a total of {len(contract_dataset)} contracts!")
n = random.randint(0, len(contract_dataset)-1)
ctx.log(f"Randomly chosen contract ({n}): {contract_dataset[n]}")


# Step 2: Load market data and transform features for all contracts then put into a concatenated torch dataset
from optrade.data.forecasting import get_forecasting_dataset
from torch.utils.data import DataLoader

concat_dataset, updated_contract_dataset = get_forecasting_dataset(
    contract_dataset=contract_dataset,
    core_feats=["option_returns"],
    tte_feats=["sqrt"],
    datetime_feats=["sin_minute_of_day"],
    tte_tolerance=(25, 35),
    seq_len=100,
    pred_len=10,
    verbose=True
)
torch_loader = DataLoader(concat_dataset)


# Total number of examples
n = sum([len(concat_dataset.datasets[i]) for i in range(len(concat_dataset.datasets))])
ctx.log(f"A total of {n} training examples (input/target pairs) were generated.")
