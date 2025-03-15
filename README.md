# OpTrade

A framework for forecasting and trading options using alpha term structures in American options markets.

<p align="center">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="optrade/assets/optrade_dark.png">
   <source media="(prefers-color-scheme: light)" srcset="optrade/assets/optrade_light.png">
   <img alt="OpTrade Framework" src="optrade/assets/optrade_light.png">
 </picture>
</p>

## Overview

OpTrade leverages state-of-the-art deep learning and machine learning models for time series forecasting in options markets. The project has two main objectives:

1. **Alpha Generation**: Discovering and forecasting alpha term structures to analyze market dynamics across various options contracts
2. **Trading Strategy Development**: Translating these insights into actionable trading signals (planned for future implementation)

Currently, the project is focused on completing the microstructure analysis framework.

## What is an Alpha Term Structure?

An alpha term structure represents how excess returns (alpha) are expected to evolve over different time horizons. It is defined as:

$$
\mathbf{r} = (r_1, r_2, \dots, r_H)^T
$$

Where:
- $r_t$ is the expected excess return of an option contract at time $t$
- The vector captures returns across multiple future time points

This structure helps traders:
- Determine optimal entry/exit points
- Develop time-specific trading strategies
- Manage risk (e.g., adjust positions) 
- Select appropriate option expiration dates

## Documentation
This project includes extensive documentation that is essential for understanding the framework. Users are strongly encouraged to review these documents before usage.

| Document | Description |
|----------|-------------|
| [DATA.md](DATA.md) | Information on the comprehensive data pipeline |
| [FEATURES.md](FEATURES.md) | Details on the selection of important predictors for option forecasting |

## Installation

### Dependencies
- Python ≥ 3.11
- Additional dependencies listed in `requirements.txt`

### Using conda (recommended)
```bash
# Create and activate conda environment
conda create -n venv python=3.11
conda activate venv

# Install requirements
cd <project_root_directory> # Go to project root directory
pip install -r requirements.txt
pip install -e .
```

### Using pip
```bash
# Create and activate virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install requirements
cd <project_root_directory> # Go to project root directory
pip install -r requirements.txt
pip install -e .
```


## Contact
For queries, please contact: `xmootoo at gmail dot com`.
