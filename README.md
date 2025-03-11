# OpTrade
OpTrade is a framework designed for forecasting of alpha term structures in American options markets. The framework leverages state-of-the-art deep learning architectures specialized for time series forecasting. This project has two objectives: $(\textbf{I})$ discovering alpha term structures to analyze market microstructure dynamics across various options contracts via forecasting, and $(\textbf{II})$ translating these insights into actionable trading signals.
Currently, the project is focused on completing objective $(\textbf{I})$, with objective $(\textbf{II})$ planned for implementation upon successful completion of the microstructure analysis framework.

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="optrade/assets/optrade_dark.png">
    <source media="(prefers-color-scheme: light)" srcset="optrade/assets/optrade_light.png">
    <img alt="OpTrade Framework" src="optrade/assets/optrade_light.png">
  </picture>
</p>



## Documentation
1. [DATA.md](DATA.md)
2. [FEATURES.md](FEATURES.md)
2. [Installation](#installation)
3. [Contact](#contact)

The aove table provides relevant docuementation and information including [DATA.md](DATA.md), which provides information on the comprehensive data pipeline, and [FEATURES.md](FEATURES.md) which details the selection of important predictors for option forecasting.


## Installation
### Dependencies
- Python $\geq$ 3.11
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
