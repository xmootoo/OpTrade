.. OpTrade documentation master file, created by
   sphinx-quickstart on Thu Mar 20 10:13:31 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

OpTrade
=====================

OpTrade is a complete toolkit for quantitative research and development of options trading strategies. By abstracting away the complexity of data handling and experimental design, researchers and traders can focus on what matters most: developing and testing alpha-generating ideas.

.. image:: _static/optrade_light.png
    :alt: OpTrade Framework
    :align: center


Installation
--------------
The recommended way to install OptTrade is via pip::

        pip install optrade

*Note: At this time OpTrade requires an active subscription to [ThetaData API](https://www.thetadata.net/subscribe) for the stocks (VALUE) and options (VALUE) packages.


Example (Single Contract)
----------

.. literalinclude:: examples/single_contract.py
   :language: python
   :linenos:


Overview
--------------

🔄 **Data Pipeline**
OpTrade integrates with ThetaData's API for affordable options and security data access (down to 1-min resolution). The framework processes NBBO quotes and OHLCVC metrics through a contract selection system optimizing for moneyness, expiration windows, and volatility-scaled strikes.

🌐 **Market Environments**
Built-in market environments enable precise universe selection through multifaceted filtering. OpTrade supports composition by major indices, fundamental-based screening (e.g., PE ratio, market cap), and Fama-French model categorization.

🧪 **Experimental Pipeline**
The experimentation framework supports PyTorch and scikit-learn for options forecasting with online Neptune logging, hyperparameter tuning, and model version control, supporting both online and offline experiment tracking.

🧮 **Featurization**
OpTrade provides option market features including mid-price derivations, order book imbalance metrics, quote spreads, and moneyness calculations. Time-to-expiration transformations capture theta decay effects, while datetime features extract cyclical market patterns for intraday seasonality.

🤖 **Models**
OpTrade includes several off-the-shelf PyTorch and scikit-learn models, including state-of-the-art architectures for time series forecasting alongside tried and true machine learning methods

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   modules
   contributing


Contact
--------------
For queries, please contact: `xmootoo at gmail dot com`.

Indices and tables
==================
* :ref:`modindex`
* :ref:`search`
