Usage
=====

Example (Single Contract)
----------

Here's a simple example of how to load and transform data for a single contract:

.. literalinclude:: examples/single_contract.py
   :language: python
   :linenos:


Example (Multiple Contracts)
--------------

When modeling multiple contracts, you can use the `ContractDataset` class
to find a set of optimal contracts with similar parameters and then use the `get_forecasting_dataset`
function to load and transform the data for all contracts:

.. literalinclude:: examples/multi_contract.py
   :language: python
   :linenos:

Example (Universe)

When modeling a universe of securities, you can use the `Universe` class to filter
by parameters such as fundamentals (e.g., P/E ratio), volatility, and Fama-French factor exposures. Here's an example:

.. literalinclude:: examples/universe.py
   :language: python
   :linenos:
