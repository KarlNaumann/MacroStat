=================
Variables
=================
.. currentmodule:: macrostat.core.variables

This class is the base class for all variable classes. It contains the common methods for all variable classes.

Constructor
~~~~~~~~~~~
.. autosummary::

   Variables

Initialization and Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Variables.from_json
   Variables.from_excel

Variable Management
~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Variables.get_stock_variables
   Variables.get_flow_variables
   Variables.get_index_variables
   Variables.initialize_tensors
   Variables.new_state
   Variables.update_history
   Variables.record_state
   Variables.verify_sfc_info

Accounting Functions
~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Variables.balance_sheet_theoretical
   Variables.balance_sheet_actual
   Variables.transaction_matrix_theoretical
   Variables.transaction_matrix_actual
   Variables.compare

Serialization / IO
~~~~~~~~~~~~~~~~~~
.. autosummary::

   Variables.to_json
   Variables.to_excel
   Variables.to_pandas
   Variables.info_to_csv

Notes
~~~~~
The Variables class manages the variables of a MacroStat model, including the output tensors from simulations. It provides methods for handling variable characteristics (dimension, name, unit, description, notation) and includes accounting functions for balance sheets and transaction matrices.

Example
-------
A typical workflow for variables might look like:

>>> variables = Variables(parameters)
>>> variables.initialize_tensors(100)
>>> variables.record_state(0, {'x': torch.tensor([1.0])})
>>> variables.to_json('variables.json')
