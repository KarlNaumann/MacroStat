================
Parameters
================
.. currentmodule:: macrostat.core.parameters

This class is the base class for all parameter classes. It contains the common methods for all parameter classes.

Constructor
~~~~~~~~~~~
.. autosummary::

   Parameters

Initialization and Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Parameters.from_json
   Parameters.from_excel
   Parameters.from_csv

Serialization / IO
~~~~~~~~~~~~~~~~~~
.. autosummary::

   Parameters.to_json
   Parameters.to_excel
   Parameters.to_csv

Parameter Management
~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Parameters.set_bound
   Parameters.set_notation
   Parameters.set_unit
   Parameters.verify_bounds
   Parameters.verify_parameters

Notes
~~~~~
The Parameters class is designed to handle both model parameters and hyperparameters in a structured way. It provides methods for loading parameters from various file formats (JSON, Excel, CSV) and includes validation to ensure parameters are within specified bounds.

Example
-------
A typical workflow for parameters might look like:

>>> params = Parameters()
>>> params['alpha'] = 0.5
>>> params.to_json('parameters.json')
