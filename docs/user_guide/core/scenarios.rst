================
Scenarios
================
.. currentmodule:: macrostat.core.scenarios

This class is the base class for all scenario classes. It contains the common methods for all scenario classes.

Constructor
~~~~~~~~~~~
.. autosummary::

   Scenarios

Initialization and Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Scenarios.from_json
   Scenarios.from_excel

Scenario Management
~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Scenarios.add_scenario
   Scenarios.get_default_scenario
   Scenarios.get_default_scenario_values
   Scenarios.get_scenario_index
   Scenarios.verify_scenario_info

Serialization / IO
~~~~~~~~~~~~~~~~~~
.. autosummary::

   Scenarios.to_json
   Scenarios.to_excel

Notes
~~~~~
The Scenarios class provides a uniform interface for handling scenarios, particularly for exogenous shocks. It supports both user-specified scenarios and calibration scenarios, and includes methods for loading and saving scenarios in various formats.

Example
-------
A typical workflow for scenarios might look like:

>>> scenarios = Scenarios(parameters)
>>> scenarios.add_scenario({'shock': 0.1}, name='shock_scenario')
>>> scenarios.to_json('scenarios.json')
