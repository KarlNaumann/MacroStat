================
Model
================
.. currentmodule:: macrostat.core.model

This class is the base class for all model classes. It contains the common methods for all model classes.

Constructor
~~~~~~~~~~~
.. autosummary::

   Model


Initialization and Loading
~~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Model.from_json
   Model.load

Simulation and Behavior
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Model.simulate

Serialization / IO
~~~~~~~~~~~~~~~~~~
.. autosummary::

   Model.save
   Model.to_json

Notes
~~~~~
The Model class is designed to be a flexible wrapper that allows users to implement their own model behavior while maintaining a consistent interface. The key method that users need to implement is `simulate()`, which should return a pandas DataFrame containing the simulation results.

Example
-------
A general workflow for a model might look like:

>>> model = Model(parameters, hyperparameters)
>>> output = model.simulate()
>>> model.save()
