================
Behavior
================
.. currentmodule:: macrostat.core.behavior

This class is the base class for all behavior classes. It contains the common methods for all behavior classes.

Constructor
~~~~~~~~~~~
.. autosummary::

   Behavior

Simulation and Behavior
~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Behavior.forward
   Behavior.initialize
   Behavior.step

Differentiable Operations
~~~~~~~~~~~~~~~~~~~~~~~~~
.. autosummary::

   Behavior.diffwhere
   Behavior.tanhmask
   Behavior.diffmin
   Behavior.diffmax
   Behavior.diffmin_v
   Behavior.diffmax_v

Notes
~~~~~
The Behavior class is designed to be a flexible base class that allows users to implement their own model behavior while maintaining a consistent interface. The key methods that users need to implement are `initialize()` and `step()`, which define the model's initialization and simulation behavior respectively.

Example
-------
A typical implementation of a behavior class might look like:

>>> class MyBehavior(Behavior):
...     def initialize(self):
...         # Initialize state variables
...         self.state['x'] = torch.zeros(1)
...
...     def step(self, t, scenario):
...         # Update state variables
...         self.state['x'] = self.state['x'] + 1
