=========
Variables
=========

The LINEAR2D model exposes a single variable via
 :class:`macrostat.models.LINEAR2D.VariablesLINEAR2D`:

- ``State`` – the 2D state vector :math:`x_t` recorded at each timestep.

Internally this is stored as a tensor of shape ``(timesteps, 2)``.


.. csv-table::
   :file: variables.csv
   :header-rows: 1
