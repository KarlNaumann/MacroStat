==========
Parameters
==========

The 2D linear testing model :class:`macrostat.models.LINEAR2D.LINEAR2D` has the
following parameters:

- ``a11``, ``a12``, ``a21``, ``a22`` – entries of the 2x2 transition matrix
  :math:`A` in :math:`x_{t+1} = A x_t`.
- ``x0_1``, ``x0_2`` – entries of the initial state :math:`x_0`.

All parameters are scalars with broad bounds (default :math:`[-10, 10]`).

Hyperparameters
===============

.. csv-table::
   :file: hyperparameters.csv
   :header-rows: 1


Parameters
==========

.. csv-table::
   :file: parameters.csv
   :header-rows: 1
