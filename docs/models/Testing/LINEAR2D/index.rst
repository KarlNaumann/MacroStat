========
LINEAR2D
========

.. toctree::
   :maxdepth: 1

   ../LINEAR2D_parameters.rst
   ../LINEAR2D_scenarios.rst
   ../LINEAR2D_variables.rst


The main testing model is :class:`macrostat.models.LINEAR2D.LINEAR2D`, a 2D
linear system with dynamics

.. math::

   x_{t+1} = A x_t,

where :math:`x_t` is a 2D state vector and :math:`A` is a fully parameterised
2x2 matrix. This model has an analytical Jacobian, so it is used in the
``tests/diff/diff_test.py`` unit tests and in the
``offline/macrostat_testing/calib_example.py`` script to validate the
implementation of :mod:`macrostat.diff`.
