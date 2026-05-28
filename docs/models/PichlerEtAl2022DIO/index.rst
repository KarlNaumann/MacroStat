==================================
Pichler et al. (2022) Dynamic IO
==================================

Dynamic input-output model from Pichler, Pangallo, del Rio-Chanona, Lafond, and
Farmer (2022), *Forecasting the propagation of pandemic shocks with a dynamic
input-output model*, Journal of Economic Dynamics & Control.

The model couples a sectoral input-output network to a partial-equilibrium
demand block. Each industry chooses output as the minimum of labour capacity,
input-based capacity (parameterised by one of five production functions), and
realised demand. Inventories evolve from deliveries and technical input usage;
households adjust consumption from labour income with a propensity-to-consume
that depends on aggregate-shock state.

Default calibration: illustrative, not empirical
------------------------------------------------

The zero-config default is a 3-sector pedagogical economy (sectors ``A``,
``B``, ``C``) with hand-chosen technical coefficients designed to exhibit
production-network amplification under essential-input shocks. It is not
a calibration to any real economy and should not be cited as such. The
spectral radius of the default ``TechnicalCoefficients`` matrix is
approximately ``0.38``; initial output and accounting identities close
exactly at the default state.

To reproduce the paper's UK 2014 results, use the
``ParametersPichlerEtAl2022DIO.from_wiod_uk`` classmethod with
user-supplied WIOD 2016-release tables and the IHS Markit critical-input
matrix from the original paper's replication archive. Neither dataset is
redistributed with this package; users must obtain them under their own
licence terms.

Sub-pages
---------

.. toctree::
   :maxdepth: 1

   notation
   PichlerEtAl2022DIO.ipynb

API
---
.. code-block:: python

   from macrostat.models.PichlerEtAl2022DIO import (
       PichlerEtAl2022DIO,
       ParametersPichlerEtAl2022DIO,
       VariablesPichlerEtAl2022DIO,
       ScenariosPichlerEtAl2022DIO,
   )

   # Zero-config: 3-sector illustrative default.
   model = PichlerEtAl2022DIO()
   result = model.simulate()

   # Custom configuration: override hyperparameters as needed.
   params = ParametersPichlerEtAl2022DIO(
       hyperparameters={"production_function": "leontief"}
   )
   custom = PichlerEtAl2022DIO(parameters=params)
   custom.simulate()

Production-function variants are selected via the
``production_function`` hyperparameter:

- ``leontief`` — all inputs with positive technical coefficient bind.
- ``strongly_critical`` — critical and important inputs bind.
- ``half_critical`` — critical inputs bind; important inputs at half capacity.
- ``weakly_critical`` — only critical inputs bind.
- ``linear`` — perfect substitution across inputs.

Source
------
Pichler, A., Pangallo, M., del Rio-Chanona, R. M., Lafond, F. & Farmer, J. D.
(2022). *Forecasting the propagation of pandemic shocks with a dynamic
input-output model*. Journal of Economic Dynamics & Control, 144, 104527.
