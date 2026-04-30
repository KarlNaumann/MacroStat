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

   params = ParametersPichlerEtAl2022DIO(
       hyperparameters={"n_sectors": 3, "timesteps": 60, "production_function": "leontief"}
   )
   variables = VariablesPichlerEtAl2022DIO(parameters=params)
   scenarios = ScenariosPichlerEtAl2022DIO(parameters=params)
   model = PichlerEtAl2022DIO(parameters=params, variables=variables, scenarios=scenarios)

   model.simulate()

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
