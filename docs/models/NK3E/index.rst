New Keynesian 3-Equation Model
==============================

Equations
---------
- IS: ``y_t = A_t - a1 * r_{t-1}``
- Phillips: ``pi_t = pi_{t-1} + a2 * (y_t - yE_t)``
- Policy: ``rs_t = (A_t - yE_t) / a1`` and ``r_t = rs_t + a3 * (pi_t - piT_t)``,
  where ``a3 = 1 / [a1 * (1/(a2*b) + a2)]``.

API
---
.. code-block:: python

   from macrostat.models.NK3E import (
       NK3E,
       ParametersNK3E,
       VariablesNK3E,
       ScenariosNK3E,
   )

   # Configure horizon and (optional) progress bar
   params = ParametersNK3E(hyperparameters={"timesteps": 50, "timesteps_initialization": 1, "use_tqdm": False})
   variables = VariablesNK3E(parameters=params)
   scenarios = ScenariosNK3E(parameters=params)
   model = NK3E(parameters=params, variables=variables, scenarios=scenarios)

   # Baseline simulation
   model.simulate()

   # Scenario examples (provided out-of-the-box)
   model.simulate(scenario="Scenario.1: Rise in A")      # demand up
   model.simulate(scenario="Scenario.2: Higher pi_T")    # higher inflation target
   model.simulate(scenario="Scenario.3: Rise in y_e")    # higher potential output

Series
------
The model records time series in ``model.variables.timeseries`` with keys:

- ``y``: output
- ``pi``: inflation
- ``r``: real interest rate
- ``r_s``: stabilizing real interest rate ``(A - y_e)/a1``

Each series has length equal to the configured ``timesteps``. Baseline parameters
``A``, ``pi_T``, and ``y_e`` are available via ``model.parameters``; scenario shocks
are managed by ``ScenariosNK3E``.

Source
------
The reduced-form and parameterizations follow the online note:
``https://macrosimulation.org/a_new_keynesian_3_equation_model``.
