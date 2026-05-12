Kirman Ants SDE
===============

Equations
---------
- SDE: :math:`dx_t = \rho\,(1 - 2 x_t)\, dt + \sqrt{2 \mu\, x_t (1 - x_t)}\, dW_t`
- Stationary distribution: :math:`x_\infty \sim \text{Beta}(\rho/\mu,\, \rho/\mu)` on :math:`(0, 1)`.
- Regime: unimodal at :math:`x = 1/2` for :math:`\rho/\mu > 1`, bimodal with mass at the boundaries for :math:`\rho/\mu < 1`.

API
---
.. code-block:: python

   from macrostat.models.KirmansAnts import (
       KirmansAnts,
       ParametersKirmansAnts,
       VariablesKirmansAnts,
       ScenariosKirmansAnts,
   )

   # Configure horizon. record_inner=True allocates a float32 buffer of
   # size timesteps * substeps for the full micro-step trajectory.
   params = ParametersKirmansAnts(
       hyperparameters={"timesteps": 1000, "record_inner": True}
   )
   variables = VariablesKirmansAnts(parameters=params)
   scenarios = ScenariosKirmansAnts(parameters=params)
   model = KirmansAnts(parameters=params, variables=variables, scenarios=scenarios)

   # Baseline (bimodal) simulation
   model.simulate()

   # Built-in regimes
   model.simulate(scenario="Scenario.1: Uniform (rho=1.0)")
   model.simulate(scenario="Scenario.2: Unimodal (rho=2.0)")

Series
------
The model records one end-of-macro-period value per outer step in
``model.variables.timeseries["density"]`` (shape ``(timesteps, 1)``). With
``record_inner=True``, the full micro-step trajectory (length
``timesteps * substeps``, dtype ``float32``) is available on the behavior
instance after a simulation run::

   model.simulate()
   trajectory = model.behavior_instance._micro_trajectory
   exhaustions = model.behavior_instance.exhaustion_count

The exhaustion counter reports the number of micro-steps where rejection
sampling hit ``max_attempts``; on exhaustion the increment is set to zero
and a warning is logged.

Differentiability
-----------------
The numpy Euler-Maruyama loop and rejection sampling are non-differentiable.
``BehaviorKirmansAnts.supports_differentiable = False`` so constructing the
behavior with ``differentiable=True`` raises ``RuntimeError``.

Source
------
Reference implementation cloned from
``packages/abmstat/abmstat/models/kirmanants.py``. Original model:
Kirman, A. (1993). "Ants, Rationality, and Recruitment".
*The Quarterly Journal of Economics*, 108(1), 137-156. Continuous-time
limit per Moran et al. (2020).
