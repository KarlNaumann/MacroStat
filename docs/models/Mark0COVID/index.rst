Mark-0 COVID Heterogeneous-Agent ABM
====================================

Mark-0 is a closed-economy ABM with :math:`N` firms, one representative
household, one commercial bank, and one central bank. Each macro period
runs an ordered 24-phase loop (price/wage adjustment, hiring/firing,
production, household consumption, interest-rate setting, bankruptcy
and firm revival). The Sharma et al. (2021) COVID extension adds the
parameter set used to study the V-, U-, L-, and W-shaped recovery
regimes.

References
----------
- Bouchaud, J.-P., Gualdi, S., Tarzia, M., Zamponi, F. (2018). "Optimal
  inflation target: insights from an agent-based model". *Economics*,
  12 (15).
- Sharma, D., Bouchaud, J.-P., Gualdi, S., Tarzia, M., Zamponi, F.
  (2021). "V-, U-, L-, or W-shaped economic recovery after COVID-19:
  insights from an agent-based model." *PLOS ONE* 16(3), e0247823.
- Gualdi, S., Tarzia, M., Zamponi, F., Bouchaud, J.-P. (2015).
  "Tipping points in macroeconomic agent-based models". *JEDC* 50,
  29-61.

API
---
.. code-block:: python

   from macrostat.models.Mark0COVID import (
       Mark0COVID,
       ParametersMark0COVID,
       VariablesMark0COVID,
       ScenariosMark0COVID,
   )

   params = ParametersMark0COVID(
       hyperparameters={"timesteps": 100, "N_firms": 1000, "seed": 0}
   )
   model = Mark0COVID(parameters=params)

   # Baseline (full-employment) simulation
   model.simulate()

   # Built-in Gualdi regimes
   model.simulate(scenario=1)  # endogenous crises
   model.simulate(scenario=2)  # unstable
   model.simulate(scenario=3)  # full collapse

Differentiability
-----------------
The forward pass is differentiable end-to-end. Parameter-dependent
boundaries are smoothed with the inherited ``Behavior.diffwhere`` (two
sites: bankruptcy ``stay_alive`` indicator and the positive-:math:`Y`
guard on the survivor-recompute path). All other gradient-irrelevant
branches use plain ``torch.where``, matching the abmstat reference
pattern. Stochastic phases (price/wage/revival draws) are
reparameterised through three independent frozen ``U(0, 1)`` noise
buffers of shape ``(timesteps, N_firms)`` pre-drawn in
``initialize()``.

Hyperparameter ``dtype`` defaults to ``torch.float64`` to match the
abmstat reference, the slow KS gate
(:mod:`tests.models.mark0covid_test`), and to keep aggregate trajectory
divergence under 2 % for prices/wages over a 50-period run.

Source
------
Port of the reference implementation at
``packages/abmstat/abmstat/models/mark0_torch.py`` into the MacroStat
6-file convention. The 24-phase forward loop and the three
ordering-critical aggregation passes are decomposed into the methods
listed under :doc:`equations <equations>`.

.. toctree::
    :maxdepth: 2

    Notebook <Mark0COVID.ipynb>
    Variables <variables.rst>
    Parameters <parameters.rst>
    Equations <equations.rst>
    Scenarios <scenarios.rst>
