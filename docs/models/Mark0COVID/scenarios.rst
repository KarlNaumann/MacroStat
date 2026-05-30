====================
Scenarios Mark0COVID
====================

Default scenarios reproduce the four endogenous-crisis regimes from
Gualdi, Tarzia, Zamponi, Bouchaud (2015), "Tipping points in
macroeconomic agent-based models", obtained by varying the
hiring/firing rate :math:`R` and default threshold :math:`\Theta`
through the additive-shock channel:

- Scenario.0 (baseline): Full employment — :math:`R = 2,\, \Theta = 2`
  (no shocks).
- Scenario.1: Endogenous crises (oscillatory) — :math:`R = 0.5,\,
  \Theta = 3` via ``HiringFiringRate_add = -1.5,
  DefaultThreshold_add = +1.0``.
- Scenario.2: Unstable (slow decay to high-U) — :math:`R = 1.5,\,
  \Theta = 4` via ``HiringFiringRate_add = -0.5,
  DefaultThreshold_add = +2.0``.
- Scenario.3: Full collapse — :math:`R = 0.5,\, \Theta = 6` via
  ``HiringFiringRate_add = -1.5, DefaultThreshold_add = +4.0``.

All scenarios are implemented in
``macrostat.models.Mark0COVID.scenarios.ScenariosMark0COVID``.
