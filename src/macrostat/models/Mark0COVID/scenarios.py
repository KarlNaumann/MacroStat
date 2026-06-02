# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
r"""Scenarios for the Mark-0 COVID model.

Defines the four endogenous-crisis regimes from Gualdi, Tarzia, Zamponi,
Bouchaud (2015), "Tipping points in macroeconomic agent-based models",
plus a baseline scenario. Each regime is a parameter override realised via
the ``Scenarios`` additive-shock channel.

The regimes are obtained by varying :math:`R` (hiring/firing rate) and
:math:`\Theta` (default threshold) holding all other parameters fixed:

- Scenario 0 — Baseline (full employment): :math:`R = 2`, :math:`\Theta = 2`.
- Scenario 1 — Endogenous crises (oscillatory): :math:`R = 0.5`,
  :math:`\Theta = 3`.
- Scenario 2 — Unstable (slow decay to high-U): :math:`R = 1.5`,
  :math:`\Theta = 4`.
- Scenario 3 — Full collapse: :math:`R = 0.5`, :math:`\Theta = 6`.

The exact regime parameterisation is provisional and will be verified
against Gualdi et al. (2015) during Stage 3.
"""

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.Mark0COVID.parameters import ParametersMark0COVID

logger = logging.getLogger(__name__)


class ScenariosMark0COVID(Scenarios):
    """Scenarios class for the Mark-0 COVID model."""

    version = "Mark0COVID"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersMark0COVID | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersMark0COVID()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

        self.add_gualdi_regimes()

    def get_default_scenario_values(self):
        return {
            "HiringFiringRate_add": 0.0,
            "DefaultThreshold_add": 0.0,
        }

    def add_gualdi_regimes(self):
        """Register the three non-baseline Gualdi regimes."""
        self.add_scenario(
            timeseries={
                "HiringFiringRate_add": -1.5,
                "DefaultThreshold_add": 1.0,
            },
            name="Scenario.1: Endogenous crises (R=0.5, Theta=3)",
        )
        self.add_scenario(
            timeseries={
                "HiringFiringRate_add": -0.5,
                "DefaultThreshold_add": 2.0,
            },
            name="Scenario.2: Unstable (R=1.5, Theta=4)",
        )
        self.add_scenario(
            timeseries={
                "HiringFiringRate_add": -1.5,
                "DefaultThreshold_add": 4.0,
            },
            name="Scenario.3: Full collapse (R=0.5, Theta=6)",
        )
