# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
"""Scenarios class for the Kirman ants SDE model.

Three named regimes for the stationary :math:`\\text{Beta}(\\rho/\\mu, \\rho/\\mu)`
distribution: bimodal (rho/mu < 1, baseline), uniform (rho/mu = 1), unimodal
(rho/mu > 1).
"""

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.KirmansAnts.parameters import ParametersKirmansAnts

logger = logging.getLogger(__name__)


class ScenariosKirmansAnts(Scenarios):
    """Scenarios class for the Kirman ants SDE model."""

    version = "KirmansAnts"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersKirmansAnts | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersKirmansAnts()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

        self.add_three_regimes()

    def get_default_scenario_values(self):
        """Return additive shocks on the two parameters.

        Both default to zero. Named regimes set ``rho_add`` to shift rho to
        the desired value while mu stays at its baseline of 1.0.
        """
        return {
            "rho_add": 0.0,
            "mu_add": 0.0,
        }

    def add_three_regimes(self):
        """Register the three regime presets.

        Baseline (scenario 0) is bimodal at rho=0.5, mu=1.0. The uniform
        and unimodal regimes shift rho via additive shocks.
        """
        # Scenario 1: uniform — rho=1.0 (Beta(1,1)) via rho_add=+0.5.
        self.add_scenario(
            timeseries={
                "rho_add": 0.5,
            },
            name="Scenario.1: Uniform (rho=1.0)",
        )

        # Scenario 2: unimodal — rho=2.0 (Beta(2,2)) via rho_add=+1.5.
        self.add_scenario(
            timeseries={
                "rho_add": 1.5,
            },
            name="Scenario.2: Unimodal (rho=2.0)",
        )
