"""
Scenarios class for the Godley-Lavoie 2006 INSOUT model (Chapter 10).
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.GL06INSOUT.parameters import ParametersGL06INSOUT

logger = logging.getLogger(__name__)


class ScenariosGL06INSOUT(Scenarios):
    """Scenarios class for the Godley-Lavoie 2006 INSOUT model."""

    version = "GL06INSOUT"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersGL06INSOUT | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of the Godley-Lavoie 2006 INSOUT model."""

        if parameters is None:
            parameters = ParametersGL06INSOUT()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

        self.add_default_scenarios()

    def get_default_scenario_values(self):
        """Return the default scenario values."""
        sc = {
            "RealGovernmentSpending": 25,
            "BillRate": 0.023,
            "BondYield": 0.027,
        }

        for k in self.parameters.values.keys():
            sc[f"{k.replace('.', '_')}_add"] = 0.0

        return sc

    def add_default_scenarios(self):
        """Register the 7 default INSOUT scenarios from Godley & Lavoie (2006, Ch. 7).

        All shocks trigger at period 50 (scenario_trigger=50).

        Scenario 1: Higher target inventory ratio (sigma_0: 0.3612 -> 0.4)
        Scenario 2: Higher government spending (g: 25 -> 30)
        Scenario 3: Higher reserve requirements (ro_1, ro_2: +0.1 each)
        Scenario 4: Wider liquidity corridor (top, bot: +0.18 each)
        Scenario 5: Lower consumption propensity (alpha_1: 0.95 -> 0.8)
        Scenario 6: Higher real wage target (Omega_0: -0.32549 -> -0.2)
        Scenario 7: Combined wage target rise + interest rate rise
        """
        # Scenario 1: Higher target inventory-sales ratio
        self.add_scenario(
            timeseries={
                "InventorySalesRatioBaseline_add": 0.0388,
            },
            name="Scenario.1: Higher target inventory ratio",
        )

        # Scenario 2: Higher government spending
        self.add_scenario(
            timeseries={
                "RealGovernmentSpending": 30,
            },
            name="Scenario.2: Higher government spending",
        )

        # Scenario 3: Higher reserve requirements
        self.add_scenario(
            timeseries={
                "ReserveRatioM1_add": 0.1,
                "ReserveRatioM2_add": 0.1,
            },
            name="Scenario.3: Higher reserve requirements",
        )

        # Scenario 4: Wider liquidity corridor
        self.add_scenario(
            timeseries={
                "BankLiquidityCeiling_add": 0.18,
                "BankLiquidityFloor_add": 0.18,
            },
            name="Scenario.4: Wider liquidity corridor",
        )

        # Scenario 5: Lower consumption propensity out of income
        self.add_scenario(
            timeseries={
                "PropensityToConsumeIncome_add": -0.15,
            },
            name="Scenario.5: Lower consumption propensity",
        )

        # Scenario 6: Higher real wage target constant
        self.add_scenario(
            timeseries={
                "RealWageTargetConstant_add": 0.12549,
            },
            name="Scenario.6: Higher real wage target",
        )

        # Scenario 7: Combined wage target rise and interest rate rise
        self.add_scenario(
            timeseries={
                "RealWageTargetConstant_add": 0.12549,
                "BillRate": 0.03,
                "BondYield": 0.039,
            },
            name="Scenario.7: Wage target + rate rise",
        )
