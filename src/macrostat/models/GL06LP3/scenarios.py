"""
Scenarios class for the Godley-Lavoie 2006 LP3 model.
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
from macrostat.models.GL06LP3.parameters import ParametersGL06LP3

logger = logging.getLogger(__name__)


class ScenariosGL06LP3(Scenarios):
    """Scenarios class for the Godley-Lavoie 2006 LP3 model.

    In LP3, government spending becomes endogenous via a fiscal rule.
    GovernmentDemand provides the initial value, and GovernmentSpendingShock
    (add2) allows external perturbations.
    """

    version = "GL06LP3"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersGL06LP3 | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of the Godley-Lavoie 2006 LP3 model."""

        if parameters is None:
            parameters = ParametersGL06LP3()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

        # Add the default named scenarios
        self.add_default_scenarios()

    def get_default_scenario_values(self):
        """Return the default scenario values.

        GovernmentDemand provides the initial value of G. After the
        first step, G evolves endogenously via the fiscal rule.
        GovernmentSpendingShock (add2) allows exogenous perturbations.
        """
        sc = {
            "GovernmentDemand": 20,
            "InterestRateBills": 0.03,
            "BondPriceInitial": 20,
            "ExpectedBondPriceShock": 0.0,
            "BondPriceShock": 0.0,
            "GovernmentSpendingShock": 0.0,
        }

        for k in self.parameters.values.keys():
            sc[f"{k.replace('.', '_')}_add"] = 0.0

        return sc

    def add_default_scenarios(self):
        """Register the default LP3 scenarios from Godley & Lavoie (2006)."""
        # Scenario 1: Drop in propensity to consume out of income
        # (triggers fiscal austerity via deficit rule, Figs 5.11-5.12)
        self.add_scenario(
            timeseries={
                "PropensityToConsumeIncome_add": -0.1,
            },
            name="Scenario.1: Drop in alpha1",
        )

        # Scenario 2: Rise in bill rate (with endogenous G and bond price)
        self.add_scenario(
            timeseries={
                "InterestRateBills": 0.04,
            },
            name="Scenario.2: Rise in bill rate",
        )
