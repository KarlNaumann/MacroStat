"""
Scenarios class for the Godley-Lavoie 2006 LP model.
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
from macrostat.models.GL06LP.parameters import ParametersGL06LP

logger = logging.getLogger(__name__)


class ScenariosGL06LP(Scenarios):
    """Scenarios class for the Godley-Lavoie 2006 LP model."""

    version = "GL06LP"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersGL06LP | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of the Godley-Lavoie 2006 LP model."""

        if parameters is None:
            parameters = ParametersGL06LP()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

        # Add the default named scenarios
        self.add_default_scenarios()

    def get_default_scenario_values(self):
        """Return the default scenario values."""
        sc = {
            "GovernmentDemand": 20,
            "InterestRateBills": 0.03,
            "BondPrice": 20,
        }

        for k in self.parameters.values.keys():
            sc[f"{k.replace('.', '_')}_add"] = 0.0

        return sc

    def add_default_scenarios(self):
        """Register the default LP scenarios from Godley & Lavoie (2006).

        Scenario 1 (Section 5.7): Combined increase in both short-term
        and long-term interest rates. The bill rate rises from 3% to 4%
        while the bond price drops from 20 to 15 (bond yield rises from
        5% to 6.67%).

        Scenario 2 (Section 5.9, LP1 baseline): A sharp decrease in the
        propensity to consume out of current income. This provides the
        comparison baseline for LP3's hysteresis result.
        """
        # Scenario 1: Combined interest rate increase (§5.7, Figs 5.2–5.4)
        self.add_scenario(
            timeseries={
                "InterestRateBills": 0.04,
                "BondPrice": 15,
            },
            name="Scenario.1: Rise in interest rates",
        )

        # Scenario 2: Drop in propensity to consume (§5.9, Fig 5.10)
        self.add_scenario(
            timeseries={
                "PropensityToConsumeIncome_add": -0.1,
            },
            name="Scenario.2: Drop in alpha1",
        )
