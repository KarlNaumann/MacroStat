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
        """Register the default LP scenarios from Godley & Lavoie (2006)."""
        # Scenario 1: Increase in the interest rate on bills
        self.add_scenario(
            timeseries={
                "InterestRateBills": 0.035,
            },
            name="Scenario.1: Rise in bill rate",
        )

        # Scenario 2: Increase in government expenditures
        self.add_scenario(
            timeseries={
                "GovernmentDemand": 25,
            },
            name="Scenario.2: Rise in G",
        )
