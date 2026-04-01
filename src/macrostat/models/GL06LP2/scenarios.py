"""
Scenarios class for the Godley-Lavoie 2006 LP2 model.
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
from macrostat.models.GL06LP2.parameters import ParametersGL06LP2

logger = logging.getLogger(__name__)


class ScenariosGL06LP2(Scenarios):
    """Scenarios class for the Godley-Lavoie 2006 LP2 model.

    In LP2, BondPrice is no longer exogenous. Instead, two shock
    variables are introduced: ExpectedBondPriceShock (add) and
    BondPriceShock (add1).
    """

    version = "GL06LP2"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersGL06LP2 | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of the Godley-Lavoie 2006 LP2 model."""

        if parameters is None:
            parameters = ParametersGL06LP2()

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

        BondPrice is removed (now endogenous). Two shock variables
        are added for expected and actual bond price perturbations.
        """
        sc = {
            "GovernmentDemand": 20,
            "InterestRateBills": 0.03,
            "BondPriceInitial": 20,
            "ExpectedBondPriceShock": 0.0,
            "BondPriceShock": 0.0,
        }

        for k in self.parameters.values.keys():
            sc[f"{k.replace('.', '_')}_add"] = 0.0

        return sc

    def add_default_scenarios(self):
        """Register the default LP2 scenarios from Godley & Lavoie (2006)."""
        # Scenario 1: Increase in the interest rate on bills
        # (tests endogenous bond price response, Figs 5.5-5.6)
        self.add_scenario(
            timeseries={
                "InterestRateBills": 0.04,
            },
            name="Scenario.1: Rise in bill rate",
        )

        # Scenario 2: Anticipated fall in bond prices
        # (households expect bond price to drop, Figs 5.7-5.9)
        self.add_scenario(
            timeseries={
                "ExpectedBondPriceShock": -1.0,
            },
            name="Scenario.2: Expected bond price fall",
        )
