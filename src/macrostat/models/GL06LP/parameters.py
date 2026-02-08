"""
Parameters class for the Godley-Lavoie 2006 LP model.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersGL06LP(Parameters):
    """Parameters class for the Godley-Lavoie 2006 LP model."""

    version = "GL06LP"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters of the Godley-Lavoie 2006 LP model.

        Parameters
        ----------
        parameters : dict | None
            The parameters of the model.
        hyperparameters : dict | None
            The hyperparameters of the model.
        bounds : dict | None
            The bounds of the parameters.
        """
        super().__init__(
            parameters=parameters,
            hyperparameters=hyperparameters,
            bounds=bounds,
            *args,
            **kwargs,
        )

    def get_default_parameters(self):
        """Return the default parameter values."""
        return {
            "TaxRate": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\theta",
                "unit": "% per period",
                "value": 0.1938,
            },
            "PropensityToConsumeIncome": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\alpha_1",
                "unit": ".",
                "value": 0.8,
            },
            "PropensityToConsumeSavings": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\alpha_2",
                "unit": ".",
                "value": 0.2,
            },
            "ExpectationWeightBondPrice": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\chi",
                "unit": ".",
                "value": 0.1,
            },
            # Portfolio allocation: Bills (row 2)
            "WealthShareBills_Constant": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\lambda_{20}",
                "unit": ".",
                "value": 0.44196,
            },
            "WealthShareBills_BillRate": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"\lambda_{22}",
                "unit": ".",
                "value": 1.1,
            },
            "WealthShareBills_BondReturn": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"\lambda_{23}",
                "unit": ".",
                "value": -1.0,
            },
            "WealthShareBills_Income": {
                "lower bound": -1.0,
                "upper bound": 1.0,
                "notation": r"\lambda_{24}",
                "unit": ".",
                "value": -0.03,
            },
            # Portfolio allocation: Bonds (row 3)
            "WealthShareBonds_Constant": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\lambda_{30}",
                "unit": ".",
                "value": 0.3997,
            },
            "WealthShareBonds_BillRate": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"\lambda_{32}",
                "unit": ".",
                "value": -1.0,
            },
            "WealthShareBonds_BondReturn": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"\lambda_{33}",
                "unit": ".",
                "value": 1.1,
            },
            "WealthShareBonds_Income": {
                "lower bound": -1.0,
                "upper bound": 1.0,
                "notation": r"\lambda_{34}",
                "unit": ".",
                "value": -0.03,
            },
        }

    def get_default_hyperparameters(self):
        """Return the default hyperparameter values."""
        hyperparameters = super().get_default_hyperparameters()
        hyperparameters["timesteps"] = 100
        hyperparameters["timesteps_initialization"] = 1
        hyperparameters["sectors"] = [
            "Household",
            "Production",
            "Government",
            "CentralBank",
        ]
        return hyperparameters
