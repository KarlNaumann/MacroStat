"""
Parameters class for the Godley-Lavoie 2006 LP2 model.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.models.GL06LP.parameters import ParametersGL06LP

logger = logging.getLogger(__name__)


class ParametersGL06LP2(ParametersGL06LP):
    """Parameters class for the Godley-Lavoie 2006 LP2 model.

    Extends LP parameters with bond-price adjustment and target-proportion
    parameters that make bond prices endogenous.
    """

    version = "GL06LP2"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters of the Godley-Lavoie 2006 LP2 model.

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
        """Return the default parameter values.

        Inherits all GL06LP parameters and adds:
        - ExpectationAdjustmentSpeed (beta_e)
        - BondPriceAdjustmentStep (beta)
        - TargetProportionUpper (top)
        - TargetProportionLower (bot)
        """
        params = super().get_default_parameters()
        params.update(
            {
                "ExpectationAdjustmentSpeed": {
                    "lower bound": 0.0,
                    "upper bound": 1.0,
                    "notation": r"\beta_e",
                    "unit": ".",
                    "value": 0.5,
                },
                "BondPriceAdjustmentStep": {
                    "lower bound": 0.0,
                    "upper bound": 0.1,
                    "notation": r"\beta",
                    "unit": ".",
                    "value": 0.02,
                },
                "TargetProportionUpper": {
                    "lower bound": 0.0,
                    "upper bound": 1.0,
                    "notation": r"\text{top}",
                    "unit": ".",
                    "value": 0.505,
                },
                "TargetProportionLower": {
                    "lower bound": 0.0,
                    "upper bound": 1.0,
                    "notation": r"\text{bot}",
                    "unit": ".",
                    "value": 0.495,
                },
            }
        )
        return params
