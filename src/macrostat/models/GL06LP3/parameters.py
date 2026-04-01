"""
Parameters class for the Godley-Lavoie 2006 LP3 model.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.models.GL06LP2.parameters import ParametersGL06LP2 as _ParametersGL06LP2

logger = logging.getLogger(__name__)


class ParametersGL06LP3(_ParametersGL06LP2):
    """Parameters class for the Godley-Lavoie 2006 LP3 model.

    Extends LP2 parameters with fiscal-rule parameters that make
    government expenditures endogenous.
    """

    version = "GL06LP3"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters of the Godley-Lavoie 2006 LP3 model.

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

        Inherits all GL06LP2 parameters and adds:
        - FiscalAdjustmentSpeed (beta_g)
        - PSBRThreshold (deficit-to-GDP trigger)
        """
        params = super().get_default_parameters()
        params.update(
            {
                "FiscalAdjustmentSpeed": {
                    "lower bound": 0.0,
                    "upper bound": 1.0,
                    "notation": r"\beta_g",
                    "unit": ".",
                    "value": 0.01,
                },
                "PSBRThreshold": {
                    "lower bound": 0.0,
                    "upper bound": 1.0,
                    "notation": r"\text{threshold}",
                    "unit": ".",
                    "value": 0.03,
                },
            }
        )
        return params
