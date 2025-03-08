"""
Parameters class for the Godley-Lavoie 2006 SIM model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersSIM(Parameters):
    """Parameters class for the Godley-Lavoie 2006 SIM model."""

    version = "SIM"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters of the Godley-Lavoie 2006 SIM model.

        Parameters
        ----------
        parameters: dict | None
            The parameters of the model.
        hyperparameters: dict | None
            The hyperparameters of the model.
        bounds: dict | None
            The bounds of the parameters.
        """
        super().__init__(
            parameters=parameters,
            hyperparameters=hyperparameters,
            bounds=bounds,
            *args,
            **kwargs,
        )

    def get_default_hyperparameters(self):
        """Return the default hyperparameters."""
        return {
            "seed": 0,
        }

    def get_default_parameters(self):
        """Return the default parameter values."""
        return {
            "P1": {
                "lower bound": 0.0,
                "upper bound": 100.0,
                "notation": "",
                "unit": "",
                "value": 0.0,
            },
        }
