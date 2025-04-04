"""
Parameters class for the Godley-Lavoie 2006 PC model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersGL06PC(Parameters):
    """Parameters class for the Godley-Lavoie 2006 PC model."""

    version = "PC"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters of the Godley-Lavoie 2006 PC model.

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

    def get_default_parameters(self):
        """Return the default parameter values."""
        return {
            "TaxRate": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\theta",
                "unit": "% per period",
                "value": 0.2,
            },
            "PropensityToConsumeIncome": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\alpha_1",
                "unit": ".",
                "value": 0.6,
            },
            "PropensityToConsumeSavings": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\alpha_2",
                "unit": ".",
                "value": 0.4,
            },
            "WealthShareBills_Constant": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\lambda_0",
                "unit": ".",
                "value": 0.635,
            },
            "WealthShareBills_InterestRate": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\lambda_1",
                "unit": ".",
                "value": 5.0,
            },
            "WealthShareBills_Income": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\lambda_2",
                "unit": ".",
                "value": 0.01,
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
