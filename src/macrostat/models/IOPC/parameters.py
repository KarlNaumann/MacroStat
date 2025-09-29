"""
Parameters class for Marco Veronese Passarella's 3IO-PC model
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersIOPC(Parameters):
    """Parameters class for the Godley-Lavoie 2006 PC model."""

    version = "IOPC"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters of Marco Veronese Passarella's 3IO-PC model

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
            "PropensityToConsumeIncomeBase": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\alpha_{10}",
                "unit": ".",
                "value": 0.8,
            },
            "PropensityToConsumeIncomeInterest": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\alpha_{11}",
                "unit": ".",
                "value": 8.0,
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
            "Markup": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\mu",
                "unit": ".",
                "value": 0.875,
            },
            "S1.LabourProductivity": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"pr_1",
                "unit": ".",
                "value": 3.5,
            },
            "S2.LabourProductivity": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"pr_2",
                "unit": ".",
                "value": 5,
            },
            "S3.LabourProductivity": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"pr_3",
                "unit": ".",
                "value": 2.2,
            },
            "S1.HouseholdBudgetShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\bar{\beta}_{C,1}",
                "unit": ".",
                "value": 0.15,
            },
            "S2.HouseholdBudgetShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\bar{\beta}_{C,2}",
                "unit": ".",
                "value": 0.35,
            },
            "S3.HouseholdBudgetShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\bar{\beta}_{C,3}",
                "unit": ".",
                "value": 0.5,
            },
            "S1.GovernmentBudgetShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\bar{\beta}_{G,1}",
                "unit": ".",
                "value": 0.10,
            },
            "S2.GovernmentBudgetShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\bar{\beta}_{G,2}",
                "unit": ".",
                "value": 0.30,
            },
            "S3.GovernmentBudgetShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\bar{\beta}_{G,3}",
                "unit": ".",
                "value": 0.6,
            },
            "S1.S1.Requirement": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"a_{1,1}",
                "unit": ".",
                "value": 0.11,
            },
            "S1.S2.Requirement": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"a_{1,2}",
                "unit": ".",
                "value": 0.12,
            },
            "S1.S3.Requirement": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"a_{1,3}",
                "unit": ".",
                "value": 0.10,
            },
            "S2.S1.Requirement": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"a_{2,1}",
                "unit": ".",
                "value": 0.21,
            },
            "S2.S2.Requirement": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"a_{2,2}",
                "unit": ".",
                "value": 0.22,
            },
            "S2.S3.Requirement": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"a_{2,3}",
                "unit": ".",
                "value": 0.20,
            },
            "S3.S1.Requirement": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"a_{3,1}",
                "unit": ".",
                "value": 0.15,
            },
            "S3.S2.Requirement": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"a_{3,2}",
                "unit": ".",
                "value": 0.18,
            },
            "S3.S3.Requirement": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"a_{3,3}",
                "unit": ".",
                "value": 0.10,
            },
        }

    def get_default_hyperparameters(self):
        """Return the default hyperparameter values."""
        hyperparameters = super().get_default_hyperparameters()
        hyperparameters["timesteps"] = 100
        hyperparameters["timesteps_initialization"] = 1
        hyperparameters["sectors"] = [
            "Household",
            "S1",
            "S2",
            "S3",
            "Government",
            "CentralBank",
        ]
        hyperparameters["iosectors"] = ["S1", "S2", "S3"]
        hyperparameters["vector_sectors"] = ["S1", "S2", "S3"]
        return hyperparameters
