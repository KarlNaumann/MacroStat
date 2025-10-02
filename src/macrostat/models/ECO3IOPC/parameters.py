"""
Parameters class for Marco Veronese Passarella's ECO-3IO-PC model
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersECO3IOPC(Parameters):
    """Parameters class for the ECO-3IO-PC model."""

    version = "ECO3IOPC"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters of Marco Veronese Passarella's ECO-3IO-PC model

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
            "PropensityToConsumeIncomeTemperature": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\alpha_{12}",
                "unit": ".",
                "value": 0.0,
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
            # Ecosystem Parameters
            "S1.MaterialIntensity": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"m_{mat,1}",
                "unit": ".",
                "value": 0.40,
            },
            "S2.MaterialIntensity": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"m_{mat,2}",
                "unit": ".",
                "value": 0.70,
            },
            "S3.MaterialIntensity": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"m_{mat,3}",
                "unit": ".",
                "value": 0.30,
            },
            "S1.DiscardedStockShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\zeta_{1}",
                "unit": ".",
                "value": 0.025,
            },
            "S2.DiscardedStockShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\zeta_{2}",
                "unit": ".",
                "value": 0.035,
            },
            "S3.DiscardedStockShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\zeta_{3}",
                "unit": ".",
                "value": 0.020,
            },
            "S1.EnergyIntensity": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\epsilon_{en,1}",
                "unit": ".",
                "value": 8,
            },
            "S2.EnergyIntensity": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\epsilon_{en,2}",
                "unit": ".",
                "value": 6,
            },
            "S3.EnergyIntensity": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"\epsilon_{en,3}",
                "unit": ".",
                "value": 3,
            },
            "S1.RenewableEnergyUseShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\eta_{en,1}",
                "unit": ".",
                "value": 0.12,
            },
            "S2.RenewableEnergyUseShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\eta_{en,2}",
                "unit": ".",
                "value": 0.15,
            },
            "S3.RenewableEnergyUseShare": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\eta_{en,3}",
                "unit": ".",
                "value": 0.18,
            },
            "RecyclingRate": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\rho_{dis}",
                "unit": ".",
                "value": 0.20,
            },
            "CO2IntensityNonRenewableEnergyGrowth": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\Delta_\%\beta_e",
                "unit": ".",
                "value": 0.00,
            },
            "TransientClimateResponseCumCO2": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"tcre",
                "unit": ".",
                "value": 1.65 / 1000,
            },
            "NonCO2AnthropocentricForcing": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"fnc",
                "unit": ".",
                "value": 0.34,
            },
            "MatterToResourceConversionRate": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\sigma_m",
                "unit": ".",
                "value": 0.0005,
            },
            "CarbonToCO2Conversion": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"car",
                "unit": ".",
                "value": 3.67,
            },
            "EnergyToResourceConversionRate": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\sigma_e",
                "unit": ".",
                "value": 0.003,
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
