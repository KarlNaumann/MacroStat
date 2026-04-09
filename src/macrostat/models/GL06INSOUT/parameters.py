"""
Parameters class for the Godley-Lavoie 2006 INSOUT model (Chapter 7).
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.constraints import LinearConstraint
from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersGL06INSOUT(Parameters):
    """Parameters class for the Godley-Lavoie 2006 INSOUT model.

    Implements the full parameter set for the INSOUT model from Chapter 7
    of Godley & Lavoie (2006), covering 5 sectors: Household, Firm,
    Government, CentralBank, and Bank.
    """

    version = "GL06INSOUT"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters of the Godley-Lavoie 2006 INSOUT model.

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
            # --- Household consumption ---
            "PropensityToConsumeIncome": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\alpha_1",
                "unit": ".",
                "value": 0.95,
            },
            "PropensityToConsumeWealth": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\alpha_2",
                "unit": ".",
                "value": 0.05,
            },
            "AutonomousConsumption": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"\alpha_0",
                "unit": ".",
                "value": 0.0,
            },
            # --- Firm expectations and inventory ---
            "SalesExpectationWeight": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\beta",
                "unit": ".",
                "value": 0.5,
            },
            "InventoryAdjustmentSpeed": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\gamma",
                "unit": ".",
                "value": 0.5,
            },
            "IncomeExpectationWeight": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\varepsilon",
                "unit": ".",
                "value": 0.5,
            },
            # --- Household portfolio ---
            "CashToConsumptionRatio": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\lambda_c",
                "unit": ".",
                "value": 0.1,
            },
            # --- Firm pricing ---
            "MarkupRate": {
                "lower bound": 0.0,
                "upper bound": 2.0,
                "notation": r"\phi",
                "unit": ".",
                "value": 0.1,
            },
            # --- Government ---
            "TaxRate": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\tau",
                "unit": ".",
                "value": 0.25,
            },
            # --- Production ---
            "LaborProductivity": {
                "lower bound": 0.01,
                "upper bound": 10.0,
                "notation": r"pr",
                "unit": "output/worker",
                "value": 1.0,
            },
            "RealGovernmentSpending_param": {
                "lower bound": 0.0,
                "upper bound": 500.0,
                "notation": r"g",
                "unit": "real units",
                "value": 25.0,
            },
            "FullEmployment": {
                "lower bound": 1.0,
                "upper bound": 10000.0,
                "notation": r"N_{fe}",
                "unit": "workers",
                "value": 133.28,
            },
            # --- Inventory ratio ---
            "InventorySalesRatioBaseline": {
                "lower bound": 0.0,
                "upper bound": 2.0,
                "notation": r"\sigma_0",
                "unit": ".",
                "value": 0.3612,
            },
            "InventorySalesRatioInterestSens": {
                "lower bound": 0.0,
                "upper bound": 20.0,
                "notation": r"\sigma_1",
                "unit": ".",
                "value": 3.0,
            },
            # --- Bank reserve ratios ---
            "ReserveRatioM1": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"ro_1",
                "unit": ".",
                "value": 0.1,
            },
            "ReserveRatioM2": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"ro_2",
                "unit": ".",
                "value": 0.1,
            },
            # --- Bank liquidity corridor ---
            "BankLiquidityFloor": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"bot",
                "unit": ".",
                "value": 0.02,
            },
            "BankLiquidityCeiling": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"top",
                "unit": ".",
                "value": 0.06,
            },
            # --- Bank profit corridor ---
            "BankProfitFloor": {
                "lower bound": 0.0,
                "upper bound": 0.1,
                "notation": r"bot_{pm}",
                "unit": ".",
                "value": 0.003,
            },
            "BankProfitCeiling": {
                "lower bound": 0.0,
                "upper bound": 0.1,
                "notation": r"top_{pm}",
                "unit": ".",
                "value": 0.005,
            },
            # --- Deposit rate ---
            "DepositRateBillSensitivity": {
                "lower bound": 0.0,
                "upper bound": 2.0,
                "notation": r"\zeta_b",
                "unit": ".",
                "value": 0.9,
            },
            "DepositRateAdjSpeed": {
                "lower bound": 0.0,
                "upper bound": 0.01,
                "notation": r"\zeta_m",
                "unit": ".",
                "value": 0.0002,
            },
            # --- Loan rate ---
            "LoanRateProfitAdjSpeed": {
                "lower bound": 0.0,
                "upper bound": 0.01,
                "notation": r"\zeta_l",
                "unit": ".",
                "value": 0.0002,
            },
            # --- Wage dynamics ---
            "RealWageTargetConstant": {
                "lower bound": -5.0,
                "upper bound": 0.0,
                "notation": r"\Omega_0",
                "unit": ".",
                "value": -0.32549,
            },
            "RealWageProductivityElasticity": {
                "lower bound": 0.0,
                "upper bound": 3.0,
                "notation": r"\Omega_1",
                "unit": ".",
                "value": 1.0,
            },
            "RealWageEmploymentElasticity": {
                "lower bound": 0.0,
                "upper bound": 5.0,
                "notation": r"\Omega_2",
                "unit": ".",
                "value": 1.5,
            },
            "WageAdjustmentSpeed": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\Omega_3",
                "unit": ".",
                "value": 0.1,
            },
            # --- Portfolio allocation: M1 (row 1) ---
            # M1 lambdas stored for documentation; not used in equations
            # since M1 is the buffer-stock residual.
            "WealthShareM1_Constant": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{10}",
                "unit": ".",
                "value": -0.17071,
            },
            "WealthShareM1_DepositRate": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{12}",
                "unit": ".",
                "value": 0.0,
            },
            "WealthShareM1_BillRate": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{13}",
                "unit": ".",
                "value": 0.0,
            },
            "WealthShareM1_BondYield": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{14}",
                "unit": ".",
                "value": 0.0,
            },
            "WealthShareM1_Income": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{15}",
                "unit": ".",
                "value": 0.18,
            },
            # --- Portfolio allocation: M2 (row 2) ---
            "WealthShareM2_Constant": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{20}",
                "unit": ".",
                "value": 0.52245,
            },
            "WealthShareM2_DepositRate": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{22}",
                "unit": ".",
                "value": 30.0,
            },
            "WealthShareM2_BillRate": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{23}",
                "unit": ".",
                "value": -15.0,
            },
            "WealthShareM2_BondYield": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{24}",
                "unit": ".",
                "value": -15.0,
            },
            "WealthShareM2_Income": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{25}",
                "unit": ".",
                "value": -0.06,
            },
            # --- Portfolio allocation: Bills (row 3) ---
            "WealthShareBills_Constant": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{30}",
                "unit": ".",
                "value": 0.47311,
            },
            "WealthShareBills_DepositRate": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{32}",
                "unit": ".",
                "value": -15.0,
            },
            "WealthShareBills_BillRate": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{33}",
                "unit": ".",
                "value": 30.0,
            },
            "WealthShareBills_BondYield": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{34}",
                "unit": ".",
                "value": -15.0,
            },
            "WealthShareBills_Income": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{35}",
                "unit": ".",
                "value": -0.06,
            },
            # --- Portfolio allocation: Bonds (row 4) ---
            "WealthShareBonds_Constant": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{40}",
                "unit": ".",
                "value": 0.17515,
            },
            "WealthShareBonds_DepositRate": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{42}",
                "unit": ".",
                "value": -15.0,
            },
            "WealthShareBonds_BillRate": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{43}",
                "unit": ".",
                "value": -15.0,
            },
            "WealthShareBonds_BondYield": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{44}",
                "unit": ".",
                "value": 30.0,
            },
            "WealthShareBonds_Income": {
                "lower bound": -50.0,
                "upper bound": 50.0,
                "notation": r"\lambda_{45}",
                "unit": ".",
                "value": -0.06,
            },
        }

    def get_constraints(self) -> tuple[LinearConstraint, ...]:
        """Return adding-up constraints for the Tobin portfolio matrix.

        The GL06INSOUT model has a 4-asset Tobin portfolio allocation
        (M1, M2, Bills, Bonds). The Godley-Lavoie adding-up constraints
        require constant shares to sum to 1, and each rate/income
        sensitivity column to sum to 0. M1 (cash) is the buffer-stock
        residual asset.
        """
        return (
            # Constants: lambda_20 + lambda_30 + lambda_40 + lambda_10 = 1
            LinearConstraint(
                param_names=(
                    "WealthShareM2_Constant",
                    "WealthShareBills_Constant",
                    "WealthShareBonds_Constant",
                    "WealthShareM1_Constant",
                ),
                target=1.0,
            ),
            # Deposit rate sensitivities sum to 0
            LinearConstraint(
                param_names=(
                    "WealthShareM2_DepositRate",
                    "WealthShareBills_DepositRate",
                    "WealthShareBonds_DepositRate",
                    "WealthShareM1_DepositRate",
                ),
                target=0.0,
            ),
            # Bill rate sensitivities sum to 0
            LinearConstraint(
                param_names=(
                    "WealthShareM2_BillRate",
                    "WealthShareBills_BillRate",
                    "WealthShareBonds_BillRate",
                    "WealthShareM1_BillRate",
                ),
                target=0.0,
            ),
            # Bond yield sensitivities sum to 0
            LinearConstraint(
                param_names=(
                    "WealthShareM2_BondYield",
                    "WealthShareBills_BondYield",
                    "WealthShareBonds_BondYield",
                    "WealthShareM1_BondYield",
                ),
                target=0.0,
            ),
            # Income sensitivities sum to 0
            LinearConstraint(
                param_names=(
                    "WealthShareM2_Income",
                    "WealthShareBills_Income",
                    "WealthShareBonds_Income",
                    "WealthShareM1_Income",
                ),
                target=0.0,
            ),
        )

    def get_default_hyperparameters(self):
        """Return the default hyperparameter values."""
        hyperparameters = super().get_default_hyperparameters()
        hyperparameters["timesteps"] = 100
        hyperparameters["timesteps_initialization"] = 0
        hyperparameters["scenario_trigger"] = 50
        hyperparameters["sectors"] = [
            "Household",
            "Firm",
            "Government",
            "CentralBank",
            "Bank",
        ]
        return hyperparameters
