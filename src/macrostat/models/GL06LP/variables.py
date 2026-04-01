"""
Variables class for the Godley-Lavoie 2006 LP model.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import numpy as np

from macrostat.core.variables import Variables
from macrostat.models.GL06LP.parameters import ParametersGL06LP

logger = logging.getLogger(__name__)


class VariablesGL06LP(Variables):
    """Variables class for the Godley-Lavoie 2006 LP model."""

    version = "GL06LP"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersGL06LP | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables of the Godley-Lavoie 2006 LP model."""

        if parameters is None:
            parameters = ParametersGL06LP()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def check_health(self, tolerance: float = 1e-4):  # pragma: no cover
        r"""Check the health of the variables by verifying that the redundant
        equations hold and that all the assets and liabilities are positive.

        Parameters
        ----------
        tolerance : float, optional
            The tolerance for the checks, by default 1e-4

        Returns
        -------
        bool
            True if the variables are healthy, False otherwise.
        """
        output = self.to_pandas()

        # Redundant equation: H_h = H_s
        diff = output["HouseholdCashStock"] - output["CentralBankMoneyStock"]
        ape = diff.div(output["HouseholdCashStock"].replace(0, np.nan)).abs()
        if np.any(ape > tolerance):
            logger.warning(
                f"Household cash stock != central bank money stock: "
                f"{ape[ape > tolerance]}"
            )
            return False

        # Check that all assets and liabilities are positive
        stocks = [
            k
            for k, v in self.info.items()
            if v["sfc"][0][0].lower() in ["asset", "liability"]
        ]
        for stock in stocks:
            if np.any(output[stock] < 0):
                logger.warning(f"{stock} is negative")
                return False

        return True

    def get_default_variables(self):
        """Return the default variables information dictionary."""
        return {
            # ---
            # Flow variables
            # ---
            "ConsumptionHousehold": {
                "notation": r"C(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Inflow", "Production"), ("Outflow", "Household")],
            },
            "ConsumptionGovernment": {
                "notation": r"G(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Inflow", "Production"), ("Outflow", "Government")],
            },
            "NationalIncome": {
                "notation": r"Y(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Outflow", "Production"), ("Inflow", "Household")],
            },
            "Taxes": {
                "notation": r"T(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Outflow", "Household"), ("Inflow", "Government")],
            },
            "InterestOnBillsHousehold": {
                "notation": r"r_b(t-1) \cdot B_h(t-1)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Inflow", "Household"), ("Outflow", "Government")],
            },
            "BondCouponIncomeHousehold": {
                "notation": r"BL_h(t-1)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Inflow", "Household"), ("Outflow", "Government")],
            },
            "CentralBankProfits": {
                "notation": r"r_b(t-1) \cdot B_{CB}(t-1)",
                "unit": "USD",
                "history": 0,
                "sectors": ["CentralBank"],
                "sfc": [("Inflow", "Government"), ("Outflow", "CentralBank")],
            },
            "CapitalGains": {
                "notation": r"CG(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "ExpectedCapitalGains": {
                "notation": r"CG^e(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            # ---
            # Stock variables
            # ---
            "Wealth": {
                "notation": r"V(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Liability", "Household"), ("Asset", "Government")],
            },
            "HouseholdBillStock": {
                "notation": r"B_h(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "GovernmentBillStock": {
                "notation": r"B_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Liability", "Government")],
            },
            "CentralBankBillStock": {
                "notation": r"B_{CB}(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["CentralBank"],
                "sfc": [("Asset", ["CentralBank", "Capital"])],
            },
            "HouseholdBondStock": {
                "notation": r"BL_h(t)",
                "unit": "bonds",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "GovernmentBondSupply": {
                "notation": r"BL_s(t)",
                "unit": "bonds",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Liability", "Government")],
            },
            "HouseholdCashStock": {
                "notation": r"H_h(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "CentralBankMoneyStock": {
                "notation": r"H_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["CentralBank"],
                "sfc": [("Liability", ["CentralBank", "Capital"])],
            },
            # ---
            # Index / intermediate variables
            # ---
            "DisposableIncome": {
                "notation": r"YD_r(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "ExpectedDisposableIncome": {
                "notation": r"YD_r^e(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "ExpectedWealth": {
                "notation": r"V^e(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "HouseholdBillDemand": {
                "notation": r"B_d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "HouseholdBondDemand": {
                "notation": r"BL_d(t)",
                "unit": "bonds",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "HouseholdCashDemand": {
                "notation": r"H_d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "InterestRateBills": {
                "notation": r"r_b(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "BondPrice": {
                "notation": r"p_{bl}(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "BondYield": {
                "notation": r"r_{bl}(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "ExpectedBondPrice": {
                "notation": r"p_{bl}^e(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "ExpectedReturnOnBonds": {
                "notation": r"ERr_{bl}(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
        }
