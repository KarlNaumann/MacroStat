"""
Variables class for the Godley-Lavoie 2006 PC model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import numpy as np

from macrostat.core.variables import Variables
from macrostat.models.GL06PC.parameters import ParametersGL06PC

logger = logging.getLogger(__name__)


class VariablesGL06PC(Variables):
    """Variables class for the Godley-Lavoie 2006 PC model."""

    version = "GL06PC"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersGL06PC | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables of the Godley-Lavoie 2006 PC model."""

        if parameters is None:
            parameters = ParametersGL06PC()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def check_health(self, tolerance: float = 1e-4):
        r"""Check the health of the variables by verifying that the redundant equations hold
        and that all the assets and liabilities are positive.

        Parameters
        ----------
        tolerance: float (default: 1e-4)
            The tolerance for the checks.

        Equations
        ---------
        Redundant equations:
            .. math::
                :nowrap:

                \begin{align}
                    H_h(t) = H_s(t)
                \end{align}

        General checks:
            .. math::
                :nowrap:

                \begin{align}
                    A(t) &> 0 & L(t) &> 0
                \end{align}

        where :math:`A(t)` are all assets and :math:`L(t)` are all liabilities.

        Returns
        -------
        bool
            True if the variables are healthy, False otherwise.
        """

        output = self.to_pandas()

        # Redundant equations
        # 1. Household money stock (H_h) = central bank money stock (H_s)
        diff = output["HouseholdMoneyStock"] - output["CentralBankMoneyStock"]
        ape = diff.div(output["HouseholdMoneyStock"]).abs()
        if np.any(ape > tolerance):
            logger.warning(
                f"Household money stock != central bank money stock: {ape[ape > tolerance]}"
            )
            return False

        # Check that all the assets and liabilities are positive
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
            # Flows from Table 4.1, top to bottom
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
            "InterestEarnedOnBillsHousehold": {
                "notation": r"r(t-1)\cdot B_h(t-1)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Inflow", "Household"), ("Outflow", "Government")],
            },
            "CentralBankProfits": {
                "notation": r"r(t-1)\cdot B_{CB}(t-1)",
                "unit": "USD",
                "history": 0,
                "sectors": ["CentralBank"],
                "sfc": [("Inflow", "Government"), ("Outflow", "CentralBank")],
            },
            "Taxes": {
                "notation": r"T(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Outflow", "Household"), ("Inflow", "Government")],
            },
            # Stocks
            "HouseholdMoneyStock": {
                "notation": r"H_h(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "CentralBankMoneyStock": {
                "notation": r"H_{s}(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["CentralBank"],
                "sfc": [("Liability", ["CentralBank", "Capital"])],
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
            "Wealth": {
                "notation": r"V(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Liability", "Household"), ("Asset", "Government")],
            },
            # Indices
            "InterestRate": {
                "notation": r"r(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "DisposableIncome": {
                "notation": r"YD(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
        }
