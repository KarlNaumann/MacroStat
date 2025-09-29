"""
Variables class for Marco Veronese Passarella's 3IO-PC model
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import numpy as np

from macrostat.core.variables import Variables
from macrostat.models.IOPC.parameters import ParametersIOPC

logger = logging.getLogger(__name__)


class VariablesIOPC(Variables):
    """Variables class for Marco Veronese Passarella's 3IO-PC model"""

    version = "IOPC"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersIOPC | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables of Marco Veronese Passarella's 3IO-PC"""

        if parameters is None:
            parameters = ParametersIOPC()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def check_health(self, tolerance: float = 1e-4):  # pragma: no cover
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
        iosectors = self.parameters.hyper["iosectors"]
        return {
            # Consumption and Production
            "RealConsumptionHousehold": {
                "notation": r"c(t)",
                "unit": "RU",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "NominalConsumptionHousehold": {
                "notation": r"p_c(t)c(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Outflow", "Household"), ("Inflow", "IOSectors")],
            },
            "RealConsumptionGovernment": {
                "notation": r"g(t)",
                "unit": "RU",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Index", "Government")],
            },
            "NominalConsumptionGovernment": {
                "notation": r"p_g(t) g(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Outflow", "Government"), ("Inflow", "IOSectors")],
            },
            "RealGrossOutput": {
                "notation": r"x_i(t)",
                "unit": "RU_i",
                "history": 0,
                "sectors": iosectors,
                "sfc": [("Index", "Production")],
            },
            "RealFinalDemand": {
                "notation": r"d_i(t)",
                "unit": "RU_i",
                "history": 0,
                "sectors": iosectors,
                "sfc": [("Index", "Production")],
            },
            # Prices and Price-indices
            "Prices": {
                "notation": r"P_i(t)",
                "unit": "USD",
                "history": 0,
                "sectors": iosectors,
                "sfc": [("Index", "Production")],
            },
            "ConsumerPriceIndex": {  # CPI
                "notation": r"p_c(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "ConsumerPriceInflation": {  # CPI
                "notation": r"\pi(t)",
                "unit": "%pp",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "GovernmentPriceIndex": {
                "notation": r"p_g(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Index", "Government")],
            },
            # Other PCEX flows
            "NationalIncome": {
                "notation": r"Y(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Outflow", "S1"), ("Inflow", "Household")],
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
            "ExpectedDisposableIncome": {
                "notation": r"YD^e(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "PropensityToConsumeIncome": {
                "notation": r"\alpha_1(t)",
                "unit": ".",
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
        }
