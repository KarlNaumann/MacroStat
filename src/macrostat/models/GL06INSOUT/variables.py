"""
Variables class for the Godley-Lavoie 2006 INSOUT model (Chapter 10).
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
from macrostat.models.GL06INSOUT.parameters import ParametersGL06INSOUT

logger = logging.getLogger(__name__)


class VariablesGL06INSOUT(Variables):
    """Variables class for the Godley-Lavoie 2006 INSOUT model."""

    version = "GL06INSOUT"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersGL06INSOUT | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables of the Godley-Lavoie 2006 INSOUT model."""

        if parameters is None:
            parameters = ParametersGL06INSOUT()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def check_health(self, tolerance: float = 1e-4):  # pragma: no cover
        r"""Check the health of the variables.

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

        # 1. Credit market: LoansSupply == LoanDemand
        diff = (output["LoansSupply"] - output["LoanDemand"]).abs()
        if np.any(diff > tolerance):
            logger.warning(f"Loan market does not clear: max diff={diff.max()}")
            return False

        # 2. Advances market: AdvancesSupply == AdvancesDemand
        diff = (output["AdvancesSupply"] - output["AdvancesDemand"]).abs()
        if np.any(diff > tolerance):
            logger.warning(f"Advances market does not clear: max diff={diff.max()}")
            return False

        # 3. Bill market: BillsSupply == BillsHousehold + BillsBank + BillsCentralBank
        diff = (
            output["BillsSupply"]
            - output["BillsHousehold"]
            - output["BillsBank"]
            - output["BillsCentralBank"]
        ).abs()
        if np.any(diff > tolerance):
            logger.warning(f"Bill market does not clear: max diff={diff.max()}")
            return False

        return True

    def get_default_variables(self):
        """Return the default variables information dictionary."""
        return {
            # ----------------------------------------------------------------
            # Exogenous / scenario variables (history=1 for rate dynamics)
            # ----------------------------------------------------------------
            "RealGovernmentSpending": {
                "notation": r"g(t)",
                "unit": "real units",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Index", "Government")],
            },
            "BillRate": {
                "notation": r"r_b(t)",
                "unit": ".",
                "history": 1,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "BondYield": {
                "notation": r"r_{bl}(t)",
                "unit": ".",
                "history": 1,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            # ----------------------------------------------------------------
            # Endogenous interest rates
            # ----------------------------------------------------------------
            "DepositRate": {
                "notation": r"r_m(t)",
                "unit": ".",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "LoanRate": {
                "notation": r"r_l(t)",
                "unit": ".",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "AdvanceRate": {
                "notation": r"r_a(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["CentralBank"],
                "sfc": [("Index", "CentralBank")],
            },
            # ----------------------------------------------------------------
            # Prices and wages
            # ----------------------------------------------------------------
            "PriceLevel": {
                "notation": r"p(t)",
                "unit": ".",
                "history": 1,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "InflationRate": {
                "notation": r"\pi(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "NominalWage": {
                "notation": r"W(t)",
                "unit": "USD/worker",
                "history": 1,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "RealWage": {
                "notation": r"\omega(t)",
                "unit": "real units/worker",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "TargetRealWage": {
                "notation": r"\omega^T(t)",
                "unit": "real units/worker",
                "history": 1,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "UnitCost": {
                "notation": r"UC(t)",
                "unit": "USD/unit",
                "history": 1,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "NormalHistoricUnitCost": {
                "notation": r"NHUC(t)",
                "unit": "USD/unit",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            # ----------------------------------------------------------------
            # Firm output, expectations, and inventory
            # ----------------------------------------------------------------
            "ExpectedSales": {
                "notation": r"s^e(t)",
                "unit": "real units",
                "history": 1,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "RealSales": {
                "notation": r"s(t)",
                "unit": "real units",
                "history": 1,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "RealOutput": {
                "notation": r"y(t)",
                "unit": "real units",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "TargetInventorySalesRatio": {
                "notation": r"\sigma^T(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "TargetInventories": {
                "notation": r"inv^T(t)",
                "unit": "real units",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "ExpectedInventories": {
                "notation": r"inv^e(t)",
                "unit": "real units",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "RealInventories": {
                "notation": r"inv(t)",
                "unit": "real units",
                "history": 1,
                "sectors": ["Firm"],
                "sfc": [("Asset", "Firm")],
            },
            "ActualInventorySalesRatio": {
                "notation": r"\sigma(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "NominalInventories": {
                "notation": r"INV(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Firm"],
                "sfc": [("Asset", "Firm")],
            },
            # ----------------------------------------------------------------
            # Employment
            # ----------------------------------------------------------------
            "Employment": {
                "notation": r"N(t)",
                "unit": "workers",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "WageBill": {
                "notation": r"WB(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Outflow", "Firm"), ("Inflow", "Household")],
            },
            # ----------------------------------------------------------------
            # Nominal firm aggregates
            # ----------------------------------------------------------------
            "NominalSales": {
                "notation": r"S(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Inflow", "Firm"), ("Outflow", "Household")],
            },
            "NominalOutput": {
                "notation": r"Y(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "NominalConsumption": {
                "notation": r"C(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Outflow", "Household"), ("Inflow", "Firm")],
            },
            "GovernmentSpending": {
                "notation": r"G(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Outflow", "Government"), ("Inflow", "Firm")],
            },
            "RealConsumption": {
                "notation": r"c(t)",
                "unit": "real units",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "Taxes": {
                "notation": r"TX(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Outflow", "Firm"), ("Inflow", "Government")],
            },
            "LoanDemand": {
                "notation": r"L^d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            "FirmProfits": {
                "notation": r"FP(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Firm"],
                "sfc": [("Index", "Firm")],
            },
            # ----------------------------------------------------------------
            # Household income and wealth
            # ----------------------------------------------------------------
            "TotalDividends": {
                "notation": r"FD(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Inflow", "Household"), ("Outflow", "Firm")],
            },
            "RegularDisposableIncome": {
                "notation": r"YD_r(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "CapitalGains": {
                "notation": r"CG(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "HaigSimonsDisposableIncome": {
                "notation": r"YD_{hs}(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "NominalWealth": {
                "notation": r"V(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Household"],
                "sfc": [("Liability", "Household")],
            },
            "RealRegularDisposableIncome": {
                "notation": r"yd_r(t)",
                "unit": "real units",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "RealWealth": {
                "notation": r"v(t)",
                "unit": "real units",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "ExpectedRealDisposableIncome": {
                "notation": r"yd_r^e(t)",
                "unit": "real units",
                "history": 1,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "NominalExpectedDisposableIncome": {
                "notation": r"YD^e(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "ExpectedNominalWealth": {
                "notation": r"V^e(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            # ----------------------------------------------------------------
            # Household portfolio assets
            # ----------------------------------------------------------------
            "CashDemand": {
                "notation": r"Hh^d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "ExpectedNonCashWealth": {
                "notation": r"V^e_{nc}(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "M1DemandTentative": {
                "notation": r"M1^{dt}(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "M2Demand": {
                "notation": r"M2^d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "BillsDemand": {
                "notation": r"B^d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "BondsDemand": {
                "notation": r"BL^d(t)",
                "unit": "bonds",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "NonCashWealth": {
                "notation": r"V_{nc}(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "SwitchM1Positive": {
                "notation": r"z_1(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "SwitchM2Absorber": {
                "notation": r"z_2(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "CashHousehold": {
                "notation": r"Hh(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "M1Household": {
                "notation": r"M1_h(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "M2Household": {
                "notation": r"M2_h(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "BillsHousehold": {
                "notation": r"B_h(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "BondsHousehold": {
                "notation": r"BL_h(t)",
                "unit": "bonds",
                "history": 1,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "BondPrice": {
                "notation": r"p_{bl}(t)",
                "unit": "USD/bond",
                "history": 1,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "ExpectedReturnOnBonds": {
                "notation": r"ERr_{bl}(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            # ----------------------------------------------------------------
            # Government balance sheet
            # ----------------------------------------------------------------
            "PublicSectorBorrowingRequirement": {
                "notation": r"PSBR(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Index", "Government")],
            },
            "BondsSupply": {
                "notation": r"BL_s(t)",
                "unit": "bonds",
                "history": 1,
                "sectors": ["Government"],
                "sfc": [("Liability", "Government")],
            },
            "BillsSupply": {
                "notation": r"B_s(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Government"],
                "sfc": [("Liability", "Government")],
            },
            "GovernmentDebt": {
                "notation": r"GD(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Government"],
                "sfc": [("Index", "Government")],
            },
            # ----------------------------------------------------------------
            # Bank balance sheet
            # ----------------------------------------------------------------
            "LoansSupply": {
                "notation": r"L_s(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Asset", "Bank")],
            },
            "M1Supply": {
                "notation": r"M1_s(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Liability", "Bank")],
            },
            "M2Supply": {
                "notation": r"M2_s(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Liability", "Bank")],
            },
            "RequiredReserves": {
                "notation": r"RR(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "CashBanksSupply": {
                "notation": r"Hb_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Bank"],
                "sfc": [("Asset", "Bank")],
            },
            "BillsBankTentative": {
                "notation": r"B_b^T(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "BankLiquidityRatioTentative": {
                "notation": r"BLR^T(t)",
                "unit": ".",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "SwitchBankBelowFloor": {
                "notation": r"z_3(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "AdvancesDemand": {
                "notation": r"A^d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "BillsBank": {
                "notation": r"B_b(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Asset", "Bank")],
            },
            "BankLiquidityRatio": {
                "notation": r"BLR(t)",
                "unit": ".",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "LaggedM1Supply": {
                "notation": r"M1_s(t-1)^{lag}",
                "unit": "USD",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "LaggedM2Supply": {
                "notation": r"M2_s(t-1)^{lag}",
                "unit": "USD",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "BankProfits": {
                "notation": r"FBP(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            "BankProfitMargin": {
                "notation": r"BPM(t)",
                "unit": ".",
                "history": 1,
                "sectors": ["Bank"],
                "sfc": [("Index", "Bank")],
            },
            # ----------------------------------------------------------------
            # Central bank balance sheet
            # ----------------------------------------------------------------
            "AdvancesSupply": {
                "notation": r"A_s(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["CentralBank"],
                "sfc": [("Asset", "CentralBank")],
            },
            "BillsCentralBank": {
                "notation": r"B_{cb}(t)",
                "unit": "USD",
                "history": 1,
                "sectors": ["CentralBank"],
                "sfc": [("Asset", "CentralBank")],
            },
            "HighPoweredMoney": {
                "notation": r"H_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["CentralBank"],
                "sfc": [("Liability", "CentralBank")],
            },
            "BankReservesSupply": {
                "notation": r"HPM_b(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["CentralBank"],
                "sfc": [("Index", "CentralBank")],
            },
            "CentralBankProfits": {
                "notation": r"FCB(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["CentralBank"],
                "sfc": [("Inflow", "Government"), ("Outflow", "CentralBank")],
            },
        }
