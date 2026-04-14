"""
Behavior class for the Godley-Lavoie 2006 INSOUT model (Chapter 10).
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging
import warnings

import torch

from macrostat.core.behavior import Behavior
from macrostat.models.GL06INSOUT.parameters import ParametersGL06INSOUT
from macrostat.models.GL06INSOUT.scenarios import ScenariosGL06INSOUT
from macrostat.models.GL06INSOUT.variables import VariablesGL06INSOUT

logger = logging.getLogger(__name__)


class BehaviorGL06INSOUT(Behavior):
    """Behavior class for the Godley-Lavoie 2006 INSOUT model.

    Implements the INSOUT model from Chapter 10 of Godley & Lavoie (2006),
    featuring 5 sectors (Household, Firm, Government, CentralBank, Bank),
    endogenous bank interest rates, inventory dynamics, and wage-price dynamics.
    """

    version = "GL06INSOUT"

    def __init__(
        self,
        parameters: ParametersGL06INSOUT | None = None,
        scenarios: ScenariosGL06INSOUT | None = None,
        variables: VariablesGL06INSOUT | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        """Initialize the behavior of the Godley-Lavoie 2006 INSOUT model.

        Parameters
        ----------
        parameters : ParametersGL06INSOUT | None
            The parameters of the model.
        scenarios : ScenariosGL06INSOUT | None
            The scenarios of the model.
        variables : VariablesGL06INSOUT | None
            The variables of the model.
        scenario : int
            The scenario to use for the model.
        debug : bool
            Whether to enable debug mode.
        """
        if parameters is None:
            parameters = ParametersGL06INSOUT()
        if scenarios is None:
            scenarios = ScenariosGL06INSOUT()
        if variables is None:
            variables = VariablesGL06INSOUT()

        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            scenario=scenario,
            debug=debug,
        )

    ############################################################################
    # Initialization
    ############################################################################

    def initialize(self):
        """Initialize all variables to zero, then set non-zero initial conditions.

        Matches the R sfcr reference: ``initial = sfcr_set(p ~ 1, W ~ 1, UC ~ 1, BPM ~ 0.0035)``.
        Everything else starts at zero.
        """
        # Use a reference tensor for device/dtype consistency
        ref = next(iter(self.state.values()))
        for key in self.state:
            self.state[key] = torch.zeros_like(ref)

        # Non-zero initial conditions (matching R sfcr)
        self.state["PriceLevel"] = torch.ones_like(ref)
        self.state["NominalWage"] = torch.ones_like(ref)
        self.state["UnitCost"] = torch.ones_like(ref)
        self.state["BankProfitMargin"] = torch.ones_like(ref) * 0.0035

    ############################################################################
    # Step
    ############################################################################

    def step(self, **kwargs):
        """Step function of the Godley-Lavoie 2006 INSOUT model."""

        # Block 0: Exogenous scenario variables
        self.set_bill_rate(**kwargs)
        self.set_bond_yield(**kwargs)
        self.set_real_government_spending(**kwargs)

        # Block 1: Bank profits -> profit margin -> endogenous rates
        self.bank_profits(**kwargs)
        self.bank_profit_margin(**kwargs)
        self.deposit_rate(**kwargs)
        self.loan_rate(**kwargs)
        self.central_bank_profits(**kwargs)

        # Block 2: Bond pricing
        self.bond_price(**kwargs)
        self.expected_return_on_bonds(**kwargs)

        # Block 3: Firm expectations, output, cost, and pricing
        # y = sE + invE - inv[-1] depends only on expectations/lagged values,
        # so y, N, W, WB, UC can all be computed before NHUC. This allows
        # NHUC to use current UC (matching R's simultaneous solver).
        self.expected_sales(**kwargs)
        self.target_inventory_sales_ratio(**kwargs)
        self.target_inventories(**kwargs)
        self.expected_inventories(**kwargs)
        self.real_output(**kwargs)
        self.employment(**kwargs)
        self.nominal_wage(**kwargs)
        self.wage_bill(**kwargs)
        self.unit_cost(**kwargs)
        self.normal_historic_unit_cost(**kwargs)
        self.price_level(**kwargs)
        self.inflation_rate(**kwargs)

        # Block 4: Household income expectations
        self.expected_real_disposable_income(**kwargs)
        self.nominal_expected_disposable_income(**kwargs)

        # Block 5: Consumption and realized sales
        self.real_consumption(**kwargs)
        self.nominal_consumption(**kwargs)
        self.real_sales(**kwargs)

        # Block 6: Wage-price dynamics (post-pricing)
        self.real_wage(**kwargs)
        self.target_real_wage(**kwargs)

        # Block 7: Nominal firm aggregates (taxes now here — indirect on sales)
        self.nominal_sales(**kwargs)
        self.taxes(**kwargs)
        self.real_inventories(**kwargs)
        self.actual_inventory_sales_ratio(**kwargs)
        self.nominal_inventories(**kwargs)
        self.loan_demand(**kwargs)
        self.firm_profits(**kwargs)
        self.nominal_output(**kwargs)

        # Block 8: Household dividends, wealth, income
        self.total_dividends(**kwargs)
        self.capital_gains(**kwargs)
        self.regular_disposable_income(**kwargs)
        self.haig_simons_disposable_income(**kwargs)
        self.nominal_wealth(**kwargs)
        self.real_regular_disposable_income(**kwargs)
        self.real_wealth(**kwargs)
        self.expected_nominal_wealth(**kwargs)
        self.cash_demand(**kwargs)
        self.expected_non_cash_wealth(**kwargs)
        self.non_cash_wealth(**kwargs)

        # Block 9: Portfolio allocation
        self.m2_demand(**kwargs)
        self.bills_demand(**kwargs)
        self.bonds_demand(**kwargs)
        self.m1_demand_tentative(**kwargs)
        self.cash_household(**kwargs)
        self.portfolio_switches(**kwargs)
        self.m1_household(**kwargs)
        self.m2_household(**kwargs)
        self.bills_household(**kwargs)
        self.bonds_household(**kwargs)

        # Block 10: Government
        self.government_spending(**kwargs)
        self.bonds_supply(**kwargs)
        self.public_sector_borrowing_requirement(**kwargs)
        self.bills_supply(**kwargs)
        self.government_debt(**kwargs)

        # Block 11: Bank balance sheet
        self.loans_supply(**kwargs)
        self.m1_supply(**kwargs)
        self.m2_supply(**kwargs)
        self.required_reserves(**kwargs)
        self.cash_banks_supply(**kwargs)
        self.bank_bills_tentative(**kwargs)
        self.bank_liquidity_ratio_tentative(**kwargs)
        self.switch_bank_below_floor(**kwargs)
        self.advances_demand(**kwargs)
        self.bank_bill_holdings(**kwargs)
        self.bank_liquidity_ratio(**kwargs)
        self.lagged_m1_supply(**kwargs)
        self.lagged_m2_supply(**kwargs)

        # Block 12: Central bank
        self.advance_rate(**kwargs)
        self.advances_supply(**kwargs)
        self.bills_central_bank(**kwargs)
        self.high_powered_money(**kwargs)
        self.bank_reserves_supply(**kwargs)

    ############################################################################
    # Block 0: Exogenous scenario variables
    ############################################################################

    def set_bill_rate(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set the bill rate from the scenario.

        Dependency
        ----------
        - scenario: BillRate

        Sets
        -----
        - BillRate
        """
        self.state["BillRate"] = scenario["BillRate"]

    def set_bond_yield(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set the bond yield from the scenario.

        Dependency
        ----------
        - scenario: BondYield

        Sets
        -----
        - BondYield
        """
        self.state["BondYield"] = scenario["BondYield"]

    def set_real_government_spending(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set real government spending from the scenario.

        Dependency
        ----------
        - scenario: RealGovernmentSpending

        Sets
        -----
        - RealGovernmentSpending
        """
        self.state["RealGovernmentSpending"] = scenario["RealGovernmentSpending"]

    ############################################################################
    # Block 1: Bank profits -> profit margin -> endogenous rates
    ############################################################################

    def bank_profits(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate bank profits from prior-period stocks and rates.

        Equations
        ---------
        .. math::
            FBP(t) = r_l(t-1) \cdot L_s(t-1) + r_b(t-1) \cdot B_b(t-1)
                     - r_m(t-1) \cdot M2_s(t-1) - r_a(t-1) \cdot A_s(t-1)

        Dependency
        ----------
        - prior: LoanRate, LoansSupply, BillRate, BillsBank
        - prior: DepositRate, M2Supply, AdvanceRate, AdvancesSupply

        Sets
        -----
        - BankProfits
        """
        self.state["BankProfits"] = (
            self.prior["LoanRate"] * self.prior["LoansSupply"]
            + self.prior["BillRate"] * self.prior["BillsBank"]
            - self.prior["DepositRate"] * self.prior["M2Supply"]
            - self.prior["AdvanceRate"] * self.prior["AdvancesSupply"]
        )

    def bank_profit_margin(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the bank profit margin as a share of the prior deposit base.

        Equations
        ---------
        .. math::
            BPM(t) = \frac{FBP(t) + FBP(t-1)}{M1_s(t-1) + M1_s(t-2)
                     + M2_s(t-1) + M2_s(t-2)}

        Uses LaggedM1Supply(t-1)=M1_s(t-1) and prior LaggedM1Supply(t-2)=M1_s(t-2).

        Dependency
        ----------
        - state: BankProfits
        - prior: BankProfits, LaggedM1Supply, LaggedM2Supply

        Sets
        -----
        - BankProfitMargin
        """
        # LaggedM1Supply(t-1) is stored in prior; its own prior gives t-2.
        # At initialization both are 0; guard ensures no division by zero.
        denom = (
            self.prior["M1Supply"]
            + self.prior["LaggedM1Supply"]
            + self.prior["M2Supply"]
            + self.prior["LaggedM2Supply"]
        )
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        self.state["BankProfitMargin"] = torch.where(
            denom > 0,
            (self.state["BankProfits"] + self.prior["BankProfits"]) / safe_denom,
            self.prior["BankProfitMargin"],
        )

    def deposit_rate(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the endogenous deposit rate based on bank liquidity.

        Equations
        ---------
        .. math::
            r_m(t) = r_m(t-1) + \zeta_m \cdot (z_4 - z_5)
                     + \zeta_b \cdot (r_b(t) - r_b(t-1))

        where :math:`z_4 = 1` if :math:`BLR^T(t-1) < bot`,
        :math:`z_5 = 1` if :math:`BLR^T(t-1) > top`.

        Dependency
        ----------
        - prior: DepositRate, BankLiquidityRatioTentative, BillRate
        - state: BillRate
        - params: DepositRateAdjSpeed, DepositRateBillSensitivity
        - params: BankLiquidityFloor, BankLiquidityCeiling

        Sets
        -----
        - DepositRate
        """
        ref = self.prior["BankLiquidityRatioTentative"]
        ones = torch.ones_like(ref)
        zeros = torch.zeros_like(ref)

        z4 = torch.where(
            ref < params["BankLiquidityFloor"],
            ones,
            zeros,
        )
        z5 = torch.where(
            ref > params["BankLiquidityCeiling"],
            ones,
            zeros,
        )

        self.state["DepositRate"] = (
            self.prior["DepositRate"]
            + params["DepositRateAdjSpeed"] * (z4 - z5)
            + params["DepositRateBillSensitivity"]
            * (self.state["BillRate"] - self.prior["BillRate"])
        )

    def loan_rate(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the endogenous loan rate based on bank profit margin.

        Equations
        ---------
        .. math::
            r_l(t) = r_l(t-1) + \zeta_l \cdot (z_6 - z_7)
                     + (r_b(t) - r_b(t-1))

        where :math:`z_6 = 1` if :math:`BPM(t) < bot_{pm}`,
        :math:`z_7 = 1` if :math:`BPM(t) > top_{pm}`.

        Dependency
        ----------
        - prior: LoanRate, BillRate
        - state: BankProfitMargin, BillRate
        - state: BillRate
        - params: LoanRateProfitAdjSpeed, BankProfitFloor, BankProfitCeiling

        Sets
        -----
        - LoanRate
        """
        ref = self.state["BankProfitMargin"]
        ones = torch.ones_like(ref)
        zeros = torch.zeros_like(ref)

        z6 = torch.where(ref < params["BankProfitFloor"], ones, zeros)
        z7 = torch.where(ref > params["BankProfitCeiling"], ones, zeros)

        self.state["LoanRate"] = (
            self.prior["LoanRate"]
            + params["LoanRateProfitAdjSpeed"] * (z6 - z7)
            + (self.state["BillRate"] - self.prior["BillRate"])
        )

    def central_bank_profits(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate central bank profits from prior-period stocks and rates.

        Equations
        ---------
        .. math::
            FCB(t) = r_b(t-1) \cdot B_{cb}(t-1) + r_a(t-1) \cdot A_s(t-1)

        Dependency
        ----------
        - prior: BillRate, BillsCentralBank, AdvanceRate, AdvancesSupply

        Sets
        -----
        - CentralBankProfits
        """
        self.state["CentralBankProfits"] = (
            self.prior["BillRate"] * self.prior["BillsCentralBank"]
            + self.prior["AdvanceRate"] * self.prior["AdvancesSupply"]
        )

    ############################################################################
    # Block 2: Bond pricing
    ############################################################################

    def bond_price(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the bond price as the inverse of the bond yield.

        Equations
        ---------
        .. math::
            p_{bl}(t) = \frac{1}{r_{bl}(t)}

        Dependency
        ----------
        - state: BondYield

        Sets
        -----
        - BondPrice
        """
        by = self.state["BondYield"]
        safe_by = torch.where(by > 0, by, torch.ones_like(by))
        self.state["BondPrice"] = torch.where(
            by > 0,
            1.0 / safe_by,
            torch.zeros_like(by),
        )

    def expected_return_on_bonds(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the expected return on bonds (static expectations).

        Equations
        ---------
        .. math::
            ERr_{bl}(t) = r_{bl}(t)

        Dependency
        ----------
        - state: BondYield

        Sets
        -----
        - ExpectedReturnOnBonds
        """
        self.state["ExpectedReturnOnBonds"] = self.state["BondYield"]

    ############################################################################
    # Block 3: Firm expectations and pricing
    ############################################################################

    def expected_sales(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate expected real sales using adaptive expectations.

        Equations
        ---------
        .. math::
            s^e(t) = \beta \cdot s(t-1) + (1 - \beta) \cdot s^e(t-1)

        Dependency
        ----------
        - prior: RealSales, ExpectedSales
        - params: SalesExpectationWeight

        Sets
        -----
        - ExpectedSales
        """
        self.state["ExpectedSales"] = (
            params["SalesExpectationWeight"] * self.prior["RealSales"]
            + (1.0 - params["SalesExpectationWeight"]) * self.prior["ExpectedSales"]
        )

    def target_inventory_sales_ratio(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the target inventory-sales ratio.

        Equations
        ---------
        .. math::
            \sigma^T(t) = \sigma_0 - \sigma_1 \cdot r_l(t)

        Dependency
        ----------
        - state: LoanRate
        - params: InventorySalesRatioBaseline, InventorySalesRatioInterestSens

        Sets
        -----
        - TargetInventorySalesRatio
        """
        self.state["TargetInventorySalesRatio"] = (
            params["InventorySalesRatioBaseline"]
            - params["InventorySalesRatioInterestSens"] * self.state["LoanRate"]
        )

    def target_inventories(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate target inventories.

        Equations
        ---------
        .. math::
            inv^T(t) = \sigma^T(t) \cdot s^e(t)

        Dependency
        ----------
        - state: TargetInventorySalesRatio, ExpectedSales

        Sets
        -----
        - TargetInventories
        """
        self.state["TargetInventories"] = (
            self.state["TargetInventorySalesRatio"] * self.state["ExpectedSales"]
        )

    def expected_inventories(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate expected end-of-period inventories via partial adjustment.

        Equations
        ---------
        .. math::
            inv^e(t) = inv(t-1) - \gamma \cdot (inv(t-1) - inv^T(t))

        Dependency
        ----------
        - prior: RealInventories
        - state: TargetInventories
        - params: InventoryAdjustmentSpeed

        Sets
        -----
        - ExpectedInventories
        """
        self.state["ExpectedInventories"] = self.prior["RealInventories"] - (
            params["InventoryAdjustmentSpeed"]
            * (self.prior["RealInventories"] - self.state["TargetInventories"])
        )

    def normal_historic_unit_cost(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the normal historic unit cost.

        A weighted average of current and prior unit cost, with inventory
        financing cost applied to the prior component.

        Equations
        ---------
        .. math::
            NHUC(t) = (1 - \sigma^T) \cdot UC(t)
                      + \sigma^T \cdot (1 + r_l(t-1)) \cdot UC(t-1)

        Dependency
        ----------
        - state: UnitCost, TargetInventorySalesRatio
        - prior: UnitCost, LoanRate

        Sets
        -----
        - NormalHistoricUnitCost
        """
        sigmaT = self.state["TargetInventorySalesRatio"]
        self.state["NormalHistoricUnitCost"] = (1.0 - sigmaT) * self.state[
            "UnitCost"
        ] + sigmaT * (1.0 + self.prior["LoanRate"]) * self.prior["UnitCost"]

    def price_level(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the price level as a markup over normal historic unit cost.

        Equations
        ---------
        .. math::
            p(t) = (1 + \tau) \cdot (1 + \phi) \cdot NHUC(t)

        Dependency
        ----------
        - state: NormalHistoricUnitCost
        - params: TaxRate, MarkupRate

        Sets
        -----
        - PriceLevel
        """
        self.state["PriceLevel"] = (
            (1.0 + params["TaxRate"])
            * (1.0 + params["MarkupRate"])
            * self.state["NormalHistoricUnitCost"]
        )

    def inflation_rate(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the inflation rate.

        Equations
        ---------
        .. math::
            \pi(t) = \frac{p(t) - p(t-1)}{p(t-1)}

        Dependency
        ----------
        - state: PriceLevel
        - prior: PriceLevel

        Sets
        -----
        - InflationRate
        """
        pp = self.prior["PriceLevel"]
        safe_pp = torch.where(pp > 0, pp, torch.ones_like(pp))
        self.state["InflationRate"] = torch.where(
            pp > 0,
            (self.state["PriceLevel"] - pp) / safe_pp,
            torch.zeros_like(pp),
        )

    ############################################################################
    # Block 4: Household income expectations
    ############################################################################

    def expected_real_disposable_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate expected real disposable income via adaptive expectations.

        Equations
        ---------
        .. math::
            yd_r^e(t) = \varepsilon \cdot yd_r(t-1)
                        + (1 - \varepsilon) \cdot yd_r^e(t-1)

        Dependency
        ----------
        - prior: RealRegularDisposableIncome, ExpectedRealDisposableIncome
        - params: IncomeExpectationWeight

        Sets
        -----
        - ExpectedRealDisposableIncome
        """
        self.state["ExpectedRealDisposableIncome"] = (
            params["IncomeExpectationWeight"]
            * self.prior["RealRegularDisposableIncome"]
            + (1.0 - params["IncomeExpectationWeight"])
            * self.prior["ExpectedRealDisposableIncome"]
        )

    def nominal_expected_disposable_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the nominal expected disposable income.

        Equations
        ---------
        .. math::
            YD^e(t) = yd_r^e(t) \cdot p(t) + \pi(t) \cdot \frac{V(t-1)}{p(t)}

        Dependency
        ----------
        - state: ExpectedRealDisposableIncome, PriceLevel, InflationRate
        - prior: NominalWealth

        Sets
        -----
        - NominalExpectedDisposableIncome
        """
        pl = self.state["PriceLevel"]
        safe_pl = torch.where(pl > 0, pl, torch.ones_like(pl))
        self.state["NominalExpectedDisposableIncome"] = (
            self.state["ExpectedRealDisposableIncome"] * pl
            + self.state["InflationRate"] * self.prior["NominalWealth"] / safe_pl
        )

    ############################################################################
    # Block 5: Consumption
    ############################################################################

    def real_consumption(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate real consumption.

        Equations
        ---------
        .. math::
            c(t) = \alpha_0 + \alpha_1 \cdot yd_r^e(t) + \alpha_2 \cdot v(t-1)

        Dependency
        ----------
        - state: ExpectedRealDisposableIncome
        - prior: RealWealth
        - params: AutonomousConsumption, PropensityToConsumeIncome,
                  PropensityToConsumeWealth

        Sets
        -----
        - RealConsumption
        """
        self.state["RealConsumption"] = (
            params["AutonomousConsumption"]
            + params["PropensityToConsumeIncome"]
            * self.state["ExpectedRealDisposableIncome"]
            + params["PropensityToConsumeWealth"] * self.prior["RealWealth"]
        )

    def nominal_consumption(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate nominal consumption.

        Equations
        ---------
        .. math::
            C(t) = c(t) \cdot p(t)

        Dependency
        ----------
        - state: RealConsumption, PriceLevel

        Sets
        -----
        - NominalConsumption
        """
        self.state["NominalConsumption"] = (
            self.state["RealConsumption"] * self.state["PriceLevel"]
        )

    def real_sales(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate real sales as consumption plus real government spending.

        Equations
        ---------
        .. math::
            s(t) = c(t) + g(t)

        Dependency
        ----------
        - state: RealConsumption, RealGovernmentSpending

        Sets
        -----
        - RealSales
        """
        self.state["RealSales"] = (
            self.state["RealConsumption"] + self.state["RealGovernmentSpending"]
        )

    ############################################################################
    # Block 6: Firm output, employment, wages
    ############################################################################

    def real_output(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate real output as expected sales plus inventory adjustment.

        Equations
        ---------
        .. math::
            y(t) = s^e(t) + inv^e(t) - inv(t-1)

        Dependency
        ----------
        - state: ExpectedSales, ExpectedInventories
        - prior: RealInventories

        Sets
        -----
        - RealOutput
        """
        self.state["RealOutput"] = (
            self.state["ExpectedSales"]
            + self.state["ExpectedInventories"]
            - self.prior["RealInventories"]
        )

    def employment(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate employment from real output and labor productivity.

        Equations
        ---------
        .. math::
            N(t) = \frac{y(t)}{pr}

        Dependency
        ----------
        - state: RealOutput
        - params: LaborProductivity

        Sets
        -----
        - Employment
        """
        lp = params["LaborProductivity"]
        safe_lp = torch.where(lp > 0, lp, torch.ones_like(lp))
        self.state["Employment"] = torch.where(
            lp > 0,
            self.state["RealOutput"] / safe_lp,
            torch.zeros_like(lp),
        )

    def target_real_wage(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the target real wage from productivity and employment.

        Equations
        ---------
        .. math::
            \omega^T(t) = \exp(\Omega_0 + \Omega_1 \log(pr)
                + \Omega_2 \log(N(t) / N_{fe}))

        Dependency
        ----------
        - state: Employment
        - params: RealWageTargetConstant, RealWageProductivityElasticity,
                  RealWageEmploymentElasticity, LaborProductivity, FullEmployment

        Sets
        -----
        - TargetRealWage
        """
        lp = params["LaborProductivity"]
        safe_lp = torch.where(lp > 0, lp, torch.ones_like(lp))
        log_pr = torch.where(
            lp > 0,
            torch.log(safe_lp),
            torch.zeros_like(lp),
        )
        emp = self.state["Employment"]
        fe = params["FullEmployment"]
        safe_ratio = torch.where(emp > 0, emp / fe, torch.ones_like(emp))
        log_ratio = torch.where(
            emp > 0,
            torch.log(safe_ratio),
            torch.zeros_like(emp),
        )
        self.state["TargetRealWage"] = torch.exp(
            params["RealWageTargetConstant"]
            + params["RealWageProductivityElasticity"] * log_pr
            + params["RealWageEmploymentElasticity"] * log_ratio
        )

    def nominal_wage(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the nominal wage via partial adjustment to target real wage.

        Equations
        ---------
        .. math::
            W(t) = W(t-1) \cdot (1 + \Omega_3 \cdot (\omega^T(t-1) - \omega(t-1)))

        where :math:`\omega(t-1) = W(t-1) / p(t-1)`.

        Dependency
        ----------
        - prior: NominalWage, PriceLevel, TargetRealWage
        - params: WageAdjustmentSpeed

        Sets
        -----
        - NominalWage
        """
        ppl = self.prior["PriceLevel"]
        safe_ppl = torch.where(ppl > 0, ppl, torch.ones_like(ppl))
        prior_real_wage = torch.where(
            ppl > 0,
            self.prior["NominalWage"] / safe_ppl,
            torch.zeros_like(ppl),
        )
        self.state["NominalWage"] = self.prior["NominalWage"] * (
            1.0
            + params["WageAdjustmentSpeed"]
            * (self.prior["TargetRealWage"] - prior_real_wage)
        )

    def wage_bill(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the nominal wage bill.

        Equations
        ---------
        .. math::
            WB(t) = N(t) \cdot W(t)

        Dependency
        ----------
        - state: Employment, NominalWage

        Sets
        -----
        - WageBill
        """
        self.state["WageBill"] = self.state["Employment"] * self.state["NominalWage"]

    def unit_cost(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the unit cost of production.

        Equations
        ---------
        .. math::
            UC(t) = \frac{WB(t)}{y(t)}

        Dependency
        ----------
        - state: WageBill, RealOutput

        Sets
        -----
        - UnitCost
        """
        # UC = WB/y = (N*W)/y = ((y/pr)*W)/y = W/pr.
        # When y = 0, WB/y is 0/0; use W/pr as the equivalent fallback.
        ro = self.state["RealOutput"]
        safe_ro = torch.where(ro > 0, ro, torch.ones_like(ro))
        self.state["UnitCost"] = torch.where(
            ro > 0,
            self.state["WageBill"] / safe_ro,
            self.state["NominalWage"] / params["LaborProductivity"],
        )

    def real_wage(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the current real wage.

        Equations
        ---------
        .. math::
            \omega(t) = W(t) / p(t)

        Dependency
        ----------
        - state: NominalWage, PriceLevel

        Sets
        -----
        - RealWage
        """
        pl = self.state["PriceLevel"]
        safe_pl = torch.where(pl > 0, pl, torch.ones_like(pl))
        self.state["RealWage"] = torch.where(
            pl > 0,
            self.state["NominalWage"] / safe_pl,
            torch.zeros_like(pl),
        )

    ############################################################################
    # Block 7: Nominal firm aggregates
    ############################################################################

    def nominal_sales(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate nominal sales.

        Equations
        ---------
        .. math::
            S(t) = s(t) \cdot p(t)

        Dependency
        ----------
        - state: RealSales, PriceLevel

        Sets
        -----
        - NominalSales
        """
        self.state["NominalSales"] = self.state["RealSales"] * self.state["PriceLevel"]

    def real_inventories(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate real end-of-period inventories.

        Equations
        ---------
        .. math::
            inv(t) = inv(t-1) + y(t) - s(t)

        Dependency
        ----------
        - prior: RealInventories
        - state: RealOutput, RealSales

        Sets
        -----
        - RealInventories
        """
        self.state["RealInventories"] = (
            self.prior["RealInventories"]
            + self.state["RealOutput"]
            - self.state["RealSales"]
        )

    def actual_inventory_sales_ratio(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the actual inventory-sales ratio (beginning-of-period).

        Equations
        ---------
        .. math::
            \sigma(t) = inv(t-1) / s(t)

        Dependency
        ----------
        - prior: RealInventories
        - state: RealSales

        Sets
        -----
        - ActualInventorySalesRatio
        """
        rs = self.state["RealSales"]
        safe_rs = torch.where(rs > 0, rs, torch.ones_like(rs))
        self.state["ActualInventorySalesRatio"] = torch.where(
            rs > 0,
            self.prior["RealInventories"] / safe_rs,
            torch.zeros_like(rs),
        )

    def nominal_inventories(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate nominal inventories valued at current unit cost.

        Equations
        ---------
        .. math::
            INV(t) = inv(t) \cdot UC(t)

        Dependency
        ----------
        - state: RealInventories, UnitCost

        Sets
        -----
        - NominalInventories
        """
        self.state["NominalInventories"] = (
            self.state["RealInventories"] * self.state["UnitCost"]
        )

    def loan_demand(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Set firm loan demand equal to nominal inventories.

        Equations
        ---------
        .. math::
            L^d(t) = INV(t)

        Dependency
        ----------
        - state: NominalInventories

        Sets
        -----
        - LoanDemand
        """
        self.state["LoanDemand"] = self.state["NominalInventories"]

    def firm_profits(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate firm profits.

        Equations
        ---------
        .. math::
            FP(t) = S(t) - TX(t) - WB(t) + (INV(t) - INV(t-1))
                    - r_l(t-1) \cdot L_s(t-1)

        Dependency
        ----------
        - state: NominalSales, Taxes, WageBill, NominalInventories
        - prior: LoanRate, LoansSupply, NominalInventories

        Sets
        -----
        - FirmProfits
        """
        self.state["FirmProfits"] = (
            self.state["NominalSales"]
            - self.state["Taxes"]
            + self.state["NominalInventories"]
            - self.prior["NominalInventories"]
            - self.state["WageBill"]
            - self.prior["LoanRate"] * self.prior["LoansSupply"]
        )

    def nominal_output(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate nominal output (sales at price + inventory change at cost).

        Equations
        ---------
        .. math::
            Y(t) = p(t) \cdot s(t) + UC(t) \cdot (inv(t) - inv(t-1))

        Dependency
        ----------
        - state: PriceLevel, RealSales, UnitCost, RealInventories
        - prior: RealInventories

        Sets
        -----
        - NominalOutput
        """
        self.state["NominalOutput"] = self.state["PriceLevel"] * self.state[
            "RealSales"
        ] + self.state["UnitCost"] * (
            self.state["RealInventories"] - self.prior["RealInventories"]
        )

    ############################################################################
    # Block 8: Household dividends, taxes, wealth, income
    ############################################################################

    def total_dividends(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate total dividends distributed to households.

        All firm and bank profits are distributed as dividends.

        Equations
        ---------
        .. math::
            FD(t) = FP(t) + FBP(t)

        Dependency
        ----------
        - state: FirmProfits, BankProfits

        Sets
        -----
        - TotalDividends
        """
        self.state["TotalDividends"] = (
            self.state["FirmProfits"] + self.state["BankProfits"]
        )

    def taxes(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate indirect production tax on sales.

        Equations
        ---------
        .. math::
            TX(t) = \frac{\tau}{1 + \tau} \cdot S(t)

        Dependency
        ----------
        - state: NominalSales
        - params: TaxRate

        Sets
        -----
        - Taxes
        """
        self.state["Taxes"] = (
            self.state["NominalSales"] * params["TaxRate"] / (1.0 + params["TaxRate"])
        )

    def capital_gains(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate capital gains on bond holdings.

        Equations
        ---------
        .. math::
            CG(t) = (p_{bl}(t) - p_{bl}(t-1)) \cdot BL_h(t-1)

        Dependency
        ----------
        - state: BondPrice
        - prior: BondPrice, BondsHousehold

        Sets
        -----
        - CapitalGains
        """
        self.state["CapitalGains"] = (
            self.state["BondPrice"] - self.prior["BondPrice"]
        ) * self.prior["BondsHousehold"]

    def regular_disposable_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate regular (non-capital-gains) disposable income.

        With indirect taxation, households receive firm/bank dividends (which
        already net out the production tax) plus interest income. No explicit
        tax subtraction at the household level.

        Equations
        ---------
        .. math::
            YD_r(t) = WB(t) + FD(t)
                      + r_m(t-1) \cdot M2_h(t-1) + r_b(t-1) \cdot B_h(t-1)
                      + BL_h(t-1)

        Dependency
        ----------
        - state: WageBill, TotalDividends
        - prior: DepositRate, M2Household, BillRate, BillsHousehold, BondsHousehold

        Sets
        -----
        - RegularDisposableIncome
        """
        interest_income = (
            self.prior["DepositRate"] * self.prior["M2Household"]
            + self.prior["BillRate"] * self.prior["BillsHousehold"]
            + self.prior["BondsHousehold"]
        )
        self.state["RegularDisposableIncome"] = (
            self.state["WageBill"] + self.state["TotalDividends"] + interest_income
        )

    def haig_simons_disposable_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate Haig-Simons disposable income (including capital gains).

        Equations
        ---------
        .. math::
            YD_{hs}(t) = YD_r(t) + CG(t)

        Dependency
        ----------
        - state: RegularDisposableIncome, CapitalGains

        Sets
        -----
        - HaigSimonsDisposableIncome
        """
        self.state["HaigSimonsDisposableIncome"] = (
            self.state["RegularDisposableIncome"] + self.state["CapitalGains"]
        )

    def nominal_wealth(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate nominal household wealth.

        Equations
        ---------
        .. math::
            V(t) = V(t-1) + YD_{hs}(t) - C(t)

        Dependency
        ----------
        - prior: NominalWealth
        - state: HaigSimonsDisposableIncome, NominalConsumption

        Sets
        -----
        - NominalWealth
        """
        self.state["NominalWealth"] = (
            self.prior["NominalWealth"]
            + self.state["HaigSimonsDisposableIncome"]
            - self.state["NominalConsumption"]
        )

    def real_regular_disposable_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate real regular disposable income with inflation erosion.

        Equations
        ---------
        .. math::
            yd_r(t) = \frac{YD_r(t)}{p(t)} - \pi(t) \cdot \frac{V(t-1)}{p(t)}

        Dependency
        ----------
        - state: RegularDisposableIncome, PriceLevel, InflationRate
        - prior: NominalWealth

        Sets
        -----
        - RealRegularDisposableIncome
        """
        pl = self.state["PriceLevel"]
        safe_pl = torch.where(pl > 0, pl, torch.ones_like(pl))
        self.state["RealRegularDisposableIncome"] = torch.where(
            pl > 0,
            self.state["RegularDisposableIncome"] / safe_pl
            - self.state["InflationRate"] * (self.prior["NominalWealth"] / safe_pl),
            torch.zeros_like(pl),
        )

    def real_wealth(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate real household wealth.

        Equations
        ---------
        .. math::
            v(t) = V(t) / p(t)

        Dependency
        ----------
        - state: NominalWealth, PriceLevel

        Sets
        -----
        - RealWealth
        """
        pl = self.state["PriceLevel"]
        safe_pl = torch.where(pl > 0, pl, torch.ones_like(pl))
        self.state["RealWealth"] = torch.where(
            pl > 0,
            self.state["NominalWealth"] / safe_pl,
            torch.zeros_like(pl),
        )

    def expected_nominal_wealth(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate expected nominal wealth for portfolio allocation.

        Equations
        ---------
        .. math::
            V^e(t) = V(t-1) + YD^e(t) - C(t)

        Dependency
        ----------
        - prior: NominalWealth
        - state: NominalExpectedDisposableIncome, NominalConsumption

        Sets
        -----
        - ExpectedNominalWealth
        """
        self.state["ExpectedNominalWealth"] = (
            self.prior["NominalWealth"]
            + self.state["NominalExpectedDisposableIncome"]
            - self.state["NominalConsumption"]
        )

    def cash_demand(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate household cash demand proportional to consumption.

        Equations
        ---------
        .. math::
            Hh^d(t) = \lambda_c \cdot C(t)

        Dependency
        ----------
        - state: NominalConsumption
        - params: CashToConsumptionRatio

        Sets
        -----
        - CashDemand
        """
        self.state["CashDemand"] = (
            params["CashToConsumptionRatio"] * self.state["NominalConsumption"]
        )

    def expected_non_cash_wealth(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate expected non-cash wealth available for portfolio allocation.

        Equations
        ---------
        .. math::
            V^e_{nc}(t) = V^e(t) - Hh^d(t)

        Dependency
        ----------
        - state: ExpectedNominalWealth, CashDemand

        Sets
        -----
        - ExpectedNonCashWealth
        """
        self.state["ExpectedNonCashWealth"] = (
            self.state["ExpectedNominalWealth"] - self.state["CashDemand"]
        )

    ############################################################################
    # Block 9: Portfolio allocation
    ############################################################################

    def m2_demand(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate M2 demand via Tobin portfolio choice.

        Equations
        ---------
        .. math::
            M2^d(t) = V^e_{nc} \cdot (\lambda_{20} + \lambda_{21} \cdot r_m
                       + \lambda_{22} \cdot r_b + \lambda_{23} \cdot ERr_{bl})
                       + \lambda_{24} \cdot YD^e

        Dependency
        ----------
        - state: ExpectedNonCashWealth, DepositRate, BillRate,
                 ExpectedReturnOnBonds, NominalExpectedDisposableIncome
        - params: WealthShareM2_*

        Sets
        -----
        - M2Demand
        """
        self.state["M2Demand"] = (
            self.state["ExpectedNonCashWealth"]
            * (
                params["WealthShareM2_Constant"]
                + params["WealthShareM2_DepositRate"] * self.state["DepositRate"]
                + params["WealthShareM2_BillRate"] * self.state["BillRate"]
                + params["WealthShareM2_BondYield"]
                * self.state["ExpectedReturnOnBonds"]
            )
            + params["WealthShareM2_Income"]
            * self.state["NominalExpectedDisposableIncome"]
        )

    def bills_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate bills demand via Tobin portfolio choice.

        Equations
        ---------
        .. math::
            B^d(t) = V^e_{nc} \cdot (\lambda_{30} + \lambda_{31} \cdot r_m
                      + \lambda_{32} \cdot r_b + \lambda_{33} \cdot ERr_{bl})
                      + \lambda_{34} \cdot YD^e

        Dependency
        ----------
        - state: ExpectedNonCashWealth, DepositRate, BillRate,
                 ExpectedReturnOnBonds, NominalExpectedDisposableIncome
        - params: WealthShareBills_*

        Sets
        -----
        - BillsDemand
        """
        self.state["BillsDemand"] = (
            self.state["ExpectedNonCashWealth"]
            * (
                params["WealthShareBills_Constant"]
                + params["WealthShareBills_DepositRate"] * self.state["DepositRate"]
                + params["WealthShareBills_BillRate"] * self.state["BillRate"]
                + params["WealthShareBills_BondYield"]
                * self.state["ExpectedReturnOnBonds"]
            )
            + params["WealthShareBills_Income"]
            * self.state["NominalExpectedDisposableIncome"]
        )

    def bonds_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate bond demand (in number of bonds) via Tobin portfolio choice.

        Equations
        ---------
        .. math::
            BL^d(t) = \frac{V^e_{nc} \cdot (\lambda_{40} + \lambda_{41} \cdot r_m
                       + \lambda_{42} \cdot r_b + \lambda_{43} \cdot ERr_{bl})
                       + \lambda_{44} \cdot YD^e}{p_{bl}(t)}

        Dependency
        ----------
        - state: ExpectedNonCashWealth, DepositRate, BillRate,
                 ExpectedReturnOnBonds, NominalExpectedDisposableIncome, BondPrice
        - params: WealthShareBonds_*

        Sets
        -----
        - BondsDemand
        """
        value_demand = (
            self.state["ExpectedNonCashWealth"]
            * (
                params["WealthShareBonds_Constant"]
                + params["WealthShareBonds_DepositRate"] * self.state["DepositRate"]
                + params["WealthShareBonds_BillRate"] * self.state["BillRate"]
                + params["WealthShareBonds_BondYield"]
                * self.state["ExpectedReturnOnBonds"]
            )
            + params["WealthShareBonds_Income"]
            * self.state["NominalExpectedDisposableIncome"]
        )
        bp = self.state["BondPrice"]
        safe_bp = torch.where(bp > 0, bp, torch.ones_like(bp))
        self.state["BondsDemand"] = torch.where(
            bp > 0,
            value_demand / safe_bp,
            torch.zeros_like(bp),
        )

    def m1_demand_tentative(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate tentative M1 demand as the portfolio residual.

        M1 is the buffer-stock asset that absorbs whatever *actual* wealth
        remains after the three Tobin-allocated assets (M2, Bills, Bonds).
        The R sfcr reference uses actual non-cash wealth (``Vnc = V - Hhh``),
        not expected, for this residual.

        Equations
        ---------
        .. math::
            M1^{dt}(t) = V_{nc}(t) - M2^d(t) - B^d(t) - p_{bl}(t) \cdot BL^d(t)

        Dependency
        ----------
        - state: NonCashWealth, M2Demand, BillsDemand,
                 BondPrice, BondsDemand

        Sets
        -----
        - M1DemandTentative
        """
        self.state["M1DemandTentative"] = (
            self.state["NonCashWealth"]
            - self.state["M2Demand"]
            - self.state["BillsDemand"]
            - self.state["BondPrice"] * self.state["BondsDemand"]
        )

    def cash_household(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set household cash equal to cash demand (always satisfied).

        Dependency
        ----------
        - state: CashDemand

        Sets
        -----
        - CashHousehold
        """
        self.state["CashHousehold"] = self.state["CashDemand"]

    def portfolio_switches(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Compute M1/M2 portfolio switches.

        z1 = 1 if M1DemandTentative > 0 (normal case, M1 satisfies demand).
        z2 = 1 - z1 (M2 absorbs residual when M1 demand would go negative).

        Dependency
        ----------
        - state: M1DemandTentative

        Sets
        -----
        - SwitchM1Positive, SwitchM2Absorber
        """
        ref = self.state["M1DemandTentative"]
        ones = torch.ones_like(ref)
        zeros = torch.zeros_like(ref)
        self.state["SwitchM1Positive"] = torch.where(ref > 0, ones, zeros)
        self.state["SwitchM2Absorber"] = torch.where(ref > 0, zeros, ones)

    def m1_household(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate household M1 holdings.

        When z1=1 (normal), M1 = M1DemandTentative. When z2=1, M1 = 0.

        Dependency
        ----------
        - state: M1DemandTentative, SwitchM1Positive

        Sets
        -----
        - M1Household
        """
        self.state["M1Household"] = (
            self.state["M1DemandTentative"] * self.state["SwitchM1Positive"]
        )

    def m2_household(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate household M2 holdings.

        When z1=1 (normal), M2 = M2Demand. When z2=1, M2 absorbs the
        residual non-cash wealth after bills and bonds.

        Equations
        ---------
        .. math::
            M2_h(t) = M2^d \cdot z_1
                      + (V_{nc} - B_h - p_{bl} \cdot BL^d) \cdot z_2

        Dependency
        ----------
        - state: M2Demand, SwitchM1Positive, SwitchM2Absorber,
                 NonCashWealth, BillsDemand, BondPrice, BondsDemand

        Sets
        -----
        - M2Household
        """
        residual = (
            self.state["NonCashWealth"]
            - self.state["BillsDemand"]
            - self.state["BondPrice"] * self.state["BondsDemand"]
        )
        self.state["M2Household"] = (
            self.state["M2Demand"] * self.state["SwitchM1Positive"]
            + residual * self.state["SwitchM2Absorber"]
        )

    def bills_household(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set household bill holdings equal to bill demand.

        Dependency
        ----------
        - state: BillsDemand

        Sets
        -----
        - BillsHousehold
        """
        self.state["BillsHousehold"] = self.state["BillsDemand"]

    def bonds_household(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set household bond holdings equal to bond demand.

        Dependency
        ----------
        - state: BondsDemand

        Sets
        -----
        - BondsHousehold
        """
        self.state["BondsHousehold"] = self.state["BondsDemand"]

    def non_cash_wealth(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate actual non-cash wealth available for portfolio allocation.

        In the R sfcr reference, M1 (the buffer-stock residual) is computed
        against actual non-cash wealth (``Vnc = V - Hhh``), not expected.
        Portfolio *demands* (M2d, Bhd, BLd) use expected wealth (VncE), but the
        realized residual uses actual wealth so that the portfolio identity holds.

        Equations
        ---------
        .. math::
            V_{nc}(t) = V(t) - Hh^d(t)

        Dependency
        ----------
        - state: NominalWealth, CashDemand

        Sets
        -----
        - NonCashWealth
        """
        self.state["NonCashWealth"] = (
            self.state["NominalWealth"] - self.state["CashDemand"]
        )

    ############################################################################
    # Block 10: Government
    ############################################################################

    def government_spending(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate nominal government spending.

        Equations
        ---------
        .. math::
            G(t) = g(t) \cdot p(t)

        Dependency
        ----------
        - state: RealGovernmentSpending, PriceLevel

        Sets
        -----
        - GovernmentSpending
        """
        self.state["GovernmentSpending"] = (
            self.state["RealGovernmentSpending"] * self.state["PriceLevel"]
        )

    def bonds_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set bond supply equal to household bond demand.

        Equations
        ---------
        .. math::
            BL_s(t) = BL_h(t)

        Dependency
        ----------
        - state: BondsHousehold

        Sets
        -----
        - BondsSupply
        """
        self.state["BondsSupply"] = self.state["BondsHousehold"]

    def public_sector_borrowing_requirement(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the public sector borrowing requirement.

        Equations
        ---------
        .. math::
            PSBR(t) = G(t) + r_b(t-1) \cdot B_s(t-1) + BL_s(t-1)
                      - T(t) - FCB(t)

        Dependency
        ----------
        - state: GovernmentSpending, Taxes, CentralBankProfits
        - prior: BillRate, BillsSupply, BondsSupply

        Sets
        -----
        - PublicSectorBorrowingRequirement
        """
        self.state["PublicSectorBorrowingRequirement"] = (
            self.state["GovernmentSpending"]
            + self.prior["BillRate"] * self.prior["BillsSupply"]
            + self.prior["BondsSupply"]
            - self.state["Taxes"]
            - self.state["CentralBankProfits"]
        )

    def bills_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate bill supply as residual government financing.

        Equations
        ---------
        .. math::
            B_s(t) = B_s(t-1) + PSBR(t)
                     - (BL_s(t) - BL_s(t-1)) \cdot p_{bl}(t)

        Dependency
        ----------
        - prior: BillsSupply, BondsSupply
        - state: PublicSectorBorrowingRequirement, BondsSupply, BondPrice

        Sets
        -----
        - BillsSupply
        """
        self.state["BillsSupply"] = (
            self.prior["BillsSupply"]
            + self.state["PublicSectorBorrowingRequirement"]
            - (self.state["BondsSupply"] - self.prior["BondsSupply"])
            * self.state["BondPrice"]
        )

    def government_debt(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate total government debt.

        Equations
        ---------
        .. math::
            GD(t) = B_s(t) + p_{bl}(t) \cdot BL_s(t)

        Dependency
        ----------
        - state: BillsSupply, BondPrice, BondsSupply

        Sets
        -----
        - GovernmentDebt
        """
        self.state["GovernmentDebt"] = (
            self.state["BillsSupply"]
            + self.state["BondPrice"] * self.state["BondsSupply"]
        )

    ############################################################################
    # Block 11: Bank balance sheet
    ############################################################################

    def loans_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set loan supply equal to loan demand (credit is demand-determined).

        Dependency
        ----------
        - state: LoanDemand

        Sets
        -----
        - LoansSupply
        """
        self.state["LoansSupply"] = self.state["LoanDemand"]

    def m1_supply(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Set M1 supply equal to M1 household demand.

        Dependency
        ----------
        - state: M1Household

        Sets
        -----
        - M1Supply
        """
        self.state["M1Supply"] = self.state["M1Household"]

    def m2_supply(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Set M2 supply equal to M2 household demand.

        Dependency
        ----------
        - state: M2Household

        Sets
        -----
        - M2Supply
        """
        self.state["M2Supply"] = self.state["M2Household"]

    def required_reserves(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate required reserves.

        Equations
        ---------
        .. math::
            RR(t) = ro_1 \cdot M1_s(t) + ro_2 \cdot M2_s(t)

        Dependency
        ----------
        - state: M1Supply, M2Supply
        - params: ReserveRatioM1, ReserveRatioM2

        Sets
        -----
        - RequiredReserves
        """
        self.state["RequiredReserves"] = (
            params["ReserveRatioM1"] * self.state["M1Supply"]
            + params["ReserveRatioM2"] * self.state["M2Supply"]
        )

    def cash_banks_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set bank cash supply equal to required reserves.

        Dependency
        ----------
        - state: RequiredReserves

        Sets
        -----
        - CashBanksSupply
        """
        self.state["CashBanksSupply"] = self.state["RequiredReserves"]

    def bank_bills_tentative(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate tentative bank bill holdings (residual of bank balance sheet).

        Equations
        ---------
        .. math::
            B_b^T(t) = M1_s + M2_s - L_s - Hb_s

        Dependency
        ----------
        - state: M1Supply, M2Supply, LoansSupply, CashBanksSupply

        Sets
        -----
        - BillsBankTentative
        """
        self.state["BillsBankTentative"] = (
            self.state["M1Supply"]
            + self.state["M2Supply"]
            - self.state["LoansSupply"]
            - self.state["CashBanksSupply"]
        )

    def bank_liquidity_ratio_tentative(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the tentative bank liquidity ratio.

        Equations
        ---------
        .. math::
            BLR^T(t) = B_b^T(t) / (M1_s(t) + M2_s(t))

        Dependency
        ----------
        - state: BillsBankTentative, M1Supply, M2Supply

        Sets
        -----
        - BankLiquidityRatioTentative
        """
        denom = self.state["M1Supply"] + self.state["M2Supply"]
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        self.state["BankLiquidityRatioTentative"] = torch.where(
            denom > 0,
            self.state["BillsBankTentative"] / safe_denom,
            torch.zeros_like(denom),
        )

    def switch_bank_below_floor(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Compute the bank liquidity floor switch.

        z3 = 1 if BankLiquidityRatioTentative < BankLiquidityFloor.

        Dependency
        ----------
        - state: BankLiquidityRatioTentative
        - params: BankLiquidityFloor

        Sets
        -----
        - SwitchBankBelowFloor
        """
        ref = self.state["BankLiquidityRatioTentative"]
        ones = torch.ones_like(ref)
        zeros = torch.zeros_like(ref)
        self.state["SwitchBankBelowFloor"] = torch.where(
            ref < params["BankLiquidityFloor"],
            ones,
            zeros,
        )

    def advances_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate bank demand for central bank advances.

        When the tentative liquidity ratio is below floor, bank borrows from CB.

        Equations
        ---------
        .. math::
            A^d(t) = \max(bot \cdot (M1_s + M2_s) - B_b^T, 0) \cdot z_3

        Dependency
        ----------
        - state: BillsBankTentative, M1Supply, M2Supply, SwitchBankBelowFloor
        - params: BankLiquidityFloor

        Sets
        -----
        - AdvancesDemand
        """
        shortfall = (
            params["BankLiquidityFloor"]
            * (self.state["M1Supply"] + self.state["M2Supply"])
            - self.state["BillsBankTentative"]
        )
        self.state["AdvancesDemand"] = shortfall * self.state["SwitchBankBelowFloor"]

    def bank_bill_holdings(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate actual bank bill holdings after advances.

        Equations
        ---------
        .. math::
            B_b(t) = B_b^T(t) + A^d(t)

        Dependency
        ----------
        - state: BillsBankTentative, AdvancesDemand

        Sets
        -----
        - BillsBank
        """
        self.state["BillsBank"] = (
            self.state["BillsBankTentative"] + self.state["AdvancesDemand"]
        )

    def bank_liquidity_ratio(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the actual bank liquidity ratio.

        Equations
        ---------
        .. math::
            BLR(t) = B_b(t) / (M1_s(t) + M2_s(t))

        Dependency
        ----------
        - state: BillsBank, M1Supply, M2Supply

        Sets
        -----
        - BankLiquidityRatio
        """
        denom = self.state["M1Supply"] + self.state["M2Supply"]
        safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
        self.state["BankLiquidityRatio"] = torch.where(
            denom > 0,
            self.state["BillsBank"] / safe_denom,
            torch.zeros_like(denom),
        )

    def lagged_m1_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Store current M1 supply for use in profit margin calculation next period.

        Dependency
        ----------
        - state: M1Supply

        Sets
        -----
        - LaggedM1Supply
        """
        self.state["LaggedM1Supply"] = self.state["M1Supply"]

    def lagged_m2_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Store current M2 supply for use in profit margin calculation next period.

        Dependency
        ----------
        - state: M2Supply

        Sets
        -----
        - LaggedM2Supply
        """
        self.state["LaggedM2Supply"] = self.state["M2Supply"]

    ############################################################################
    # Block 12: Central bank
    ############################################################################

    def advance_rate(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set the advance rate equal to the bill rate (policy rate).

        Dependency
        ----------
        - state: BillRate

        Sets
        -----
        - AdvanceRate
        """
        self.state["AdvanceRate"] = self.state["BillRate"]

    def advances_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set advances supply equal to advances demand (CB accommodates).

        Dependency
        ----------
        - state: AdvancesDemand

        Sets
        -----
        - AdvancesSupply
        """
        self.state["AdvancesSupply"] = self.state["AdvancesDemand"]

    def bills_central_bank(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate central bank bill holdings as residual.

        Equations
        ---------
        .. math::
            B_{cb}(t) = B_s(t) - B_h(t) - B_b(t)

        Dependency
        ----------
        - state: BillsSupply, BillsHousehold, BillsBank

        Sets
        -----
        - BillsCentralBank
        """
        self.state["BillsCentralBank"] = (
            self.state["BillsSupply"]
            - self.state["BillsHousehold"]
            - self.state["BillsBank"]
        )

    def high_powered_money(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate high-powered money supply.

        Equations
        ---------
        .. math::
            H_s(t) = Hh(t) + Hb_s(t)

        Dependency
        ----------
        - state: CashHousehold, CashBanksSupply

        Sets
        -----
        - HighPoweredMoney
        """
        self.state["HighPoweredMoney"] = (
            self.state["CashHousehold"] + self.state["CashBanksSupply"]
        )

    def bank_reserves_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Store bank reserves supply (alias for CashBanksSupply).

        Dependency
        ----------
        - state: CashBanksSupply

        Sets
        -----
        - BankReservesSupply
        """
        self.state["BankReservesSupply"] = self.state["CashBanksSupply"]

    ############################################################################
    # Theoretical Steady State
    ############################################################################

    @torch.no_grad()
    def _ss_solve(self, params: dict, scenario: dict) -> dict:
        r"""Analytical steady-state solver for GL06INSOUT.

        Computes all state variables from parameters alone (no iterative
        convergence needed for real quantities).  The solver:

        1. Computes UC/p analytically from the markup identity.
        2. Expresses all portfolio quantities as linear functions of y*.
        3. Derives r_check(y*) as a rational function via Tobin portfolio
           algebra and the HPM identity (cf. step methods: bills_demand,
           bonds_demand, bills_central_bank, government_debt).
        4. Substitutes into corrected GL06 eq. 10.98 → quadratic in y*.
        5. Iterates on π via Picard: π = Ω₃·(ωᵀ(y*) − ω*(π)).

        Bank rates (r_m, r_l) are inherited from self.prior — they are
        path-dependent through the corridor mechanism (cf. deposit_rate,
        loan_rate) and converge in the outer base-class loop (~20 steps).

        Parameters
        ----------
        params : dict
            Current-period parameters.
        scenario : dict
            Current-period scenario values.

        Returns
        -------
        dict
            Complete state dictionary with all variable keys.
        """
        # ── Phase 0: Extract parameters ──────────────────────────────────────
        alpha1 = params["PropensityToConsumeIncome"]
        alpha2 = params["PropensityToConsumeWealth"]
        tau = params["TaxRate"]
        phi = params["MarkupRate"]
        pr = params["LaborProductivity"]
        N_fe = params["FullEmployment"]
        sigma0 = params["InventorySalesRatioBaseline"]
        sigma1 = params["InventorySalesRatioInterestSens"]
        lam_c = params["CashToConsumptionRatio"]
        ro1 = params["ReserveRatioM1"]
        ro2 = params["ReserveRatioM2"]
        Omega0 = params["RealWageTargetConstant"]
        Omega1 = params["RealWageProductivityElasticity"]
        Omega2 = params["RealWageEmploymentElasticity"]
        Omega3 = params["WageAdjustmentSpeed"]
        bot = params["BankLiquidityFloor"]

        # Tobin portfolio lambdas (cf. m2_demand, bills_demand, bonds_demand)
        lam20 = params["WealthShareM2_Constant"]
        lam22 = params["WealthShareM2_DepositRate"]
        lam23 = params["WealthShareM2_BillRate"]
        lam24 = params["WealthShareM2_BondYield"]
        lam25 = params["WealthShareM2_Income"]
        lam30 = params["WealthShareBills_Constant"]
        lam32 = params["WealthShareBills_DepositRate"]
        lam33 = params["WealthShareBills_BillRate"]
        lam34 = params["WealthShareBills_BondYield"]
        lam35 = params["WealthShareBills_Income"]
        lam40 = params["WealthShareBonds_Constant"]
        lam42 = params["WealthShareBonds_DepositRate"]
        lam43 = params["WealthShareBonds_BillRate"]
        lam44 = params["WealthShareBonds_BondYield"]
        lam45 = params["WealthShareBonds_Income"]

        # Exogenous scenario values (cf. set_bill_rate, etc.)
        g = scenario["RealGovernmentSpending"]
        r_b = scenario["BillRate"]
        r_bl = scenario["BondYield"]

        # Derived constants
        safe_a2 = torch.where(alpha2 > 0, alpha2, torch.ones_like(alpha2))
        alpha3 = torch.where(
            alpha2 > 0,
            (1.0 - alpha1) / safe_a2,
            torch.zeros_like(alpha2),
        )

        # Bond price (cf. bond_price): p_bl = 1/r_bl
        safe_rbl = torch.where(r_bl > 0, r_bl, torch.ones_like(r_bl))
        p_bl = torch.where(r_bl > 0, 1.0 / safe_rbl, torch.zeros_like(r_bl))
        ERr_bl = r_bl  # static expectations (cf. expected_return_on_bonds)

        # ── Phase 1: Bank rates and inventory ratio ──────────────────────────
        # Bank rates are path-dependent (corridor mechanism). Read from
        # self.state — the wrapper calls the actual deposit_rate/loan_rate
        # methods before _ss_solve so they converge via the outer loop.
        r_m = self.state["DepositRate"]
        r_l = self.state["LoanRate"]
        r_a = r_b  # advance rate = bill rate (cf. advance_rate)

        # Target inventory-sales ratio (cf. target_inventory_sales_ratio)
        sigmaT = sigma0 - sigma1 * r_l

        # ── Phase 2: Picard iteration on (y*, π) ────────────────────────────
        pi = torch.zeros_like(g)
        max_iter = 50
        tol_pi = 1e-10

        for _picard_iter in range(max_iter):
            pi_old = pi.clone()

            # UC/p(π): analytical from markup identity
            # At SS: NHUC = UC·[(1-σᵀ) + σᵀ·(1+r_l)/(1+π)]
            # p = (1+τ)(1+φ)·NHUC, so UC/p = 1/((1+τ)(1+φ)·nhuc_factor)
            # (cf. normal_historic_unit_cost, price_level)
            safe_pi = torch.clamp(1.0 + pi, min=0.5)
            nhuc_factor = (1.0 - sigmaT) + sigmaT * (1.0 + r_l) / safe_pi
            uc_over_p = 1.0 / ((1.0 + tau) * (1.0 + phi) * nhuc_factor)

            # ── Portfolio ξ coefficients (p-normalized, linear in c*) ────
            # V_nc/f = α₃ − λ_c where f = c*·p  (cf. cash_demand)
            vnc_frac = alpha3 - lam_c

            # Portfolio share constants at current rates
            share_B = lam30 + lam32 * r_m + lam33 * r_b + lam34 * ERr_bl
            share_BL = lam40 + lam42 * r_m + lam43 * r_b + lam44 * ERr_bl
            share_M2 = lam20 + lam22 * r_m + lam23 * r_b + lam24 * ERr_bl

            # ξ: fraction of f = c*·p allocated to each asset
            # (cf. bills_demand, bonds_demand, m2_demand, m1_demand_tentative)
            # At SS: YD_e = f·(1+π·α₃) because yd_r = c* but
            # YD_r/p = c* + π·v* = c*·(1+π·α₃) (cf. eq 10.32, 10.26)
            yde_factor = 1.0 + pi * alpha3
            xi_B = vnc_frac * share_B + lam35 * yde_factor
            xi_BL = vnc_frac * share_BL + lam45 * yde_factor
            xi_M2 = vnc_frac * share_M2 + lam25 * yde_factor
            xi_M1 = vnc_frac - xi_M2 - xi_B - xi_BL  # buffer stock residual

            # Portfolio switch: when M1 demand < 0, M1 = 0 and M2 absorbs
            # residual (cf. portfolio_switches, m1_household, m2_household).
            # This changes bank BS and HPM coefficients.
            switch = xi_M1 < 0
            xi_M1 = torch.where(switch, torch.zeros_like(xi_M1), xi_M1)
            xi_M2 = torch.where(switch, vnc_frac - xi_B - xi_BL, xi_M2)

            # ── Bank balance sheet (p-normalized) ────────────────────────
            # B_b/p = μ₁·y* + μ₀
            # (cf. loans_supply, required_reserves, bank_bills_tentative)
            mu1 = (1.0 - ro1) * xi_M1 + (1.0 - ro2) * xi_M2 - sigmaT * uc_over_p
            mu0 = -g * ((1.0 - ro1) * xi_M1 + (1.0 - ro2) * xi_M2)

            # B_cb/p = κ·c* via HPM identity (A_s = 0 assumption)
            # H_s = Hh + Hb = (λ_c + ρ₁·ξ_M1 + ρ₂·ξ_M2)·c*·p
            # (cf. bills_central_bank, high_powered_money)
            kappa = lam_c + ro1 * xi_M1 + ro2 * xi_M2

            # B_s/p = B_h/p + B_b/p + B_cb/p = χ_s·y* + χ₀
            # (cf. bill market clearing: B_s = B_h + B_b + B_cb)
            chi_s = xi_B + kappa + mu1
            chi_0 = -g * (xi_B + kappa) + mu0

            # ── r_check as rational function of y* ───────────────────────
            # DS/p = c*·(r_b·ξ_B + r_bl·ξ_BL)  [debt service to households]
            # (cf. regular_disposable_income: r_b·B_h + BL_h coupon)
            ds = r_b * xi_B + r_bl * xi_BL
            a_n = ds
            b_n = -g * ds

            # GD/p = B_s/p + c*·ξ_BL  (cf. government_debt: B_s + p_bl·BL_s)
            a_d = chi_s + xi_BL
            b_d = chi_0 - g * xi_BL

            # ── Quadratic coefficients ───────────────────────────────────
            # Eq. 10.98: y* = g·(1 − α₃·R) / (T − R·Q)
            # where R = r_check − π, T = τ/(1+τ), Q = α₃ − σᵀ·UC/p
            # Cross-multiplying with r_check = (a_n·y*+b_n)/(a_d·y*+b_d)
            # yields A·y*² + B·y* + C = 0
            T_tax = tau / (1.0 + tau)
            Q_inv = alpha3 - sigmaT * uc_over_p

            a_R = a_n - pi * a_d
            b_R = b_n - pi * b_d

            A_coeff = T_tax * a_d - Q_inv * a_R
            B_coeff = T_tax * b_d - Q_inv * b_R - g * (a_d - alpha3 * a_R)
            C_coeff = -g * (b_d - alpha3 * b_R)

            # ── Solve quadratic ──────────────────────────────────────────
            discriminant = B_coeff**2 - 4.0 * A_coeff * C_coeff
            safe_disc = torch.clamp(discriminant, min=0.0)
            sqrt_disc = torch.sqrt(safe_disc)

            safe_A = torch.where(
                A_coeff.abs() > 1e-12,
                A_coeff,
                torch.ones_like(A_coeff) * 1e-12,
            )
            y1 = (-B_coeff + sqrt_disc) / (2.0 * safe_A)
            y2 = (-B_coeff - sqrt_disc) / (2.0 * safe_A)

            # Select positive root with c* = y* − g > 0.
            # When A_coeff < 0 (large α₃, e.g. Scenario 5), the +√ root
            # is the smaller one.  Always prefer the larger valid root.
            valid1 = (y1 > g) & (y1 > 0)
            valid2 = (y2 > g) & (y2 > 0)
            y_star = torch.where(
                valid1 & valid2,
                torch.max(y1, y2),
                torch.where(
                    valid1,
                    y1,
                    torch.where(valid2, y2, torch.clamp(torch.max(y1, y2), min=0.0)),
                ),
            )

            # ── Inflation update ─────────────────────────────────────────
            # ω* = pr·UC/p  [markup-implied real wage] (cf. real_wage, unit_cost)
            # ωᵀ = exp(Ω₀ + Ω₁·log(pr) + Ω₂·log(N*/N_fe))
            #      (cf. target_real_wage)
            # π = Ω₃·(ωᵀ − ω*)  [from nominal_wage at SS: W(t)/W(t-1) = 1+π]
            omega_star = uc_over_p * pr
            N_star = y_star / pr
            safe_ratio = torch.clamp(N_star / N_fe, min=1e-10)
            omega_T = torch.exp(
                Omega0
                + Omega1 * torch.log(torch.clamp(pr, min=1e-10))
                + Omega2 * torch.log(safe_ratio)
            )
            pi_raw = Omega3 * (omega_T - omega_star)
            pi_raw = torch.clamp(pi_raw, min=-0.5)
            # Damped update: undamped Picard has spectral radius > 1
            # when α₃ is large (e.g. Scenario 5, α₃=4).
            pi = 0.7 * pi + 0.3 * pi_raw

            # Check convergence
            pi_err = (pi - pi_old).abs().max()
            if pi_err.item() < tol_pi:
                break

        # ── Phase 3: Derive all state variables from (y*, π) ────────────────
        c_star = y_star - g
        inv_star = sigmaT * y_star

        # Nominal anchor: W from prior + inflation adjustment
        # (cf. nominal_wage; model has unit root in nominal level)
        W = self.prior["NominalWage"] * (1.0 + pi)
        UC = W / pr  # cf. unit_cost
        NHUC = UC * nhuc_factor  # cf. normal_historic_unit_cost
        p = (1.0 + tau) * (1.0 + phi) * NHUC  # cf. price_level

        # Common nominal factor
        f = c_star * p

        # ── Build output dictionary ──────────────────────────────────────────
        ss: dict[str, torch.Tensor] = {}
        zero = torch.zeros_like(g)

        # Block 0: Exogenous (cf. set_bill_rate, set_bond_yield, etc.)
        ss["RealGovernmentSpending"] = g
        ss["BillRate"] = r_b
        ss["BondYield"] = r_bl

        # Block 2: Bond pricing (cf. bond_price, expected_return_on_bonds)
        ss["BondPrice"] = p_bl
        ss["ExpectedReturnOnBonds"] = ERr_bl

        # Block 3: Firm output and inventory (cf. real_output, real_sales, etc.)
        ss["TargetInventorySalesRatio"] = sigmaT
        ss["RealOutput"] = y_star
        ss["RealSales"] = y_star  # s* = y* at SS
        ss["RealConsumption"] = c_star
        ss["RealInventories"] = inv_star
        ss["TargetInventories"] = inv_star
        ss["ExpectedInventories"] = inv_star
        ss["ExpectedSales"] = y_star
        ss["ActualInventorySalesRatio"] = sigmaT

        # Block 6: Wages, prices, employment
        # (cf. employment, nominal_wage, wage_bill, unit_cost, price_level, etc.)
        ss["Employment"] = N_star
        ss["NominalWage"] = W
        WB = N_star * W
        ss["WageBill"] = WB
        ss["UnitCost"] = UC
        ss["NormalHistoricUnitCost"] = NHUC
        ss["PriceLevel"] = p
        ss["InflationRate"] = pi
        ss["RealWage"] = W / p
        ss["TargetRealWage"] = omega_T

        # Block 7: Nominal aggregates (cf. nominal_consumption, taxes, etc.)
        C_nom = c_star * p
        S_nom = y_star * p  # s* = y* at SS
        G_nom = g * p
        INV_nom = inv_star * UC
        ss["NominalConsumption"] = C_nom
        ss["NominalSales"] = S_nom
        ss["GovernmentSpending"] = G_nom
        TX = tau / (1.0 + tau) * S_nom  # cf. taxes
        ss["Taxes"] = TX
        ss["NominalInventories"] = INV_nom
        ss["LoanDemand"] = INV_nom  # cf. loan_demand: L^d = INV
        ss["NominalOutput"] = S_nom  # at SS, inv change = 0

        # Firm profits: S − TX − WB − r_l·L_s  (cf. firm_profits; inv change=0)
        L_s = INV_nom  # cf. loans_supply: L_s = L^d
        FP = S_nom - TX - WB - r_l * L_s
        ss["FirmProfits"] = FP

        # Block 8: Household income and wealth
        # (cf. regular_disposable_income, nominal_wealth, etc.)
        V_star = alpha3 * f  # V* = α₃·c*·p
        v_star = alpha3 * c_star  # real wealth

        # Portfolio asset holdings (from ξ coefficients)
        B_h = f * xi_B  # cf. bills_household
        BL_h_val = f * xi_BL  # bond value = BL_h·p_bl
        BL_h = BL_h_val * r_bl  # number of bonds (cf. bonds_demand)
        M2_h = f * xi_M2  # cf. m2_household
        M1_h = f * xi_M1  # cf. m1_household
        Hh = lam_c * C_nom  # cf. cash_demand

        # Bank balance sheet (cf. loans_supply through bank_liquidity_ratio)
        M1_s = M1_h
        M2_s = M2_h
        RR = ro1 * M1_s + ro2 * M2_s  # cf. required_reserves
        Hb = RR  # cf. cash_banks_supply
        B_b_tent = M1_s + M2_s - L_s - Hb  # cf. bank_bills_tentative

        # Bank profits at SS (cf. bank_profits)
        # FBP = r_l·L_s + r_b·B_b − r_m·M2_s − r_a·A_s
        A_s = zero  # A_s = 0 assumption
        B_b = B_b_tent + A_s  # cf. bank_bill_holdings
        FBP = r_l * L_s + r_b * B_b - r_m * M2_s - r_a * A_s
        ss["BankProfits"] = FBP

        # Bank profit margin: FBP/(M1+M2) at SS (cf. bank_profit_margin)
        dep_base = M1_s + M2_s
        safe_dep = torch.where(dep_base > 0, dep_base, torch.ones_like(dep_base))
        BPM = torch.where(dep_base > 0, FBP / safe_dep, zero)
        ss["BankProfitMargin"] = BPM

        # Total dividends (cf. total_dividends: FD = FP + FBP)
        FD = FP + FBP
        ss["TotalDividends"] = FD

        # Capital gains = 0 (constant bond price at SS; cf. capital_gains)
        ss["CapitalGains"] = zero

        # Disposable income: YD_r = WB + FD + r_m·M2 + r_b·B_h + BL_h
        # (cf. regular_disposable_income)
        YD_r = WB + FD + r_m * M2_h + r_b * B_h + BL_h
        ss["RegularDisposableIncome"] = YD_r
        ss["HaigSimonsDisposableIncome"] = YD_r  # no CG at SS

        ss["NominalWealth"] = V_star

        # Real disposable income: yd_r = YD_r/p − π·V/p
        # (cf. real_regular_disposable_income)
        safe_p = torch.where(p > 0, p, torch.ones_like(p))
        yd_r_real = torch.where(
            p > 0,
            YD_r / safe_p - pi * V_star / safe_p,
            zero,
        )
        ss["RealRegularDisposableIncome"] = yd_r_real
        ss["RealWealth"] = v_star

        # Expected income/wealth (converged at SS)
        ss["ExpectedRealDisposableIncome"] = yd_r_real
        # YD_e = f·(1+π·α₃): nominal income includes inflation erosion
        # of wealth (cf. eq 10.32: YD_r^e = p·yd_r^e + π·V/p)
        ss["NominalExpectedDisposableIncome"] = f * (1.0 + pi * alpha3)
        ss["ExpectedNominalWealth"] = V_star

        # Portfolio details (cf. cash_demand through bonds_household)
        ss["CashDemand"] = Hh
        V_nc = V_star - Hh
        ss["ExpectedNonCashWealth"] = V_nc
        ss["NonCashWealth"] = V_nc

        ss["M2Demand"] = M2_h
        ss["BillsDemand"] = B_h
        ss["BondsDemand"] = BL_h
        M1_tent = V_nc - M2_h - B_h - BL_h_val
        ss["M1DemandTentative"] = M1_tent

        # Portfolio switches (cf. portfolio_switches)
        z1 = torch.where(M1_tent > 0, torch.ones_like(M1_tent), zero)
        z2 = 1.0 - z1
        ss["SwitchM1Positive"] = z1
        ss["SwitchM2Absorber"] = z2

        ss["CashHousehold"] = Hh
        ss["M1Household"] = M1_h
        ss["M2Household"] = M2_h
        ss["BillsHousehold"] = B_h
        ss["BondsHousehold"] = BL_h

        # Block 1: Bank rates (cf. deposit_rate, loan_rate, advance_rate)
        ss["DepositRate"] = r_m
        ss["LoanRate"] = r_l
        ss["AdvanceRate"] = r_a

        # Block 10: Government (cf. government_spending through government_debt)
        ss["BondsSupply"] = BL_h  # BL_s = BL_h at SS
        B_cb = Hh + Hb  # HPM identity with A_s = 0
        B_s = B_h + B_b + B_cb  # bill market clearing
        ss["BillsSupply"] = B_s
        ss["PublicSectorBorrowingRequirement"] = zero
        GD = B_s + p_bl * BL_h  # cf. government_debt
        ss["GovernmentDebt"] = GD

        # Central bank profits: FCB = r_b·B_cb (+ r_a·A_s = 0)
        # (cf. central_bank_profits)
        FCB = r_b * B_cb
        ss["CentralBankProfits"] = FCB

        # Block 11: Bank balance sheet details
        ss["LoansSupply"] = L_s
        ss["M1Supply"] = M1_s
        ss["M2Supply"] = M2_s
        ss["RequiredReserves"] = RR
        ss["CashBanksSupply"] = Hb
        ss["BillsBankTentative"] = B_b_tent

        BLR_tent = B_b_tent / torch.clamp(dep_base, min=1e-10)
        ss["BankLiquidityRatioTentative"] = BLR_tent
        z3 = torch.where(BLR_tent < bot, torch.ones_like(BLR_tent), zero)
        ss["SwitchBankBelowFloor"] = z3
        ss["AdvancesDemand"] = A_s
        ss["BillsBank"] = B_b
        BLR = B_b / torch.clamp(dep_base, min=1e-10)
        ss["BankLiquidityRatio"] = BLR
        ss["LaggedM1Supply"] = M1_s  # at SS, lag = current
        ss["LaggedM2Supply"] = M2_s

        # Block 12: Central bank (cf. advances_supply through bank_reserves)
        ss["AdvancesSupply"] = A_s
        ss["BillsCentralBank"] = B_cb
        H_s = Hh + Hb
        ss["HighPoweredMoney"] = H_s
        ss["BankReservesSupply"] = Hb

        # ── Warnings ─────────────────────────────────────────────────────────
        if A_s.abs().max().item() > 0 or z3.max().item() > 0:
            warnings.warn(
                "SS solver assumes A_s = 0 (bank within liquidity corridor). "
                "Banking corridor may be binding — results approximate.",
                stacklevel=2,
            )
        denom_check = (a_d * y_star + b_d).abs().min().item()
        if denom_check < 1e-6:
            warnings.warn(
                f"Small GD denominator in r_check ({denom_check:.4f}). "
                "SS formula may be numerically unstable.",
                stacklevel=2,
            )

        return ss

    def compute_theoretical_steady_state_per_step(
        self, t: int, params: dict, scenario: dict
    ):
        r"""Compute steady-state values for one timestep.

        Runs bank-rate blocks (0-2) first so that the corridor mechanism
        updates r_m and r_l in ``self.state``, then calls :meth:`_ss_solve`
        for all remaining variables.  The base-class outer loop calls this
        at each timestep, updating ``self.prior`` between steps.

        Parameters
        ----------
        t : int
            Current timestep index.
        params : dict
            Current-period parameters.
        scenario : dict
            Current-period scenario values.
        """
        kwargs = dict(t=t, params=params, scenario=scenario)

        # Block 0: Exogenous scenario variables
        self.set_bill_rate(**kwargs)
        self.set_bond_yield(**kwargs)
        self.set_real_government_spending(**kwargs)

        # Block 1: Bank profits → profit margin → endogenous rates
        # These read from self.prior and write updated r_m, r_l to self.state.
        self.bank_profits(**kwargs)
        self.bank_profit_margin(**kwargs)
        self.deposit_rate(**kwargs)
        self.loan_rate(**kwargs)
        self.central_bank_profits(**kwargs)

        # Block 2: Bond pricing
        self.bond_price(**kwargs)
        self.expected_return_on_bonds(**kwargs)

        # Analytical solver reads r_m, r_l from self.state (set above)
        ss = self._ss_solve(params, scenario)

        # Hard assert: returned keys must match state keys
        state_keys = set(self.state.keys())
        ss_keys = set(ss.keys())
        missing = state_keys - ss_keys
        extra = ss_keys - state_keys
        assert not missing, f"_ss_solve missing keys: {missing}"
        assert not extra, f"_ss_solve extra keys: {extra}"

        for key, val in ss.items():
            self.state[key] = val
