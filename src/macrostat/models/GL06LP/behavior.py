"""
This module defines the forward and simulate behavior of the
Godley-Lavoie 2006 LP model (Long-term bonds, Capital gains,
and Liquidity preference).
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import torch

from macrostat.core.behavior import Behavior
from macrostat.models.GL06LP.parameters import ParametersGL06LP
from macrostat.models.GL06LP.scenarios import ScenariosGL06LP
from macrostat.models.GL06LP.variables import VariablesGL06LP

logger = logging.getLogger(__name__)


class BehaviorGL06LP(Behavior):
    """Behavior class for the Godley-Lavoie 2006 LP model.

    Implements Model LP from Chapter 5 of Godley & Lavoie (2006),
    introducing long-term bonds, capital gains, and portfolio choice
    across cash, bills, and bonds.
    """

    version = "GL06LP"

    def __init__(
        self,
        parameters: ParametersGL06LP | None = None,
        scenarios: ScenariosGL06LP | None = None,
        variables: VariablesGL06LP | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        """Initialize the behavior of the Godley-Lavoie 2006 LP model.

        Parameters
        ----------
        parameters : ParametersGL06LP | None
            The parameters of the model.
        scenarios : ScenariosGL06LP | None
            The scenarios of the model.
        variables : VariablesGL06LP | None
            The variables of the model.
        scenario : int
            The scenario to use for the model.
        debug : bool
            Whether to enable debug mode.
        """

        if parameters is None:
            parameters = ParametersGL06LP()
        if scenarios is None:
            scenarios = ScenariosGL06LP()
        if variables is None:
            variables = VariablesGL06LP()

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
        r"""Initialize the behavior of the Godley-Lavoie 2006 LP model.

        All non-scenario variables are set to zero, consistent with the
        standard SFC initialization approach.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                C(0) = G(0) = Y(0) = T(0) = YD_r(0) = 0 \\
                V(0) = H_h(0) = H_s(0) = B_h(0) = B_s(0) = 0 \\
                B_{CB}(0) = BL_h(0) = BL_s(0) = 0 \\
                r_b(0) = p_{bl}(0) = r_{bl}(0) = 0
            \end{align}

        Sets
        -----
        All model variables to zero.
        """
        # Flows
        self.state["ConsumptionHousehold"] = torch.zeros(1)
        self.state["ConsumptionGovernment"] = torch.zeros(1)
        self.state["NationalIncome"] = torch.zeros(1)
        self.state["Taxes"] = torch.zeros(1)
        self.state["InterestOnBillsHousehold"] = torch.zeros(1)
        self.state["BondCouponIncomeHousehold"] = torch.zeros(1)
        self.state["CentralBankProfits"] = torch.zeros(1)
        self.state["CapitalGains"] = torch.zeros(1)
        self.state["ExpectedCapitalGains"] = torch.zeros(1)
        # Stocks
        self.state["Wealth"] = torch.zeros(1)
        self.state["HouseholdBillStock"] = torch.zeros(1)
        self.state["GovernmentBillStock"] = torch.zeros(1)
        self.state["CentralBankBillStock"] = torch.zeros(1)
        self.state["HouseholdBondStock"] = torch.zeros(1)
        self.state["GovernmentBondSupply"] = torch.zeros(1)
        self.state["HouseholdCashStock"] = torch.zeros(1)
        self.state["CentralBankMoneyStock"] = torch.zeros(1)
        # Indices
        self.state["DisposableIncome"] = torch.zeros(1)
        self.state["ExpectedDisposableIncome"] = torch.zeros(1)
        self.state["ExpectedWealth"] = torch.zeros(1)
        self.state["HouseholdBillDemand"] = torch.zeros(1)
        self.state["HouseholdBondDemand"] = torch.zeros(1)
        self.state["HouseholdCashDemand"] = torch.zeros(1)
        self.state["InterestRateBills"] = torch.zeros(1)
        self.state["BondPrice"] = torch.zeros(1)
        self.state["BondYield"] = torch.zeros(1)
        self.state["ExpectedBondPrice"] = torch.zeros(1)
        self.state["ExpectedReturnOnBonds"] = torch.zeros(1)

    ############################################################################
    # Step
    ############################################################################

    def step(self, **kwargs):
        """Step function of the Godley-Lavoie 2006 LP model."""

        # 1. Scenario exogenous variables
        self.set_government_demand(**kwargs)
        self.set_interest_rate_bills(**kwargs)
        self.set_bond_price(**kwargs)

        # 2. Bond yield and expected return
        self.bond_yield(**kwargs)
        self.expected_bond_price(**kwargs)
        self.expected_return_on_bonds(**kwargs)

        # 3. Prior-based flows
        self.interest_on_bills_household(**kwargs)
        self.bond_coupon_income_household(**kwargs)
        self.capital_gains(**kwargs)
        self.central_bank_profits(**kwargs)

        # 4. Expected disposable income
        self.expected_disposable_income(**kwargs)

        # 5. Consumption and income
        self.consumption(**kwargs)
        self.national_income(**kwargs)
        self.taxes(**kwargs)
        self.disposable_income(**kwargs)

        # 6. Wealth
        self.wealth(**kwargs)
        self.expected_wealth(**kwargs)

        # 7. Portfolio demands and holdings
        self.household_bill_demand(**kwargs)
        self.household_bond_demand(**kwargs)
        self.household_bill_holdings(**kwargs)
        self.household_bond_holdings(**kwargs)
        self.household_cash_stock(**kwargs)
        self.household_cash_demand(**kwargs)

        # 8. Government and central bank balance sheet
        self.government_bond_supply(**kwargs)
        self.government_bill_issuance(**kwargs)
        self.central_bank_bill_holdings(**kwargs)
        self.central_bank_money_stock(**kwargs)

        # 9. Diagnostic outputs
        self.expected_capital_gains(**kwargs)

    ############################################################################
    # Scenario Exogenous Variables
    ############################################################################

    def set_government_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set the government demand from the scenario.

        Parameters
        ----------
        t : int
            The time step.
        scenario : dict
            The scenario.
        params : dict | None
            The parameters.

        Dependency
        ----------
        - scenario: GovernmentDemand

        Sets
        -----
        - ConsumptionGovernment
        """
        self.state["ConsumptionGovernment"] = scenario["GovernmentDemand"]

    def set_interest_rate_bills(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set the interest rate on bills from the scenario.

        Parameters
        ----------
        t : int
            The time step.
        scenario : dict
            The scenario.
        params : dict | None
            The parameters.

        Dependency
        ----------
        - scenario: InterestRateBills

        Sets
        -----
        - InterestRateBills
        """
        self.state["InterestRateBills"] = scenario["InterestRateBills"]

    def set_bond_price(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set the bond price from the scenario.

        Parameters
        ----------
        t : int
            The time step.
        scenario : dict
            The scenario.
        params : dict | None
            The parameters.

        Dependency
        ----------
        - scenario: BondPrice

        Sets
        -----
        - BondPrice
        """
        self.state["BondPrice"] = scenario["BondPrice"]

    ############################################################################
    # Bond Yield and Expected Return
    ############################################################################

    def bond_yield(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the bond yield (coupon / price).

        Since the coupon is 1 per bond, the yield is simply the inverse
        of the bond price.

        Equations
        ---------
        .. math::
            r_{bl}(t) = \frac{1}{p_{bl}(t)}

        Dependency
        ----------
        - state: BondPrice

        Sets
        -----
        - BondYield
        """
        self.state["BondYield"] = torch.where(
            self.state["BondPrice"] > 0,
            1.0 / self.state["BondPrice"],
            torch.zeros_like(self.state["BondPrice"]),
        )

    def expected_bond_price(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the expected bond price.

        In the base LP model, agents have static expectations: the
        expected bond price equals the current bond price. This method
        is designed to be overridden in LP2/LP3 for adaptive expectations.

        Equations
        ---------
        .. math::
            p_{bl}^e(t) = p_{bl}(t)

        Dependency
        ----------
        - state: BondPrice

        Sets
        -----
        - ExpectedBondPrice
        """
        self.state["ExpectedBondPrice"] = self.state["BondPrice"]

    def expected_return_on_bonds(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the expected return on bonds.

        The expected return combines the current yield with the expected
        capital gain from bond price changes.

        Equations
        ---------
        .. math::
            ERr_{bl}(t) = r_{bl}(t)
                + \chi \cdot \frac{p_{bl}^e(t) - p_{bl}(t)}{p_{bl}(t)}

        Dependency
        ----------
        - state: BondYield
        - state: ExpectedBondPrice
        - state: BondPrice
        - params: ExpectationWeightBondPrice

        Sets
        -----
        - ExpectedReturnOnBonds
        """
        self.state["ExpectedReturnOnBonds"] = self.state["BondYield"] + (
            params["ExpectationWeightBondPrice"]
            * torch.where(
                self.state["BondPrice"] > 0,
                (self.state["ExpectedBondPrice"] - self.state["BondPrice"])
                / self.state["BondPrice"],
                torch.zeros_like(self.state["BondPrice"]),
            )
        )

    ############################################################################
    # Prior-Based Flows
    ############################################################################

    def interest_on_bills_household(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the interest earned on bills by the household.

        Equations
        ---------
        .. math::
            r_b(t-1) \cdot B_h(t-1)

        Dependency
        ----------
        - prior: InterestRateBills
        - prior: HouseholdBillStock

        Sets
        -----
        - InterestOnBillsHousehold
        """
        self.state["InterestOnBillsHousehold"] = (
            self.prior["InterestRateBills"] * self.prior["HouseholdBillStock"]
        )

    def bond_coupon_income_household(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the coupon income from bonds held by the household.

        Each bond pays a coupon of 1 monetary unit per period, so bond
        coupon income is simply the number of bonds held last period.

        Equations
        ---------
        .. math::
            BL_h(t-1)

        Dependency
        ----------
        - prior: HouseholdBondStock

        Sets
        -----
        - BondCouponIncomeHousehold
        """
        self.state["BondCouponIncomeHousehold"] = self.prior["HouseholdBondStock"]

    def capital_gains(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the capital gains on bonds.

        Capital gains arise from changes in the bond price applied to the
        stock of bonds held at the beginning of the period.

        Equations
        ---------
        .. math::
            CG(t) = \left(p_{bl}(t) - p_{bl}(t-1)\right) \cdot BL_h(t-1)

        Dependency
        ----------
        - state: BondPrice
        - prior: BondPrice
        - prior: HouseholdBondStock

        Sets
        -----
        - CapitalGains
        """
        self.state["CapitalGains"] = (
            self.state["BondPrice"] - self.prior["BondPrice"]
        ) * self.prior["HouseholdBondStock"]

    def central_bank_profits(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the central bank profits (income on bills held).

        Equations
        ---------
        .. math::
            r_b(t-1) \cdot B_{CB}(t-1)

        Dependency
        ----------
        - prior: InterestRateBills
        - prior: CentralBankBillStock

        Sets
        -----
        - CentralBankProfits
        """
        self.state["CentralBankProfits"] = (
            self.prior["InterestRateBills"] * self.prior["CentralBankBillStock"]
        )

    ############################################################################
    # Expected Disposable Income
    ############################################################################

    def expected_disposable_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the expected disposable income.

        Expectations are adaptive: expected disposable income equals the
        prior period's actual disposable income.

        Equations
        ---------
        .. math::
            YD_r^e(t) = YD_r(t-1)

        Dependency
        ----------
        - prior: DisposableIncome

        Sets
        -----
        - ExpectedDisposableIncome
        """
        self.state["ExpectedDisposableIncome"] = self.prior["DisposableIncome"]

    ############################################################################
    # Consumption, Income, Taxes
    ############################################################################

    def consumption(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate consumption.

        Equations
        ---------
        .. math::
            C(t) = \alpha_1 \cdot YD_r^e(t) + \alpha_2 \cdot V(t-1)

        Dependency
        ----------
        - state: ExpectedDisposableIncome
        - prior: Wealth
        - params: PropensityToConsumeIncome
        - params: PropensityToConsumeSavings

        Sets
        -----
        - ConsumptionHousehold
        """
        self.state["ConsumptionHousehold"] = (
            params["PropensityToConsumeIncome"] * self.state["ExpectedDisposableIncome"]
            + params["PropensityToConsumeSavings"] * self.prior["Wealth"]
        )

    def national_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the national income.

        Equations
        ---------
        .. math::
            Y(t) = C(t) + G(t)

        Dependency
        ----------
        - state: ConsumptionHousehold
        - state: ConsumptionGovernment

        Sets
        -----
        - NationalIncome
        """
        self.state["NationalIncome"] = (
            self.state["ConsumptionHousehold"] + self.state["ConsumptionGovernment"]
        )

    def taxes(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the taxes.

        The tax base includes national income, interest on bills, and
        bond coupon income.

        Equations
        ---------
        .. math::
            T(t) = \theta \cdot \left(Y(t) + r_b(t-1) \cdot B_h(t-1)
                + BL_h(t-1)\right)

        Dependency
        ----------
        - params: TaxRate
        - state: NationalIncome
        - state: InterestOnBillsHousehold
        - state: BondCouponIncomeHousehold

        Sets
        -----
        - Taxes
        """
        self.state["Taxes"] = params["TaxRate"] * (
            self.state["NationalIncome"]
            + self.state["InterestOnBillsHousehold"]
            + self.state["BondCouponIncomeHousehold"]
        )

    def disposable_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the disposable income.

        Disposable income includes national income minus taxes plus
        interest on bills and bond coupon income.

        Equations
        ---------
        .. math::
            YD_r(t) = Y(t) - T(t) + r_b(t-1) \cdot B_h(t-1) + BL_h(t-1)

        Dependency
        ----------
        - state: NationalIncome
        - state: Taxes
        - state: InterestOnBillsHousehold
        - state: BondCouponIncomeHousehold

        Sets
        -----
        - DisposableIncome
        """
        self.state["DisposableIncome"] = (
            self.state["NationalIncome"]
            - self.state["Taxes"]
            + self.state["InterestOnBillsHousehold"]
            + self.state["BondCouponIncomeHousehold"]
        )

    ############################################################################
    # Wealth
    ############################################################################

    def wealth(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the wealth.

        Wealth evolves as savings plus capital gains on bonds.

        Equations
        ---------
        .. math::
            V(t) = V(t-1) + (YD_r(t) - C(t)) + CG(t)

        Dependency
        ----------
        - prior: Wealth
        - state: DisposableIncome
        - state: ConsumptionHousehold
        - state: CapitalGains

        Sets
        -----
        - Wealth
        """
        self.state["Wealth"] = (
            self.prior["Wealth"]
            + self.state["DisposableIncome"]
            - self.state["ConsumptionHousehold"]
            + self.state["CapitalGains"]
        )

    def expected_wealth(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the expected wealth.

        Expected wealth uses expected disposable income instead of
        actual disposable income, but actual capital gains (which are
        known at the start of the period since bond prices are
        exogenous).

        Equations
        ---------
        .. math::
            V^e(t) = V(t-1) + (YD_r^e(t) - C(t)) + CG(t)

        Dependency
        ----------
        - prior: Wealth
        - state: ExpectedDisposableIncome
        - state: ConsumptionHousehold
        - state: CapitalGains

        Sets
        -----
        - ExpectedWealth
        """
        self.state["ExpectedWealth"] = (
            self.prior["Wealth"]
            + self.state["ExpectedDisposableIncome"]
            - self.state["ConsumptionHousehold"]
            + self.state["CapitalGains"]
        )

    ############################################################################
    # Portfolio Demands
    ############################################################################

    def household_bill_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the household bill demand using Tobin portfolio choice.

        Equations
        ---------
        .. math::
            B_d(t) = V^e(t) \cdot \lambda_{20}
                + V^e(t) \cdot (\lambda_{22} \cdot r_b(t)
                + \lambda_{23} \cdot ERr_{bl}(t))
                + \lambda_{24} \cdot YD_r^e(t)

        Dependency
        ----------
        - state: ExpectedWealth
        - state: InterestRateBills
        - state: ExpectedReturnOnBonds
        - state: ExpectedDisposableIncome
        - params: WealthShareBills_Constant
        - params: WealthShareBills_BillRate
        - params: WealthShareBills_BondReturn
        - params: WealthShareBills_Income

        Sets
        -----
        - HouseholdBillDemand
        """
        self.state["HouseholdBillDemand"] = (
            self.state["ExpectedWealth"] * params["WealthShareBills_Constant"]
            + self.state["ExpectedWealth"]
            * (
                params["WealthShareBills_BillRate"] * self.state["InterestRateBills"]
                + params["WealthShareBills_BondReturn"]
                * self.state["ExpectedReturnOnBonds"]
            )
            + params["WealthShareBills_Income"] * self.state["ExpectedDisposableIncome"]
        )

    def household_bond_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the household bond demand using Tobin portfolio choice.

        The demand is in number of bonds (value / price).

        Equations
        ---------
        .. math::
            BL_d(t) = \frac{V^e(t) \cdot \left(\lambda_{30}
                + \lambda_{32} \cdot r_b(t)
                + \lambda_{33} \cdot ERr_{bl}(t)
                + \lambda_{34} \cdot \frac{YD_r^e(t)}{V^e(t)}\right)}
                {p_{bl}(t)}

        Dependency
        ----------
        - state: ExpectedWealth
        - state: InterestRateBills
        - state: ExpectedReturnOnBonds
        - state: ExpectedDisposableIncome
        - state: BondPrice
        - params: WealthShareBonds_Constant
        - params: WealthShareBonds_BillRate
        - params: WealthShareBonds_BondReturn
        - params: WealthShareBonds_Income

        Sets
        -----
        - HouseholdBondDemand
        """
        # Compute the bond share of wealth (in value terms)
        bond_value_demand = self.state["ExpectedWealth"] * (
            params["WealthShareBonds_Constant"]
            + params["WealthShareBonds_BillRate"] * self.state["InterestRateBills"]
            + params["WealthShareBonds_BondReturn"]
            * self.state["ExpectedReturnOnBonds"]
            + params["WealthShareBonds_Income"]
            * torch.where(
                self.state["ExpectedWealth"].abs() > 1e-10,
                self.state["ExpectedDisposableIncome"] / self.state["ExpectedWealth"],
                torch.zeros_like(self.state["ExpectedDisposableIncome"]),
            )
        )

        # Convert from value to number of bonds
        self.state["HouseholdBondDemand"] = torch.where(
            self.state["BondPrice"].abs() > 1e-10,
            bond_value_demand / self.state["BondPrice"],
            torch.zeros_like(bond_value_demand),
        )

    ############################################################################
    # Portfolio Holdings
    ############################################################################

    def household_bill_holdings(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the household bill holdings.

        Equations
        ---------
        .. math::
            B_h(t) = B_d(t)

        Dependency
        ----------
        - state: HouseholdBillDemand

        Sets
        -----
        - HouseholdBillStock
        """
        self.state["HouseholdBillStock"] = self.state["HouseholdBillDemand"]

    def household_bond_holdings(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the household bond holdings.

        Equations
        ---------
        .. math::
            BL_h(t) = BL_d(t)

        Dependency
        ----------
        - state: HouseholdBondDemand

        Sets
        -----
        - HouseholdBondStock
        """
        self.state["HouseholdBondStock"] = self.state["HouseholdBondDemand"]

    def household_cash_stock(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the household cash stock as a residual.

        Cash is what remains of wealth after bills and bonds.

        Equations
        ---------
        .. math::
            H_h(t) = V(t) - B_h(t) - p_{bl}(t) \cdot BL_h(t)

        Dependency
        ----------
        - state: Wealth
        - state: HouseholdBillStock
        - state: BondPrice
        - state: HouseholdBondStock

        Sets
        -----
        - HouseholdCashStock
        """
        self.state["HouseholdCashStock"] = (
            self.state["Wealth"]
            - self.state["HouseholdBillStock"]
            - self.state["BondPrice"] * self.state["HouseholdBondStock"]
        )

    def household_cash_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the household cash demand as a residual.

        Equations
        ---------
        .. math::
            H_d(t) = V^e(t) - B_d(t) - p_{bl}(t) \cdot BL_d(t)

        Dependency
        ----------
        - state: ExpectedWealth
        - state: HouseholdBillDemand
        - state: BondPrice
        - state: HouseholdBondDemand

        Sets
        -----
        - HouseholdCashDemand
        """
        self.state["HouseholdCashDemand"] = (
            self.state["ExpectedWealth"]
            - self.state["HouseholdBillDemand"]
            - self.state["BondPrice"] * self.state["HouseholdBondDemand"]
        )

    ############################################################################
    # Government and Central Bank
    ############################################################################

    def government_bond_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the government bond supply.

        Bond supply equals bond demand from households.

        Equations
        ---------
        .. math::
            BL_s(t) = BL_h(t)

        Dependency
        ----------
        - state: HouseholdBondStock

        Sets
        -----
        - GovernmentBondSupply
        """
        self.state["GovernmentBondSupply"] = self.state["HouseholdBondStock"]

    def government_bill_issuance(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the government bill issuance.

        The government budget constraint determines the supply of bills:
        new bills finance the deficit net of bond issuance revenue.

        Equations
        ---------
        .. math::
            B_s(t) = B_s(t-1)
                + G(t) + r_b(t-1) \cdot B_s(t-1) + BL_s(t-1)
                - T(t) - r_b(t-1) \cdot B_{CB}(t-1)
                - (BL_s(t) - BL_s(t-1)) \cdot p_{bl}(t)

        Dependency
        ----------
        - prior: GovernmentBillStock
        - state: ConsumptionGovernment
        - prior: InterestRateBills
        - prior: GovernmentBondSupply
        - state: Taxes
        - state: CentralBankProfits
        - state: GovernmentBondSupply
        - state: BondPrice

        Sets
        -----
        - GovernmentBillStock
        """
        self.state["GovernmentBillStock"] = (
            self.prior["GovernmentBillStock"]
            + (
                # Government expenditure
                self.state["ConsumptionGovernment"]
                # Interest on outstanding bills
                + self.prior["InterestRateBills"] * self.prior["GovernmentBillStock"]
                # Coupon payments on outstanding bonds
                + self.prior["GovernmentBondSupply"]
            )
            - (
                # Tax revenue
                self.state["Taxes"]
                # Central bank profits returned to government
                + self.state["CentralBankProfits"]
            )
            # Revenue from net bond issuance
            - (
                (
                    self.state["GovernmentBondSupply"]
                    - self.prior["GovernmentBondSupply"]
                )
                * self.state["BondPrice"]
            )
        )

    def central_bank_bill_holdings(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the central bank bill holdings.

        Equations
        ---------
        .. math::
            B_{CB}(t) = B_s(t) - B_h(t)

        Dependency
        ----------
        - state: GovernmentBillStock
        - state: HouseholdBillStock

        Sets
        -----
        - CentralBankBillStock
        """
        self.state["CentralBankBillStock"] = (
            self.state["GovernmentBillStock"] - self.state["HouseholdBillStock"]
        )

    def central_bank_money_stock(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the central bank money stock.

        Equations
        ---------
        .. math::
            H_s(t) = H_s(t-1) + (B_{CB}(t) - B_{CB}(t-1))

        Dependency
        ----------
        - prior: CentralBankMoneyStock
        - state: CentralBankBillStock
        - prior: CentralBankBillStock

        Sets
        -----
        - CentralBankMoneyStock
        """
        self.state["CentralBankMoneyStock"] = (
            self.prior["CentralBankMoneyStock"]
            + self.state["CentralBankBillStock"]
            - self.prior["CentralBankBillStock"]
        )

    ############################################################################
    # Diagnostic Outputs
    ############################################################################

    def expected_capital_gains(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the expected capital gains on bonds.

        In the base LP model with static expectations (pebl = pbl),
        expected capital gains are zero.

        Equations
        ---------
        .. math::
            CG^e(t) = \chi \cdot (p_{bl}^e(t) - p_{bl}(t))
                \cdot BL_h(t)

        Dependency
        ----------
        - params: ExpectationWeightBondPrice
        - state: ExpectedBondPrice
        - state: BondPrice
        - state: HouseholdBondStock

        Sets
        -----
        - ExpectedCapitalGains
        """
        self.state["ExpectedCapitalGains"] = (
            params["ExpectationWeightBondPrice"]
            * (self.state["ExpectedBondPrice"] - self.state["BondPrice"])
            * self.state["HouseholdBondStock"]
        )
