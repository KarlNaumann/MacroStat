"""
This module will define the forward and simulate behavior of
Marco Veronese Passarella's 3IO-PC model
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import torch

from macrostat.core.behavior import Behavior
from macrostat.models.IOPC.parameters import ParametersIOPC
from macrostat.models.IOPC.scenarios import ScenariosIOPC
from macrostat.models.IOPC.variables import VariablesIOPC

logger = logging.getLogger(__name__)


class BehaviorIOPC(Behavior):
    """Behavior class for Marco Veronese Passarella's 3IO-PC model"""

    version = "IOPC"

    def __init__(
        self,
        parameters: ParametersIOPC | None = None,
        scenarios: ScenariosIOPC | None = None,
        variables: VariablesIOPC | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        """Initialize the behavior of Marco Veronese Passarella's 3IO-PC model

        Parameters
        ----------
        parameters: ParametersIOPC | None
            The parameters of the model.
        scenarios: ScenariosIOPC | None
            The scenarios of the model.
        variables: VariablesIOPC | None
            The variables of the model.
        record: bool
            Whether to record the model output.
        scenario: int
            The scenario to use for the model.
        """

        if parameters is None:
            parameters = ParametersIOPC()
        if scenarios is None:
            scenarios = ScenariosIOPC()
        if variables is None:
            variables = VariablesIOPC()

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
        r"""Initialize the behavior of Passarella's 3IO-PC model in the spirit
        of Godley & Lavoie, by keeping all variables as zero. Accordingly, we
        can just "pass" the function as by default the state variables are all
        zero. The only exceptions are the price-indices which we initialize to
        one
        """
        self.state["Prices"] = torch.ones_like(self.state["Prices"])
        self.state["ConsumerPriceIndex"] = torch.ones(1)
        self.state["ConsumerPriceInflation"] = torch.ones(1)
        self.state["GovernmentPriceIndex"] = torch.ones(1)

    ############################################################################
    # Step
    ############################################################################

    def step(self, **kwargs):
        """Step function of the Godley-Lavoie 2006 PC model."""

        # Scenario items
        self.consumption_government(**kwargs)
        self.set_interest_rate(**kwargs)

        # Items based on prior
        self.interest_earned_on_bills_household(**kwargs)
        self.expected_disposable_income(**kwargs)

        # Solution of the step
        # Input-output mechanisms
        self.prices(**kwargs)
        self.price_indices(**kwargs)
        self.inflation(**kwargs)
        self.propensity_to_consume_income(**kwargs)
        self.consumption(**kwargs)
        self.final_demand(**kwargs)
        self.real_gross_output(**kwargs)
        self.national_income(**kwargs)
        self.taxes(**kwargs)
        self.disposable_income(**kwargs)
        self.wealth(**kwargs)
        self.expected_wealth(**kwargs)
        self.household_bill_demand(**kwargs)
        self.household_bill_holdings(**kwargs)
        self.household_money_stock(**kwargs)
        self.central_bank_profits(**kwargs)
        self.government_bill_issuance(**kwargs)
        self.central_bank_bill_holdings(**kwargs)
        self.central_bank_money_stock(**kwargs)

    def consumption_government(
        self,
        t: int,
        scenario: dict,
        params: dict | None = None,
        **kwargs,
    ):
        r"""Calculate the consumption of the government. This is
        given exogenously by the scenario.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Dependency
        ----------
        - scenario: ConsumptionGovernment

        Sets
        -----
        - RealConsumptionGovernment
        """
        self.state["RealConsumptionGovernment"] = scenario["GovernmentDemand"]

    def set_interest_rate(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set the interest rate. This is given exogenously by the scenario.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                r(t) = \bar{r}
            \end{align)

        Dependency
        ----------
        - scenario: InterestRate

        Sets
        -----
        - InterestRate
        """
        self.state["InterestRate"] = scenario["InterestRate"]

    def interest_earned_on_bills_household(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the interest earned on bills by the household.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                r(t-1)B_h(t-1)
            \end{align}

        Dependency
        ----------
        - prior: InterestRate
        - prior: HouseholdBillStock

        Sets
        -----
        - InterestEarnedOnBillsHousehold
        """
        self.state["InterestEarnedOnBillsHousehold"] = (
            self.prior["InterestRate"] * self.prior["HouseholdBillStock"]
        )

    def expected_disposable_income(
        self, t: torch.tensor, scenario: dict, params: dict | None = None
    ):
        r"""The expected disposable income is simply the prior period's
        disposable income. Equation (3.20) in the book.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict

        Equations
        ---------
        .. math::
            YD^e(t) = YD(t-1)

        Dependency
        ----------
        - prior: DisposableIncome

        Sets
        -----
        - ExpectedDisposableIncome
        """
        self.state["ExpectedDisposableIncome"] = self.prior["DisposableIncome"]

    def prices(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Compute the sectoral prices as the sum of unit labour cost and a
        markup on intermediate prices

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                P_i(t) = \frac{w}{pr_i} + (1 + \mu)\sum_j a_{ij}P_j(t)
            \end{align}

        Dependency
        ----------
        - scenario: WageRate
        - params: LabourProductivity
        - params: Requirement
        - params: Markup

        Sets
        -----
        - Prices
        """
        self.state["Prices"] = torch.linalg.solve(
            A=(
                torch.eye(params["Requirement"].shape[0])
                - (1 + params["Markup"]) * params["Requirement"].T
            ),
            B=scenario["WageRate"] / params["LabourProductivity"],
        )

    def price_indices(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Compute the consumer and government price indices based on their
        consumption shares

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                p_c(t) &= \beta_{HH}^\top P(t)\\
                p_g(t) &= \beta_{G}^\top P(t)
            \end{align}

        Dependency
        ----------
        - state: Prices
        - params: HouseholdBudgetShare
        - params: GovernmentBudgetShare

        Sets
        -----
        - ConsumerPriceIndex
        - GovernmentPriceIndex
        """
        self.state["ConsumerPriceIndex"] = (
            self.state["Prices"] @ params["HouseholdBudgetShare"]
        )
        self.state["GovernmentPriceIndex"] = (
            self.state["Prices"] @ params["GovernmentBudgetShare"]
        )

    def inflation(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Compute the inflation (i.e. term for absence of money illusion)

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                \pi(t) &= \left(\frac{p_c(t) - p_c(t-1)}{p_c(t-1)}\right)\left(\frac{V(t-1)}{p_c(t-1)}\right)
            \end{align}

        Dependency
        ----------
        - prior: Wealth
        - prior: ConsumerPriceIndex
        - state: ConsumerPriceIndex

        Sets
        -----
        - ConsumerPriceInflation
        """
        self.state["ConsumerPriceInflation"] = (
            (self.state["ConsumerPriceIndex"] - self.prior["ConsumerPriceIndex"])
            / self.prior["ConsumerPriceIndex"]
        ) * (self.prior["Wealth"] / self.state["ConsumerPriceIndex"])

    def propensity_to_consume_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Endogenous propensity to consume out of income, dependent on the
        rate of interest

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                \alpha_1(t) = \alpha_{10} - \alpha_{11} r(t-1)
            \end{align}

        Dependency
        ----------
        - prior: InterestRate
        - params: PropensityToConsumeIncomeBase
        - params: PropensityToConsumeIncomeInterest

        Sets
        -----
        - PropensityToConsumeIncome
        """
        self.state["PropensityToConsumeIncome"] = params[
            "PropensityToConsumeIncomeBase"
        ] - (params["PropensityToConsumeIncomeInterest"] * self.prior["InterestRate"])

    def consumption(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the consumption.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                c(t) = \alpha_1 \left(\frac{YD^e(t)}{p_c(t)} - \pi(t)\right) + \alpha_2 \frac{V(t-1)}{p_c(t)}
            \end{align}

        Dependency
        ----------
        - state: PropensityToConsumeIncome
        - state: ExpectedDisposableIncome
        - state: ConsumerPriceIndex
        - state: ConsumerPriceInflation
        - prior: Wealth
        - params: PropensityToConsumeSavings

        Sets
        -----
        - RealConsumptionHousehold
        """
        self.state["RealConsumptionHousehold"] = self.state[
            "PropensityToConsumeIncome"
        ] * (
            (self.state["ExpectedDisposableIncome"] / self.state["ConsumerPriceIndex"])
            - self.state["ConsumerPriceInflation"]
        ) + params[
            "PropensityToConsumeSavings"
        ] * (
            self.prior["Wealth"] / self.state["ConsumerPriceIndex"]
        )

    def final_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the final demand as the sum of household and government
        demands spread over the sectors

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                d_i(t) = \beta_{HH,i}C_{HH}(t) + \beta_{GOV,i}G(t)
            \end{align}

        Dependency
        ----------
        - state: RealConsumptionHousehold
        - state: RealConsumptionGovernment
        - params: HouseholdBudgetShare
        - params: GovernmentBudgetShare

        Sets
        -----
        - RealFinalDemand
        """
        self.state["RealFinalDemand"] = +(
            params["HouseholdBudgetShare"] * self.state["RealConsumptionHousehold"]
        ) + (params["GovernmentBudgetShare"] * self.state["RealConsumptionGovernment"])

    def real_gross_output(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Compute real gross output as the solution to the linear set of
        equations

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                x(t) = (I - A)^{-1}d(t)
            \end{align}

        Dependency
        ----------
        - state: RealFinalDemand
        - params: Requirement

        Sets
        -----
        - RealGrossOutput
        """
        self.state["RealGrossOutput"] = torch.linalg.solve(
            A=(torch.eye(params["Requirement"].shape[0]) - params["Requirement"]),
            B=self.state["RealFinalDemand"],
        )

    def national_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""National income is the sum of nominal final demand

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                Y(t) = P^\top(t)d(t)
            \end{align}

        Dependency
        ----------
        - state: Prices
        - state: RealFinalDemand

        Sets
        -----
        - NationalIncome
        """
        self.state["NationalIncome"] = (
            self.state["Prices"] @ self.state["RealFinalDemand"]
        )

    def taxes(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the taxes.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                T(t) = \theta (Y(t) + r(t-1)B_h(t-1))
            \end{align}

        Dependency
        ----------
        - params: TaxRate
        - state: NationalIncome
        - state: InterestEarnedOnBillsHousehold

        Sets
        -----
        - Taxes
        """
        self.state["Taxes"] = params["TaxRate"] * (
            self.state["NationalIncome"] + self.state["InterestEarnedOnBillsHousehold"]
        )

    def disposable_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the disposable income.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                YD(t) = Y(t) - T(t) + r(t-1)B_h(t-1)
            \end{align}

        Dependency
        ----------
        - state: NationalIncome
        - state: Taxes
        - state: InterestEarnedOnBillsHousehold

        Sets
        -----
        - DisposableIncome
        """
        self.state["DisposableIncome"] = (
            self.state["NationalIncome"]
            - self.state["Taxes"]
            + self.state["InterestEarnedOnBillsHousehold"]
        )

    def wealth(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the wealth.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                V(t) = V(t-1) + YD(t) - C(t)
            \end{align}

        Dependency
        ----------
        - state: DisposableIncome
        - state: ConsumerPriceIndex
        - state: RealConsumptionHousehold
        - prior: Wealth

        Sets
        -----
        - Wealth
        """
        self.state["Wealth"] = (
            self.prior["Wealth"]
            + self.state["DisposableIncome"]
            - (
                self.state["ConsumerPriceIndex"]
                * self.state["RealConsumptionHousehold"]
            )
        )

    def expected_wealth(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the expected wealth.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                V^e(t) = V(t-1) + YD^e(t) - C(t)
            \end{align}

        Dependency
        ----------
        - state: ExpectedDisposableIncome
        - state: RealConsumptionHousehold
        - state: ConsumerPriceIndex
        - prior: Wealth

        Sets
        -----
        - ExpectedWealth
        """
        self.state["ExpectedWealth"] = (
            self.prior["Wealth"]
            + self.state["ExpectedDisposableIncome"]
            - (
                self.state["ConsumerPriceIndex"]
                * self.state["RealConsumptionHousehold"]
            )
        )

    def household_bill_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the household bill demand.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                \frac{B_h(t)}{V^e(t)} = \lambda_0 + \lambda_1 r(t) - \lambda_2 \frac{YD^e(t)}{V^e(t)}
            \end{align}

        Dependency
        ----------
        - state: ExpectedWealth
        - state: ExpectedDisposableIncome
        - state: InterestRate
        - params: WealthShareBills_Constant
        - params: WealthShareBills_InterestRate
        - params: WealthShareBills_Income

        Sets
        -----
        - HouseholdBillDemand
        """
        self.state["HouseholdBillDemand"] = self.state["ExpectedWealth"] * (
            # Baseline share
            params["WealthShareBills_Constant"]
            # Interest rate effect
            + params["WealthShareBills_InterestRate"] * self.state["InterestRate"]
            # Income-to-wealth ratio effect
            - params["WealthShareBills_Income"]
            * torch.where(
                self.state["ExpectedWealth"] > 0,
                self.state["ExpectedDisposableIncome"] / self.state["ExpectedWealth"],
                torch.zeros_like(self.state["ExpectedDisposableIncome"]),
            )
        )

    def household_bill_holdings(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the household bill holdings.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                B_h(t) = B_h(t-1) + (B_h^d(t) - B_h(t-1))
            \end{align}

        Dependency
        ----------
        - state: HouseholdBillDemand

        Sets
        -----
        - HouseholdBillStock
        """
        self.state["HouseholdBillStock"] = self.state["HouseholdBillDemand"]

    def household_money_stock(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the household deposits as a residual.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                H_h(t) = V(t) - B_h(t)
            \end{align}

        Dependency
        ----------
        - state: Wealth
        - state: HouseholdBillStock

        Sets
        -----
        - HouseholdMoneyStock
        """
        self.state["HouseholdMoneyStock"] = (
            self.state["Wealth"] - self.state["HouseholdBillStock"]
        )

    def central_bank_profits(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the central bank profits (income on bills held).

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                r(t-1)B_{CB}(t-1)
            \end{align}

        Dependency
        ----------
        - prior: InterestRate
        - prior: CentralBankBillStock

        Sets
        -----
        - CentralBankProfits
        """
        self.state["CentralBankProfits"] = (
            self.prior["InterestRate"] * self.prior["CentralBankBillStock"]
        )

    def government_bill_issuance(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the government bill issuance.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                B_s(t) = B_s(t-1) + (G(t) - r(t-1)B_s(t-1)) - (T(t) + r(t-1)B_{CB}(t-1))
            \end{align}

        Dependency
        ----------
        - prior: InterestRate
        - prior: GovernmentBillStock
        - state: GovernmentPriceIndex
        - state: RealConsumptionGovernment
        - state: Taxes
        - state: CentralBankProfits

        Sets
        -----
        - GovernmentBillStock
        """
        self.state["GovernmentBillStock"] = (
            self.prior["GovernmentBillStock"]
            + (
                # Government demand
                self.state["GovernmentPriceIndex"]
                * self.state["RealConsumptionGovernment"]
                # Interest expense on bills issued
                + self.prior["InterestRate"] * self.prior["GovernmentBillStock"]
            )
            - (
                # Tax revenue
                self.state["Taxes"]
                # Central bank profits
                + self.state["CentralBankProfits"]
            )
        )

    def central_bank_bill_holdings(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the central bank bill holdings.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                B_{CB}(t) = B_{s}(t) - B_{h}(t)
            \end{align}

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

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.
        params: dict | None
            The parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                H_{s}(t) = H_{s}(t-1) + (B_{CB}(t) - B_{CB}(t-1))
            \end{align}

        Dependency
        ----------
        - state: CentralBankBillStock
        - prior: CentralBankMoneyStock
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
