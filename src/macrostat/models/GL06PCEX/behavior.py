"""
This module will define the forward and simulate behavior of the Godley-Lavoie 2006 PCEX model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import torch

from macrostat.core.behavior import Behavior
from macrostat.models.GL06PCEX.parameters import ParametersGL06PCEX
from macrostat.models.GL06PCEX.scenarios import ScenariosGL06PCEX
from macrostat.models.GL06PCEX.variables import VariablesGL06PCEX

logger = logging.getLogger(__name__)


class BehaviorGL06PCEX(Behavior):
    """Behavior class for the Godley-Lavoie 2006 PCEX model."""

    version = "GL06PCEX"

    def __init__(
        self,
        parameters: ParametersGL06PCEX | None = None,
        scenarios: ScenariosGL06PCEX | None = None,
        variables: VariablesGL06PCEX | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        """Initialize the behavior of the Godley-Lavoie 2006 PCEX model.

        Parameters
        ----------
        parameters: ParametersGL06PCEX | None
            The parameters of the model.
        scenarios: ScenariosGL06PCEX | None
            The scenarios of the model.
        variables: VariablesGL06PCEX | None
            The variables of the model.
        record: bool
            Whether to record the model output.
        scenario: int
            The scenario to use for the model.
        """

        if parameters is None:
            parameters = ParametersGL06PCEX()
        if scenarios is None:
            scenarios = ScenariosGL06PCEX()
        if variables is None:
            variables = VariablesGL06PCEX()

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
        r"""Initialize the behavior of the Godley-Lavoie 2006 PCEX model.

        Within the book the initialization is generally to set all non-scenario
        variables to zero. Accordingly

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                C(0) &= 0 \\
                G(0) &= 0 \\
                Y(0) &= 0 \\
                T(0) &= 0 \\
                YD(0) &= 0 \\
                V(0) &= 0 \\
                H_s(0) &= 0 \\
                H_h(0) &= 0 \\
                B_h(0) &= 0 \\
                B_s(0) &= 0 \\
                B_{CB}(0) &= 0 \\
                r(0) &= 0 \\
            \end{align}

        Dependency
        ----------


        Sets
        -----
        - ConsumptionHousehold
        - ConsumptionGovernment
        - NationalIncome
        - InterestEarnedOnBillsHousehold
        - CentralBankProfits
        - Taxes
        - HouseholdMoneyStock
        - CentralBankMoneyStock
        - HouseholdBillStock
        - GovernmentBillStock
        - CentralBankBillStock
        - Wealth
        - InterestRate
        - DisposableIncome

        """
        # Flows
        self.state["ConsumptionHousehold"] = torch.zeros(1)
        self.state["ConsumptionGovernment"] = torch.zeros(1)
        self.state["NationalIncome"] = torch.zeros(1)
        self.state["InterestEarnedOnBillsHousehold"] = torch.zeros(1)
        self.state["CentralBankProfits"] = torch.zeros(1)
        self.state["Taxes"] = torch.zeros(1)
        # Stocks
        self.state["HouseholdMoneyStock"] = torch.zeros(1)
        self.state["CentralBankMoneyStock"] = torch.zeros(1)
        self.state["HouseholdBillStock"] = torch.zeros(1)
        self.state["GovernmentBillStock"] = torch.zeros(1)
        self.state["CentralBankBillStock"] = torch.zeros(1)
        self.state["Wealth"] = torch.zeros(1)
        # Indices
        self.state["InterestRate"] = torch.zeros(1)
        self.state["DisposableIncome"] = torch.zeros(1)
        self.state["ExpectedDisposableIncome"] = torch.zeros(1)
        self.state["ExpectedWealth"] = torch.zeros(1)
        self.state["HouseholdBillDemand"] = torch.zeros(1)

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
        self.consumption(**kwargs)
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
        - ConsumptionGovernment
        """
        self.state["ConsumptionGovernment"] = scenario["GovernmentDemand"]

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
                C(t) = \alpha_1 YD^e(t) + \alpha_2 V(t-1)
            \end{align}

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
        r"""Calculate the national income based on the closed-form solution derived in the documentation.

        The closed-form solution is used to avoid the need to solve the system of equations iteratively, thus
        preserving the differentiability of the model trajectory.

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
                Y(t) = C(t) + G(t)
            \end{align}

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
        - state: ConsumptionHousehold
        - prior: Wealth

        Sets
        -----
        - Wealth
        """
        self.state["Wealth"] = (
            self.prior["Wealth"]
            + self.state["DisposableIncome"]
            - self.state["ConsumptionHousehold"]
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
        - state: ConsumptionHousehold
        - prior: Wealth

        Sets
        -----
        - ExpectedWealth
        """
        self.state["ExpectedWealth"] = (
            self.prior["Wealth"]
            + self.state["ExpectedDisposableIncome"]
            - self.state["ConsumptionHousehold"]
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
        - prior: HouseholdBillStock

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
        - prior: GovernmentBillStock
        - state: GovernmentDemand
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
                scenario["GovernmentDemand"]
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

    ############################################################################
    # Steady State
    ############################################################################

    def compute_theoretical_steady_state_per_step(
        self, t: int, params: dict, scenario: dict
    ):
        r"""Compute the theoretical steady state of the model for each given
        period. This is done per-period as there are parameters and scenarios
        that may be time-varying, so the interpretation is a timeseries of the
        theoretical steady state at a given period based on the parameters and
        scenarios at that period.

        Parameters
        ----------
        params: dict
            The parameters at the given period
        scenario: dict
            The scenarios at the given period

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
            G^\star(t) &= G(t)\\
            r^\star(t) &= r(t)\\
            \alpha_3 &= \frac{1-\alpha_1}{\alpha_2}\\
            YD^\star(t) = YD^{e\star}(t) &= \frac{G^\star(t)}{\frac{\theta}{1-\theta} - r^\star(t)\cdot\left(\left(\lambda_0 + \lambda_1 r^\star(t) \right)\alpha_3 - \lambda_2\right)}\\
            C^\star(t) &= YD^\star(t)\\
            Y^\star(t) &= C^\star(t) + G^\star(t)\\
            V^\star(t) = V^{e\star}(t) &= \alpha_3 YD^\star(t)\\
            B_d^\star(t) = B_h^\star(t) &= \left(\left(\lambda_0 + \lambda_1 r^\star(t) \right)\alpha_3 - \lambda_2\right)\cdot YD^\star(t)\\
            T^\star(t) &= \theta\cdot \left(Y^\star(t) + r^\star(t) B_h^\star(t)\right)\\
            H_h^\star(t) &= V^{e\star}(t) - B_h^\star(t)\\
            B_s^\star(t) &= \frac{r^\star(t) B_{CB}^\star(t) + T^\star(t) - G^\star(t)}{r^\star(t)}\\
            B_{CB}^\star(t) &= B_s^\star(t) - B_h^\star(t)\\
            H_s^\star(t) &= H_{s}(t-1) + (B_{CB}(t) - B_{CB}(t-1))
            \end{align}
        """
        kwargs = dict(t=t, params=params, scenario=scenario)
        # Scenario specific items, as in normal step()
        self.consumption_government(**kwargs)
        self.set_interest_rate(**kwargs)

        # Compute the steady state disposable income and consumption
        alpha3 = (1 - params["PropensityToConsumeIncome"]) / params[
            "PropensityToConsumeSavings"
        ]
        self.state["DisposableIncome"] = scenario["GovernmentDemand"] / (
            (params["TaxRate"] / (1 - params["TaxRate"]))
            - self.state["InterestRate"]
            * (
                (
                    params["WealthShareBills_Constant"]
                    + params["WealthShareBills_InterestRate"]
                    * self.state["InterestRate"]
                )
                * alpha3
                - params["WealthShareBills_Income"]
            )
        )
        self.state["ExpectedDisposableIncome"] = self.state["DisposableIncome"]
        self.state["ConsumptionHousehold"] = self.state["DisposableIncome"]
        self.national_income(**kwargs)

        # Compute the steady state wealth
        self.state["Wealth"] = self.state["DisposableIncome"] * alpha3
        self.state["ExpectedWealth"] = self.state["Wealth"]

        # Compute the steady state bill holdings
        self.state["HouseholdBillDemand"] = (
            (
                params["WealthShareBills_Constant"]
                + params["WealthShareBills_InterestRate"] * self.state["InterestRate"]
            )
            * alpha3
            - params["WealthShareBills_Income"]
        ) * self.state["DisposableIncome"]
        self.state["HouseholdBillStock"] = self.state["HouseholdBillDemand"]
        self.state["InterestEarnedOnBillsHousehold"] = (
            self.state["InterestRate"] * self.state["HouseholdBillStock"]
        )

        # Compute remaining variables, using the step functions where possible
        self.taxes(**kwargs)
        self.household_money_stock(**kwargs)
        self.state["CentralBankBillStock"] = self.state["HouseholdMoneyStock"]
        self.state["CentralBankProfits"] = (
            self.state["InterestRate"] * self.state["CentralBankBillStock"]
        )
        self.state["GovernmentBillStock"] = (
            self.state["Taxes"]
            + self.state["CentralBankProfits"]
            - self.state["ConsumptionGovernment"]
        ) / self.state["InterestRate"]
        # Via the redundant equation
        self.state["CentralBankMoneyStock"] = self.state["HouseholdMoneyStock"]
