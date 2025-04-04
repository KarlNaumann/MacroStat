"""
This module will define the forward and simulate behavior of the Godley-Lavoie 2006 PC model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import torch

from macrostat.core.behavior import Behavior
from macrostat.models.GL06PC.parameters import ParametersGL06PC
from macrostat.models.GL06PC.scenarios import ScenariosGL06PC
from macrostat.models.GL06PC.variables import VariablesGL06PC

logger = logging.getLogger(__name__)


class BehaviorGL06PC(Behavior):
    """Behavior class for the Godley-Lavoie 2006 PC model."""

    version = "GL06PC"

    def __init__(
        self,
        parameters: ParametersGL06PC | None = None,
        scenarios: ScenariosGL06PC | None = None,
        variables: VariablesGL06PC | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        """Initialize the behavior of the Godley-Lavoie 2006 SIM model.

        Parameters
        ----------
        parameters: ParametersGL06PC | None
            The parameters of the model.
        scenarios: ScenariosGL06PC | None
            The scenarios of the model.
        variables: VariablesGL06PC | None
            The variables of the model.
        record: bool
            Whether to record the model output.
        scenario: int
            The scenario to use for the model.
        """

        if parameters is None:
            parameters = ParametersGL06PC()
        if scenarios is None:
            scenarios = ScenariosGL06PC()
        if variables is None:
            variables = VariablesGL06PC()

        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            scenario=scenario,
            debug=debug,
        )

    def initialize(self):
        r"""Initialize the behavior of the Godley-Lavoie 2006 PC model.

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
        - InterestEarnedOnBillsCentralBank
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
        self.state["ConsumptionHousehold"] = torch.zeros(1)
        self.state["ConsumptionGovernment"] = torch.zeros(1)
        self.state["NationalIncome"] = torch.zeros(1)
        self.state["InterestEarnedOnBillsHousehold"] = torch.zeros(1)
        self.state["InterestEarnedOnBillsCentralBank"] = torch.zeros(1)
        self.state["CentralBankProfits"] = torch.zeros(1)
        self.state["Taxes"] = torch.zeros(1)
        self.state["HouseholdMoneyStock"] = torch.zeros(1)
        self.state["CentralBankMoneyStock"] = torch.zeros(1)
        self.state["HouseholdBillStock"] = torch.zeros(1)
        self.state["GovernmentBillStock"] = torch.zeros(1)
        self.state["CentralBankBillStock"] = torch.zeros(1)
        self.state["Wealth"] = torch.zeros(1)
        self.state["InterestRate"] = torch.zeros(1)
        self.state["DisposableIncome"] = torch.zeros(1)

    def step(self, t: int, scenario: dict):
        """Step function of the Godley-Lavoie 2006 PC model."""

        # Scenario items
        self.consumption_government(t, scenario)
        self.set_interest_rate(t, scenario)

        # Items based on prior
        self.interest_earned_on_bills_household(t, scenario)
        self.interest_earned_on_bills_central_bank(t, scenario)

        # Solution of the step
        self.national_income(t, scenario)
        self.taxes(t, scenario)
        self.disposable_income(t, scenario)
        self.consumption(t, scenario)
        self.wealth(t, scenario)
        self.household_bill_holdings(t, scenario)
        self.household_money_stock(t, scenario)
        self.central_bank_profits(t, scenario)
        self.government_bill_issuance(t, scenario)
        self.central_bank_bill_holdings(t, scenario)
        self.central_bank_money_stock(t, scenario)

    def consumption_government(self, t: int, scenario: dict):
        r"""Calculate the consumption of the government. This is
        given exogenously by the scenario.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

        Dependency
        ----------
        - scenario: ConsumptionGovernment

        Sets
        -----
        - ConsumptionGovernment
        """
        self.state["ConsumptionGovernment"] = scenario["GovernmentDemand"]

    def set_interest_rate(self, t: int, scenario: dict):
        r"""Set the interest rate. This is given exogenously by the scenario.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

        Dependency
        ----------
        - scenario: InterestRate

        Sets
        -----
        - InterestRate
        """
        self.state["InterestRate"] = scenario["InterestRate"]

    def interest_earned_on_bills_household(self, t: int, scenario: dict):
        r"""Calculate the interest earned on bills by the household.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

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

    def interest_earned_on_bills_central_bank(self, t: int, scenario: dict):
        r"""Calculate the interest earned on bills by the central bank.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

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
        - InterestEarnedOnBillsCentralBank
        """
        self.state["InterestEarnedOnBillsCentralBank"] = (
            self.prior["InterestRate"] * self.prior["CentralBankBillStock"]
        )

    def national_income(self, t: int, scenario: dict):
        r"""Calculate the national income based on the closed-form solution derived in the documentation.

        The closed-form solution is used to avoid the need to solve the system of equations iteratively, thus
        preserving the differentiability of the model trajectory.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.


        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                Y(t) = \frac{\alpha_1(1-\theta)r(t-1)B_h(t-1) + \alpha_2 V(t-1) + G(t)}{1 - \alpha_1(1-\theta)}
            \end{align}

        Dependency
        ----------
        - state: InterestEarnedOnBillsHousehold
        - state: ConsumptionGovernment
        - prior: Wealth

        Sets
        -----
        - NationalIncome
        """
        self.state["NationalIncome"] = (
            # Spending out of bond income
            self.params["PropensityToConsumeIncome"]
            * (1 - self.params["TaxRate"])
            * self.state["InterestEarnedOnBillsHousehold"]
            # Spending out of wealth
            + self.params["PropensityToConsumeSavings"] * self.prior["Wealth"]
            # Government spending
            + self.state["ConsumptionGovernment"]
        ) / (
            # Multiplier
            1
            - self.params["PropensityToConsumeIncome"] * (1 - self.params["TaxRate"])
        )

    def taxes(self, t: int, scenario: dict):
        r"""Calculate the taxes.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                T(t) = \theta (Y(t) + r(t-1)B_h(t-1))
            \end{align}

        Dependency
        ----------
        - state: NationalIncome
        - state: InterestEarnedOnBillsHousehold

        Sets
        -----
        - Taxes
        """
        self.state["Taxes"] = self.params["TaxRate"] * (
            self.state["NationalIncome"] + self.state["InterestEarnedOnBillsHousehold"]
        )

    def disposable_income(self, t: int, scenario: dict):
        r"""Calculate the disposable income.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

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

    def consumption(self, t: int, scenario: dict):
        r"""Calculate the consumption.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                C(t) = \alpha_1 YD(t) + \alpha_2 V(t-1)
            \end{align}

        Dependency
        ----------
        - state: DisposableIncome
        - prior: Wealth

        Sets
        -----
        - ConsumptionHousehold
        """
        self.state["ConsumptionHousehold"] = (
            self.params["PropensityToConsumeIncome"] * self.state["DisposableIncome"]
            + self.params["PropensityToConsumeSavings"] * self.prior["Wealth"]
        )

    def wealth(self, t: int, scenario: dict):
        r"""Calculate the wealth.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

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

    def household_bill_holdings(self, t: int, scenario: dict):
        r"""Calculate the household bill holdings.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                \frac{B_h(t)}{V(t)} = \lambda_0 + \lambda_1 r(t) - \lambda_2 \frac{YD(t)}{V(t)}
            \end{align}

        Dependency
        ----------
        - state: DisposableIncome
        - state: Wealth
        - state: InterestRate

        Sets
        -----
        - HouseholdBillStock
        """
        self.state["HouseholdBillStock"] = self.state["Wealth"] * (
            # Baseline share
            self.params["WealthShareBills_Constant"]
            # Interest rate effect
            + self.params["WealthShareBills_InterestRate"] * self.state["InterestRate"]
            # Income-to-wealth ratio effect
            - self.params["WealthShareBills_Income"]
            * self.state["DisposableIncome"]
            / self.state["Wealth"]
        )

    def household_money_stock(self, t: int, scenario: dict):
        r"""Calculate the household deposits as a residual.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

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

    def central_bank_profits(self, t: int, scenario: dict):
        r"""Calculate the central bank profits (income on bills held).

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

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

    def government_bill_issuance(self, t: int, scenario: dict):
        r"""Calculate the government bill issuance.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

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
        - state: InterestEarnedOnBillsCentralBank

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

    def central_bank_bill_holdings(self, t: int, scenario: dict):
        r"""Calculate the central bank bill holdings.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

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

    def central_bank_money_stock(self, t: int, scenario: dict):
        r"""Calculate the central bank money stock.

        Parameters
        ----------
        t: int
            The time step.
        scenario: dict
            The scenario.

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
