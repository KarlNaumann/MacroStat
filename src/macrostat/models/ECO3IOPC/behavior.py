"""
This module will define the forward and simulate behavior of
Marco Veronese Passarella's ECO-3IO-PC model
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import torch

from macrostat.core.behavior import Behavior
from macrostat.models.ECO3IOPC.parameters import ParametersECO3IOPC
from macrostat.models.ECO3IOPC.scenarios import ScenariosECO3IOPC
from macrostat.models.ECO3IOPC.variables import VariablesECO3IOPC

logger = logging.getLogger(__name__)


class BehaviorECO3IOPC(Behavior):
    """Behavior class for Marco Veronese Passarella's ECO-3IO-PC model"""

    version = "ECO3IOPC"

    def __init__(
        self,
        parameters: ParametersECO3IOPC | None = None,
        scenarios: ScenariosECO3IOPC | None = None,
        variables: VariablesECO3IOPC | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        """Initialize the behavior of Marco Veronese Passarella's ECO-3IO-PC model

        Parameters
        ----------
        parameters: ParametersECO3IOPC | None
            The parameters of the model.
        scenarios: ScenariosECO3IOPC | None
            The scenarios of the model.
        variables: VariablesECO3IOPC | None
            The variables of the model.
        record: bool
            Whether to record the model output.
        scenario: int
            The scenario to use for the model.
        """

        if parameters is None:
            parameters = ParametersECO3IOPC()
        if scenarios is None:
            scenarios = ScenariosECO3IOPC()
        if variables is None:
            variables = VariablesECO3IOPC()

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

        # Economic
        self.state["Prices"] = torch.ones_like(self.state["Prices"])
        self.state["ConsumerPriceIndex"] = torch.ones(1)
        self.state["ConsumerPriceInflation"] = torch.ones(1)
        self.state["GovernmentPriceIndex"] = torch.ones(1)

        # Environmental
        self.state["MatterReserves"] = 6000 * torch.ones(1)
        self.state["MatterResources"] = 388889 * torch.ones(1)
        self.state["EnergyReserves"] = 37000 * torch.ones(1)
        self.state["EnergyResources"] = 542000 * torch.ones(1)
        self.state["CO2IntensityNonRenewableEnergy"] = 0.07 * torch.ones(1)

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

        # ECO: Matter
        self.material_goods_production(**kwargs)
        self.discarding_of_socioeconomic_stock(**kwargs)
        self.recycling_of_discarded_stock(**kwargs)
        self.extraction_of_matter(**kwargs)
        self.stock_of_durable_goods(**kwargs)
        self.socioeconomic_stock(**kwargs)
        self.waste(**kwargs)

        # ECO: Energy
        self.energy_used_in_production(**kwargs)
        self.renewable_energy_used_in_production(**kwargs)
        self.non_renewable_energy_used_in_production(**kwargs)
        self.co2_intensity_change(**kwargs)
        self.emissions_from_nonrenewable_energy(**kwargs)
        self.cumulative_co2_emissions(**kwargs)
        self.temperature(**kwargs)

        # ECO: Reserves
        self.matter_to_resource_conversion(**kwargs)
        self.matter_reserves(**kwargs)
        self.carbon_mass_nonrenewable_energy(**kwargs)
        self.oxygen(**kwargs)
        self.energy_to_resource_conversion(**kwargs)
        self.energy_reserves(**kwargs)

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

    ###########################################################################
    # Ecosystem functions
    ###########################################################################
    def material_goods_production(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The material goods production in the economy

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
                x_{mat}(t) &= m_{mat}^\top x(t)
            \end{align}

        Dependency
        ----------
        - state: RealGrossOutput
        - params: MaterialIntensity

        Sets
        -----
        - RawMaterialProduction
        """
        self.state["RawMaterialProduction"] = (
            params["MaterialIntensity"] @ self.state["RealGrossOutput"]
        )

    def discarding_of_socioeconomic_stock(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The discarding of socioeconomic stock occurs as a percentage of
        existing stock, converted into units of matter

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
                dis(t) &= m_{mat}^\top (\zeta \cdot dc(t-1))
            \end{align}

        Dependency
        ----------
        - prior: DurableGoodsStock
        - params: MaterialIntensity
        - params: DiscardedStockShare

        Sets
        -----
        - DiscardedSocioeconomicStock
        """
        self.state["DiscardedSocioeconomicStock"] = params["MaterialIntensity"] @ (
            params["DiscardedStockShare"] * self.prior["DurableGoodsStock"]
        )

    def recycling_of_discarded_stock(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""A fixed share of the discarded socioeconomic stock is recycled

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
                rec(t) &= \rho_{dis} dis(t)
            \end{align}

        Dependency
        ----------
        - state: DiscardedSocioeconomicStock
        - params: RecyclingRate

        Sets
        -----
        - RecycledSocioeconomicStock
        """
        self.state["RecycledSocioeconomicStock"] = (
            params["RecyclingRate"] * self.state["DiscardedSocioeconomicStock"]
        )

    def extraction_of_matter(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The matter extracted is the difference in the matter consumed and
        the matter that was recycled

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
                mat(t) &= x_{mat} - rec(t)
            \end{align}

        Dependency
        ----------
        - state: RawMaterialProduction
        - state: RecycledSocioeconomicStock

        Sets
        -----
        - ExtractedMatter
        """
        self.state["ExtractedMatter"] = (
            self.state["RawMaterialProduction"]
            - self.state["RecycledSocioeconomicStock"]
        )

    def stock_of_durable_goods(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The stock of durable goods evolves based on inflows from consumption
        and outflows from discard

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
                dc(t) &= dc(t-1) + B_c c(t) - \zeta dc(t-1)
            \end{align}

        Dependency
        ----------
        - prior: DurableGoodsStock
        - state: RealConsumptionHousehold
        - params: HouseholdBudgetShare
        - params: DiscardedStockShare

        Sets
        -----
        - DurableGoodsStock
        """
        self.state["DurableGoodsStock"] = (
            self.prior["DurableGoodsStock"]
            + params["HouseholdBudgetShare"] * self.state["RealConsumptionHousehold"]
            - params["DiscardedStockShare"] * self.prior["DurableGoodsStock"]
        )

    def socioeconomic_stock(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The socioeconomic stock grows through material extraction and
        shrinks due to discards

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
                k_h(t) &= k_h(t-1) + x_{mat}(t) - dis(t)
            \end{align}

        Dependency
        ----------
        - prior: SocioeconomicStock
        - state: RawMaterialProduction
        - state: DiscardedSocioeconomicStock

        Sets
        -----
        - SocioeconomicStock
        """
        self.state["SocioeconomicStock"] = (
            self.prior["SocioeconomicStock"]
            + self.state["RawMaterialProduction"]
            - self.prior["DiscardedSocioeconomicStock"]
        )

    def waste(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Waste is computed as the difference in matter extraction and the
        growth in the SocioeconomicStock

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
                wa(t) &= mat(t) - (k_h(t) - k_h(t-1))
            \end{align}

        Dependency
        ----------
        - prior: SocioeconomicStock
        - state: SocioeconomicStock
        - state: ExtractedMatter

        Sets
        -----
        - Waste
        """
        self.state["Waste"] = (
            self.state["ExtractedMatter"]
            - self.state["SocioeconomicStock"]
            + self.prior["SocioeconomicStock"]
        )

    def energy_used_in_production(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Energy use in production is given by a fixed energy intensity of
        production

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
                en(t) = \epsilon_e^\top x(t)
            \end{align}

        Dependency
        ----------
        - state: RealGrossOutput
        - params: EnergyIntensity

        Sets
        -----
        - EnergyRequiredForProduction
        """
        self.state["EnergyRequiredForProduction"] = (
            params["EnergyIntensity"] @ self.state["RealGrossOutput"]
        )

    def renewable_energy_used_in_production(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Renewable energy use in production is given by a fixed energy intensity of
        production combined with a fixed share of energy sourced from renewables

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
                ren(t) = \epsilon_e^\top (\eta_{en} \cdot x(t))
            \end{align}

        Dependency
        ----------
        - state: RealGrossOutput
        - params: EnergyIntensity
        - params: RenewableEnergyUseShare

        Sets
        -----
        - RenewableEnergy
        """
        self.state["RenewableEnergy"] = params["EnergyIntensity"] @ (
            params["RenewableEnergyUseShare"] * self.state["RealGrossOutput"]
        )

    def non_renewable_energy_used_in_production(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Non-renewable energy use in production is given by the difference in
        energy used and renewable energy used.

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
                nen(t) = en(t) - ren(t)
            \end{align}

        Dependency
        ----------
        - state: EnergyRequiredForProduction
        - state: RenewableEnergy

        Sets
        -----
        - NonRenewableEnergy
        """
        self.state["NonRenewableEnergy"] = (
            self.state["EnergyRequiredForProduction"] - self.state["RenewableEnergy"]
        )

    def co2_intensity_change(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The energy emission intensity decreases by a fixed percentage
        each period

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
                \beta_e(t) = \beta_e(t-1) (1 - \Delta_\% \beta_e)
            \end{align}

        Dependency
        ----------
        - prior: CO2IntensityNonRenewableEnergy
        - params: CO2IntensityNonRenewableEnergyGrowth

        Sets
        -----
        - CO2IntensityNonRenewableEnergy
        """
        self.state["CO2IntensityNonRenewableEnergy"] = self.prior[
            "CO2IntensityNonRenewableEnergy"
        ] * (1 - params["CO2IntensityNonRenewableEnergyGrowth"])

    def emissions_from_nonrenewable_energy(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Emissions are based on the use of non-renewable energy, with a fixed
        emission intensity

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
                emis(t) = \beta_e nen(t)
            \end{align}

        Dependency
        ----------
        - state: NonRenewableEnergy
        - state: CO2IntensityNonRenewableEnergy

        Sets
        -----
        - Emissions
        """
        self.state["Emissions"] = (
            self.state["NonRenewableEnergy"]
            * self.state["CO2IntensityNonRenewableEnergy"]
        )

    def cumulative_co2_emissions(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Cumulative CO2 emissions are simply incremented by the current
        emissions

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
                co2_{cum}(t) = co2_{cum}(t-1) + emis(t)
            \end{align}

        Dependency
        ----------
        - state: Emissions
        - prior: CumulativeCO2

        Sets
        -----
        - CumulativeCO2
        """
        self.state["CumulativeCO2"] = (
            self.prior["CumulativeCO2"] + self.state["Emissions"]
        )

    def temperature(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Temperature is determined by a transformation of cumulative CO2

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
                temp(t) = \frac{1}{1-fnc}\cdot tcre \cdot co2_{cum}(t)
            \end{align}

        Dependency
        ----------
        - state: CumulativeCO2
        - params: TransientClimateResponseCumCO2
        - params: NonCO2AnthropocentricForcing

        Sets
        -----
        - CumulativeCO2
        """
        self.state["Temperature"] = (
            (1 / (1 - params["NonCO2AnthropocentricForcing"]))
            * params["TransientClimateResponseCumCO2"]
            * self.state["CumulativeCO2"]
        )

    def matter_to_resource_conversion(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Matter resources is converted into reserves at a fixed rate

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
                res(t) &= res(t-1) - conv_m(t)\\
                conv_m(t) &= \sigma_m res(t)
            \end{align}

        Dependency
        ----------
        - prior: MatterResources
        - params: MatterToResourceConversionRate

        Sets
        -----
        - MatterResources
        - ConversionMatterToReserves
        """
        self.state["MatterResources"] = self.prior["MatterResources"] / (
            1 + params["MatterToResourceConversionRate"]
        )
        self.state["ConversionMatterToReserves"] = (
            self.prior["MatterResources"] - self.state["MatterResources"]
        )

    def matter_reserves(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Matter reserves are depleted by human use and incremented by the
        conversion from resources

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
                k_m(t) &= k_m(t-1) + conv_m(t) - mat(t)
            \end{align}

        Dependency
        ----------
        - prior: MatterReserves
        - state: ConversionMatterToReserves
        - state: ExtractedMatter

        Sets
        -----
        - MatterReserves
        """
        self.state["MatterReserves"] = (
            self.prior["MatterReserves"]
            + self.state["ConversionMatterToReserves"]
            - self.state["ExtractedMatter"]
        )

    def carbon_mass_nonrenewable_energy(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The carbon mass of non-renewable energy is given by the conversion
        of emissions (due to non-renewable energy) with a fixed constant

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
                cen(t) &= \frac{emis(t)}{car}
            \end{align}

        Dependency
        ----------
        - state: Emissions
        - params: CarbonToCO2Conversion

        Sets
        -----
        - CarbonMassOfEnergy
        """
        self.state["CarbonMassOfEnergy"] = (
            self.state["Emissions"] / params["CarbonToCO2Conversion"]
        )

    def oxygen(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""The oxygen level is given by the difference in emissions and the
        carbon mass of energy

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
                o2(t) &= emis(t) - cen(t)
            \end{align}

        Dependency
        ----------
        - state: Emissions
        - state: CarbonMassOfEnergy

        Sets
        -----
        - MassOfOxygen
        """
        self.state["MassOfOxygen"] = (
            self.state["Emissions"] - self.state["CarbonMassOfEnergy"]
        )

    def energy_to_resource_conversion(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Energy resources are converted into reserves at a fixed rate

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
                res_e(t) &= res_e(t-1) - conv_e(t)\\
                conv_e(t) &= \sigma_e res_e(t)
            \end{align}

        Dependency
        ----------
        - prior: EnergyResources
        - params: EnergyToResourceConversionRate

        Sets
        -----
        - EnergyResources
        - ConversionEnergyToReserves
        """
        self.state["EnergyResources"] = self.prior["EnergyResources"] / (
            1 + params["EnergyToResourceConversionRate"]
        )
        self.state["ConversionEnergyToReserves"] = (
            self.prior["EnergyResources"] - self.state["EnergyResources"]
        )

    def energy_reserves(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Energyreserves are depleted by human use and incremented by the
        conversion from resources

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
                k_e(t) &= k_e(t-1) + conv_e(t) - mat(t)
            \end{align}

        Dependency
        ----------
        - prior: EnergyReserves
        - state: ConversionEnergyToReserves
        - state: EnergyRequiredForProduction

        Sets
        -----
        - EnergyReserves
        """
        self.state["EnergyReserves"] = (
            self.prior["EnergyReserves"]
            + self.state["ConversionEnergyToReserves"]
            - self.state["EnergyRequiredForProduction"]
        )
