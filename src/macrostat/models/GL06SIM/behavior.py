"""
Behavior classes for the Godley-Lavoie 2006 SIM model.
This module will define the forward and simulate behavior of the Godley-Lavoie 2006 SIM model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

import torch

from macrostat.core.behavior import Behavior
from macrostat.models.GL06SIM.parameters import ParametersGL06SIM
from macrostat.models.GL06SIM.scenarios import ScenariosGL06SIM
from macrostat.models.GL06SIM.variables import VariablesGL06SIM

logger = logging.getLogger(__name__)


class BehaviorGL06SIM(Behavior):
    """Behavior class for the Godley-Lavoie 2006 SIM model."""

    version = "GL06SIM"

    def __init__(
        self,
        parameters: ParametersGL06SIM | None = None,
        scenarios: ScenariosGL06SIM | None = None,
        variables: VariablesGL06SIM | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        """Initialize the behavior of the Godley-Lavoie 2006 SIM model.

        Parameters
        ----------
        parameters: ParametersGL06SIM | None
            The parameters of the model.
        scenarios: ScenariosGL06SIM | None
            The scenarios of the model.
        variables: VariablesGL06SIM | None
            The variables of the model.
        record: bool
            Whether to record the model output.
        scenario: int
            The scenario to use for the model.
        """

        if parameters is None:
            parameters = ParametersGL06SIM()
        if scenarios is None:
            scenarios = ScenariosGL06SIM()
        if variables is None:
            variables = VariablesGL06SIM()

        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            scenario=scenario,
            debug=debug,
        )

    def initialize(self):
        r"""Initialize the behavior of the Godley-Lavoie 2006 SIM model.

        Within the book the initialization is generally to set all non-scenario
        variables to zero. Accordingly

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                C_d(0) &= C_s(0) = 0 \\
                G_d(0) &= G_s(0) = 0 \\
                T_s(0) &= T_d(0) = 0 \\
                N_s(0) &= N_d(0) = 0 \\
                YD(0) &= 0 \\
                W(0) &= 0 \\
                H_s(0) &= 0 \\
                H_h(0) &= 0
            \end{align}

        Dependency
        ----------


        Sets
        -----
        - ConsumptionDemand
        - ConsumptionSupply
        - GovernmentDemand
        - GovernmentSupply
        - TaxSupply
        - TaxDemand
        - LabourSupply
        - LabourDemand
        - DisposableIncome
        - WageRate
        - MoneySupply
        - HouseholdMoneyStock

        """
        self.state["ConsumptionDemand"] = torch.zeros(1)
        self.state["ConsumptionSupply"] = torch.zeros(1)
        self.state["GovernmentDemand"] = torch.zeros(1)
        self.state["GovernmentSupply"] = torch.zeros(1)
        self.state["TaxSupply"] = torch.zeros(1)
        self.state["TaxDemand"] = torch.zeros(1)
        self.state["LabourSupply"] = torch.zeros(1)
        self.state["LabourDemand"] = torch.zeros(1)
        self.state["DisposableIncome"] = torch.zeros(1)
        self.state["MoneySupply"] = torch.zeros(1)
        self.state["HouseholdMoneyStock"] = torch.zeros(1)

    def step(self, **kwargs):
        """Step function of the Godley-Lavoie 2006 SIM model."""

        self.government_supply(**kwargs)
        self.labour_demand(**kwargs)
        self.labour_supply(**kwargs)
        self.tax_demand(**kwargs)
        self.tax_supply(**kwargs)
        self.labour_income(**kwargs)
        self.disposable_income(**kwargs)
        self.consumption_demand(**kwargs)
        self.consumption_supply(**kwargs)
        self.government_money_stock(**kwargs)
        self.household_money_stock(**kwargs)
        self.national_income(**kwargs)

    def government_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""In the model it is assumed that the supply will adjust to the demand,
        that is, whatever is demanded can and will be produced. Equation (3.2)
        in the book.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            G_s(t) = G_d(t)

        Dependency
        ----------
        - scenario: GovernmentDemand

        Sets
        -----
        - GovernmentSupply

        """
        self.state["GovernmentSupply"] = scenario["GovernmentDemand"]

    def labour_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""We can resolve the labour demand from the national income equation,
        together with the consumption demand (+ disposable income) and the government demand
        knowing that labour demand is equal to labour supply.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            N_d(t) =\frac{\alpha_2 H_h(t-1) + G_d}{W(t)(1-\alpha_1(1-\theta))}

        Dependency
        ----------
        - prior: HouseholdMoneyStock
        - scenario: GovernmentDemand
        - scenario: WageRate

        Sets
        -----
        - LabourDemand

        """
        numerator = (
            scenario["GovernmentDemand"]
            + params["PropensityToConsumeSavings"] * self.prior["HouseholdMoneyStock"]
        )
        denominator = scenario["WageRate"] * (
            1 - params["PropensityToConsumeIncome"] * (1 - params["TaxRate"])
        )
        self.state["LabourDemand"] = numerator / denominator

    def labour_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""In the model it is assumed that the supply will be equal to
        the amount of labour demanded. Equation (3.4) in the book

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            N_s(t) = N_d(t)

        Dependency
        ----------
        - state: LabourDemand

        Sets
        -----
        - LabourSupply

        """
        self.state["LabourSupply"] = self.state["LabourDemand"]

    def tax_demand(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""The tax demand is a function of the tax rate, the labour supply,
        and the wage rate. Equation (3.6) in the book.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            T_d(t) = \theta N_s(t) W(t)

        Dependency
        ----------
        - parameters: TaxRate
        - state: LabourSupply
        - scenario: WageRate

        Sets
        -----
        - TaxDemand

        """
        self.state["TaxDemand"] = (
            params["TaxRate"] * self.state["LabourSupply"] * scenario["WageRate"]
        )

    def tax_supply(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""In the model it is assumed that the supply will be equal to
        the amount of taxes demanded. Equation (3.3) in the book

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            T_s(t) = T_d(t)

        Dependency
        ----------
        - state: TaxDemand

        Sets
        -----
        - TaxSupply

        """
        self.state["TaxSupply"] = self.state["TaxDemand"]

    def labour_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The labour income is the wage rate times the labour supply. This is
        an intermediate variable used to calculate the disposable income, but is
        computed explicitly here to compute the transaction flows.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            W(t) N_s(t)

        Dependency
        ----------
        - scenario: WageRate
        - state: LabourSupply

        Sets
        -----
        - LabourIncome

        """
        self.state["LabourIncome"] = scenario["WageRate"] * self.state["LabourSupply"]

    def disposable_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The disposable income is the wage bill minus the taxes.
        Equation (3.5) in the book.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            YD(t) = W(t) N_s(t) - T_s(t)

        Dependency
        ----------
        - state: LabourIncome
        - state: TaxSupply

        Sets
        -----
        - DisposableIncome

        """
        self.state["DisposableIncome"] = (
            self.state["LabourIncome"] - self.state["TaxSupply"]
        )

    def consumption_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The consumption demand is a function of the disposable income,
        the propensity to consume income, and the propensity to consume savings.
        Equation (3.7) in the book.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            C_d(t) = \alpha_1 YD(t) + \alpha_2 H_h(t-1)

        Dependency
        ----------
        - state: DisposableIncome
        - prior: HouseholdMoneyStock

        Sets
        -----
        - ConsumptionDemand
        """
        self.state["ConsumptionDemand"] = (
            params["PropensityToConsumeIncome"] * self.state["DisposableIncome"]
            + params["PropensityToConsumeSavings"] * self.prior["HouseholdMoneyStock"]
        )

    def consumption_supply(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""In the model it is assumed that the supply will adjust to the demand,
        that is, whatever is demanded can and will be produced. Equation (3.1)
        in the book.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            C_s(t) = C_d(t)

        Dependency
        ----------
        - state: ConsumptionDemand

        Sets
        -----
        - ConsumptionSupply

        """
        self.state["ConsumptionSupply"] = self.state["ConsumptionDemand"]

    def government_money_stock(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The government money stock is a function of the government demand,
        and the tax supply. Equation (3.8) in the book.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            H_s(t) = H_s(t-1) + G_d(t) - T_d(t)

        Dependency
        ----------
        - scenario: GovernmentDemand
        - state: TaxDemand
        - prior: GovernmentMoneyStock

        Sets
        -----
        - GovernmentMoneyStock

        """
        self.state["GovernmentMoneyStock"] = (
            self.prior["GovernmentMoneyStock"]
            + scenario["GovernmentDemand"]
            - self.state["TaxDemand"]
        )

    def household_money_stock(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The household money stock is a function of the disposable income,
        the propensity to consume income, and the propensity to consume savings.
        Equation (3.9) in the book.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            H_h(t) = H_h(t-1) + YD(t) - C_d(t)

        Dependency
        ----------
        - state: DisposableIncome
        - state: ConsumptionDemand
        - prior: HouseholdMoneyStock

        Sets
        -----
        - HouseholdMoneyStock

        """
        self.state["HouseholdMoneyStock"] = (
            self.prior["HouseholdMoneyStock"]
            + self.state["DisposableIncome"]
            - self.state["ConsumptionDemand"]
        )

    def national_income(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""The national income is the sum of the consumption demand,
        the government demand, and the tax supply. Equation (3.10) in the book.

        Parameters
        ----------
        t : torch.tensor
            Current time step
        scenario : dict
            Scenario dictionary
        params : dict
            Parameter dictionary

        Equations
        ---------
        .. math::
            Y(t) = C_s(t) + G_s(t)

        Dependency
        ----------
        - state: ConsumptionSupply
        - state: GovernmentSupply

        Sets
        -----
        - NationalIncome

        """
        self.state["NationalIncome"] = (
            self.state["ConsumptionSupply"] + self.state["GovernmentSupply"]
        )
