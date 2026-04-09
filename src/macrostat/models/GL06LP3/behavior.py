"""
This module defines the forward and simulate behavior of the
Godley-Lavoie 2006 LP3 model (Endogenous government spending
via fiscal austerity rule).
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

from macrostat.models.GL06LP2.behavior import BehaviorGL06LP2 as _BehaviorGL06LP2
from macrostat.models.GL06LP3.parameters import ParametersGL06LP3
from macrostat.models.GL06LP3.scenarios import ScenariosGL06LP3
from macrostat.models.GL06LP3.variables import VariablesGL06LP3
from macrostat.models.GL06LP.behavior import BehaviorGL06LP as _BehaviorGL06LP

logger = logging.getLogger(__name__)


class BehaviorGL06LP3(_BehaviorGL06LP2):
    """Behavior class for the Godley-Lavoie 2006 LP3 model.

    Extends Model LP2 by making government expenditures endogenous
    through a fiscal austerity rule (equations 5.28–5.31). When the
    deficit-to-GDP ratio exceeds a threshold, the government cuts
    spending. This introduces hysteresis — the steady state becomes
    path-dependent.
    """

    version = "GL06LP3"

    def __init__(
        self,
        parameters: ParametersGL06LP3 | None = None,
        scenarios: ScenariosGL06LP3 | None = None,
        variables: VariablesGL06LP3 | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        """Initialize the behavior of the Godley-Lavoie 2006 LP3 model.

        Parameters
        ----------
        parameters : ParametersGL06LP3 | None
            The parameters of the model.
        scenarios : ScenariosGL06LP3 | None
            The scenarios of the model.
        variables : VariablesGL06LP3 | None
            The variables of the model.
        scenario : int
            The scenario to use for the model.
        debug : bool
            Whether to enable debug mode.
        """

        if parameters is None:
            parameters = ParametersGL06LP3()
        if scenarios is None:
            scenarios = ScenariosGL06LP3()
        if variables is None:
            variables = VariablesGL06LP3()

        # Bypass _BehaviorGL06LP2.__init__ and _BehaviorGL06LP.__init__
        super(_BehaviorGL06LP, self).__init__(
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
        r"""Initialize the behavior of the Godley-Lavoie 2006 LP3 model.

        Extends LP2 initialization by adding PublicSectorBorrowingRequirement.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                PSBR(0) = 0
            \end{align}

        Sets
        -----
        All LP2 variables to zero/defaults, plus PSBR.
        """
        super().initialize()
        ref = next(iter(self.state.values()))
        self.state["PublicSectorBorrowingRequirement"] = torch.zeros_like(ref)

    ############################################################################
    # Step
    ############################################################################

    def step(self, **kwargs):
        """Step function of the Godley-Lavoie 2006 LP3 model."""

        # 1. Scenario exogenous variables
        # Government demand is now endogenous (fiscal rule)
        self.set_government_demand(**kwargs)
        self.set_interest_rate_bills(**kwargs)

        # 2. Endogenous bond price via target-proportion mechanism (from LP2)
        self.target_proportion(**kwargs)
        self.compute_bond_price(**kwargs)

        # 3. Bond yield and expected return
        self.bond_yield(**kwargs)
        self.expected_bond_price(**kwargs)
        self.expected_return_on_bonds(**kwargs)

        # 4. Prior-based flows
        self.interest_on_bills_household(**kwargs)
        self.bond_coupon_income_household(**kwargs)
        self.capital_gains(**kwargs)
        self.central_bank_profits(**kwargs)

        # 5. Expected disposable income
        self.expected_disposable_income(**kwargs)

        # 6. Consumption and income
        self.consumption(**kwargs)
        self.national_income(**kwargs)
        self.taxes(**kwargs)
        self.disposable_income(**kwargs)

        # 7. Wealth
        self.wealth(**kwargs)
        self.expected_wealth(**kwargs)

        # 8. Portfolio demands and holdings
        self.household_bill_demand(**kwargs)
        self.household_bond_demand(**kwargs)
        self.household_bill_holdings(**kwargs)
        self.household_bond_holdings(**kwargs)
        self.household_cash_stock(**kwargs)
        self.household_cash_demand(**kwargs)

        # 9. Government and central bank balance sheet
        self.government_bond_supply(**kwargs)
        self.government_bill_issuance(**kwargs)
        self.central_bank_bill_holdings(**kwargs)
        self.central_bank_money_stock(**kwargs)

        # 10. Diagnostic outputs
        self.expected_capital_gains(**kwargs)

        # 11. PSBR (stored for next period's fiscal rule)
        self.psbr(**kwargs)

    ############################################################################
    # Endogenous Government Demand (replaces set_government_demand from LP)
    ############################################################################

    def set_government_demand(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Set government demand using the fiscal austerity rule.

        In the first period (when prior national income is ~0), G is set
        from the scenario. In subsequent periods, G evolves endogenously:
        if the deficit-to-GDP ratio exceeds the threshold, the government
        cuts spending.

        Equations
        ---------
        .. math::
            PSBR(t-1) / Y(t-1) > \text{threshold} \Rightarrow z_3 = 1

            PSBR(t-1) / Y(t-1) < -\text{threshold} \Rightarrow z_4 = 1

            G(t) = G(t-1) - (z_3 + z_4) \cdot \beta_g \cdot PSBR(t-1)
                + \text{add2}

        Dependency
        ----------
        - prior: NationalIncome
        - prior: PublicSectorBorrowingRequirement
        - prior: ConsumptionGovernment
        - params: FiscalAdjustmentSpeed
        - params: PSBRThreshold
        - scenario: GovernmentDemand (initial value)
        - scenario: GovernmentSpendingShock (add2)

        Sets
        -----
        - ConsumptionGovernment
        """
        prior_y = self.prior["NationalIncome"]

        if prior_y.abs().item() < 1e-10:
            # First period: use scenario value as initial G
            self.state["ConsumptionGovernment"] = (
                torch.ones_like(prior_y) * scenario["GovernmentDemand"]
            )
        else:
            ratio = self.prior["PublicSectorBorrowingRequirement"] / prior_y
            threshold = params["PSBRThreshold"]

            z3 = (ratio > threshold).float()
            z4 = (ratio < -threshold).float()

            self.state["ConsumptionGovernment"] = (
                self.prior["ConsumptionGovernment"]
                - (z3 + z4)
                * params["FiscalAdjustmentSpeed"]
                * self.prior["PublicSectorBorrowingRequirement"]
                + scenario.get("GovernmentSpendingShock", 0.0)
            )

    ############################################################################
    # Public Sector Borrowing Requirement
    ############################################################################

    def psbr(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Calculate the public sector borrowing requirement.

        PSBR is the government deficit: total government outlays minus
        total government revenues.

        Equations
        ---------
        .. math::
            PSBR(t) = \left(G(t) + r_b(t) \cdot B_s(t-1) + BL_s(t-1)\right)
                - \left(T(t) + r_b(t) \cdot B_{CB}(t-1)\right)

        Dependency
        ----------
        - state: ConsumptionGovernment
        - state: InterestRateBills
        - prior: GovernmentBillStock
        - prior: GovernmentBondSupply
        - state: Taxes
        - prior: CentralBankBillStock

        Sets
        -----
        - PublicSectorBorrowingRequirement
        """
        self.state["PublicSectorBorrowingRequirement"] = (
            self.state["ConsumptionGovernment"]
            + self.state["InterestRateBills"] * self.prior["GovernmentBillStock"]
            + self.prior["GovernmentBondSupply"]
        ) - (
            self.state["Taxes"]
            + self.state["InterestRateBills"] * self.prior["CentralBankBillStock"]
        )
