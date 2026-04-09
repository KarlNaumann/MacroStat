"""
This module defines the forward and simulate behavior of the
Godley-Lavoie 2006 LP2 model (Endogenous bond price via
target-proportion mechanism with adaptive expectations).
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

from macrostat.models.GL06LP2.parameters import ParametersGL06LP2
from macrostat.models.GL06LP2.scenarios import ScenariosGL06LP2
from macrostat.models.GL06LP2.variables import VariablesGL06LP2
from macrostat.models.GL06LP.behavior import BehaviorGL06LP as _BehaviorGL06LP

logger = logging.getLogger(__name__)


class BehaviorGL06LP2(_BehaviorGL06LP):
    """Behavior class for the Godley-Lavoie 2006 LP2 model.

    Extends Model LP by making the bond price endogenous through a
    target-proportion mechanism (equations 5.24A–5.27) and replacing
    static expectations with adaptive expectations for the expected
    bond price (equation 5.20B).
    """

    version = "GL06LP2"

    def __init__(
        self,
        parameters: ParametersGL06LP2 | None = None,
        scenarios: ScenariosGL06LP2 | None = None,
        variables: VariablesGL06LP2 | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        """Initialize the behavior of the Godley-Lavoie 2006 LP2 model.

        Parameters
        ----------
        parameters : ParametersGL06LP2 | None
            The parameters of the model.
        scenarios : ScenariosGL06LP2 | None
            The scenarios of the model.
        variables : VariablesGL06LP2 | None
            The variables of the model.
        scenario : int
            The scenario to use for the model.
        debug : bool
            Whether to enable debug mode.
        """

        if parameters is None:
            parameters = ParametersGL06LP2()
        if scenarios is None:
            scenarios = ScenariosGL06LP2()
        if variables is None:
            variables = VariablesGL06LP2()

        # Bypass _BehaviorGL06LP.__init__ to pass our own types
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
        r"""Initialize the behavior of the Godley-Lavoie 2006 LP2 model.

        Extends LP initialization by adding TargetProportion and setting
        the initial bond price from the scenario.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                TP(0) = 0 \\
                p_{bl}(0) = p_{bl}^{\text{init}}
            \end{align}

        Sets
        -----
        All LP variables to zero, plus TargetProportion.
        BondPrice is set to BondPriceInitial from the scenario.
        """
        super().initialize()
        ref = next(iter(self.state.values()))
        self.state["TargetProportion"] = torch.zeros_like(ref)
        # Set initial bond price from scenario so prior is available
        # self.scenarios is a ParameterDict; take the first element
        self.state["BondPrice"] = self.scenarios["BondPriceInitial"][0]
        self.state["ExpectedBondPrice"] = self.state["BondPrice"].clone()

    ############################################################################
    # Step
    ############################################################################

    def step(self, **kwargs):
        """Step function of the Godley-Lavoie 2006 LP2 model."""

        # 1. Scenario exogenous variables
        self.set_government_demand(**kwargs)
        self.set_interest_rate_bills(**kwargs)

        # 2. Endogenous bond price via target-proportion mechanism
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

    ############################################################################
    # Endogenous Bond Price (replaces set_bond_price from LP)
    ############################################################################

    def target_proportion(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the target proportion of bonds in household govt debt.

        The target proportion is the ratio of the value of long-term
        bonds to the total value of bonds and bills held by households.

        Equations
        ---------
        .. math::
            TP(t) = \frac{BL_{h}(t-1) \cdot p_{bl}(t-1)}
                         {BL_{h}(t-1) \cdot p_{bl}(t-1) + B_{h}(t-1)}

        Dependency
        ----------
        - prior: HouseholdBondStock
        - prior: BondPrice
        - prior: HouseholdBillStock

        Sets
        -----
        - TargetProportion
        """
        bond_value = self.prior["HouseholdBondStock"] * self.prior["BondPrice"]
        total = bond_value + self.prior["HouseholdBillStock"]

        safe_total = torch.where(total.abs() > 1e-10, total, torch.ones_like(total))
        self.state["TargetProportion"] = torch.where(
            total.abs() > 1e-10,
            bond_value / safe_total,
            torch.zeros_like(total),
        )

    def compute_bond_price(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Compute the endogenous bond price via the target-proportion mechanism.

        When the target proportion exceeds the upper bound, the Treasury
        lets bond prices drift upwards. When below the lower bound, prices
        drift downwards.

        Equations
        ---------
        .. math::
            z_1 = 1 \text{ iff } TP > \text{top}

            z_2 = 1 \text{ iff } TP < \text{bot}

            p_{bl}(t) = (1 + z_1 \cdot \beta - z_2 \cdot \beta)
                \cdot p_{bl}(t-1) + \text{add1}

        Dependency
        ----------
        - state: TargetProportion
        - prior: BondPrice
        - params: BondPriceAdjustmentStep
        - params: TargetProportionUpper
        - params: TargetProportionLower
        - scenario: BondPriceShock

        Sets
        -----
        - BondPrice
        """
        tp = self.state["TargetProportion"]
        top = params["TargetProportionUpper"]
        bot = params["TargetProportionLower"]
        beta = params["BondPriceAdjustmentStep"]

        z1 = torch.where(
            tp > top,
            torch.ones_like(self.prior["BondPrice"]),
            torch.zeros_like(self.prior["BondPrice"]),
        )
        z2 = torch.where(
            tp < bot,
            torch.ones_like(self.prior["BondPrice"]),
            torch.zeros_like(self.prior["BondPrice"]),
        )
        self.state["BondPrice"] = (1.0 + z1 * beta - z2 * beta) * self.prior[
            "BondPrice"
        ] + scenario.get("BondPriceShock", 0.0)

    ############################################################################
    # Adaptive Expectations (replaces static from LP)
    ############################################################################

    def expected_bond_price(
        self, t: int, scenario: dict, params: dict | None = None, **kwargs
    ):
        r"""Calculate the expected bond price using adaptive expectations.

        The expected bond price adjusts towards the actual bond price
        with an error-correction mechanism.

        Equations
        ---------
        .. math::
            p_{bl}^e(t) = p_{bl}^e(t-1)
                - \beta_e \cdot (p_{bl}^e(t-1) - p_{bl}(t))
                + \text{add}

        Dependency
        ----------
        - prior: ExpectedBondPrice
        - state: BondPrice
        - params: ExpectationAdjustmentSpeed
        - scenario: ExpectedBondPriceShock

        Sets
        -----
        - ExpectedBondPrice
        """
        self.state["ExpectedBondPrice"] = (
            self.prior["ExpectedBondPrice"]
            - params["ExpectationAdjustmentSpeed"]
            * (self.prior["ExpectedBondPrice"] - self.state["BondPrice"])
            + scenario.get("ExpectedBondPriceShock", 0.0)
        )
