"""
Behavior (simulation logic) for the Pichler et al. (2022) Dynamic IO model.

Implements the daily-timestep dynamic disequilibrium input-output model
with labour adjustment, Muellbauer consumption, inventory-based ordering,
partially binding Leontief production, and proportional rationing.

Equation numbers throughout this module refer to Pichler, Pangallo, del
Rio-Chanona, Lafond & Farmer (2022). Paper symbols (e.g. :math:`A_{ji}`,
:math:`S_{ji,t}`) map to descriptive code variable names; see
:doc:`/models/PichlerEtAl2022DIO/notation` for the full table.
"""

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import logging

import torch

from macrostat.core.behavior import Behavior

from .parameters import ParametersPichlerEtAl2022DIO
from .scenarios import ScenariosPichlerEtAl2022DIO
from .variables import VariablesPichlerEtAl2022DIO

logger = logging.getLogger(__name__)


class BehaviorPichlerEtAl2022DIO(Behavior):
    """Behavior for the Pichler et al. (2022) Dynamic IO model.

    The ``step`` method executes the following sub-steps each period:

    1. ``hire_fire`` -- sluggish labour adjustment.
    2. ``productive_capacity`` -- labour-scaled capacity.
    3. ``consumption_demand`` -- Muellbauer consumption function.
    4. ``intermediate_orders`` -- inventory-gap ordering.
    5. ``aggregate_demand`` -- total demand aggregation.
    6. ``compute_production`` -- min(capacity, input limit, demand).
    7. ``rationing`` -- proportional rationing of output.
    8. ``inventory_update`` -- inventory accumulation.
    9. ``accounting`` -- profits and household savings.

    Equation numbers cited in sub-method summaries refer to Pichler et al.
    (2022). Paper-symbol ↔ code-variable mapping is documented in
    :doc:`/models/PichlerEtAl2022DIO/notation`.

    Parameters
    ----------
    parameters : ParametersPichlerEtAl2022DIO or None
        Model parameters.  If ``None``, a default instance is created.
    scenarios : ScenariosPichlerEtAl2022DIO or None
        Scenario definitions.  If ``None``, defaults are created.
    variables : VariablesPichlerEtAl2022DIO or None
        Variable definitions.  If ``None``, defaults are created.
    scenario : int
        Index of the active scenario.
    debug : bool
        If ``True``, enable additional logging.
    """

    version = "PichlerEtAl2022DIO"

    def __init__(
        self, parameters=None, scenarios=None, variables=None, scenario=0, debug=False
    ):
        if parameters is None:
            parameters = ParametersPichlerEtAl2022DIO()
        if scenarios is None:
            scenarios = ScenariosPichlerEtAl2022DIO(parameters=parameters)
        if variables is None:
            variables = VariablesPichlerEtAl2022DIO(parameters=parameters)

        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            scenario=scenario,
            debug=debug,
        )

        match self.hyper["production_function"]:
            case "leontief":
                self.production = self.production_leontief
            case "strongly_critical":
                self.production = self.production_strongly_critical
            case "half_critical":
                self.production = self.production_half_critical
            case "weakly_critical":
                self.production = self.production_weakly_critical
            case "linear":
                self.production = self.production_linear
            case _:
                raise ValueError(
                    f"Unknown production function: {self.hyper['production_function']}"
                )

    def initialize(self):
        """Set the initial state from data parameters.

        Populates all state variables with their steady-state values
        derived from the loaded IO data, and caches reference quantities
        used during the simulation (``_x0``, ``_l0``, ``_xcap0``,
        ``_S_tar``, ``_mpc``).
        """
        self.state["GrossOutput"] = self.params["InitialGrossOutput"].clone()
        self.state["AggregateDemand"] = self.params["InitialGrossOutput"].clone()
        self.state["LabourCompensation"] = self.params[
            "InitialLabourCompensation"
        ].clone()
        self.state["ConsumptionDemand"] = self.params[
            "InitialHouseholdConsumption"
        ].clone()
        self.state["RealizedConsumption"] = self.params[
            "InitialHouseholdConsumption"
        ].clone()
        self.state["Profits"] = self.params["InitialProfits"].clone()
        self.state["ProductiveCapacity"] = self.params["InitialGrossOutput"].clone()
        self.state["IntermediateConsumption"] = self.params[
            "IntermediateConsumptionMatrix"
        ].clone()
        self.state["IntermediateOrders"] = self.params[
            "IntermediateConsumptionMatrix"
        ].clone()
        self.state["TotalConsumptionDemand"] = (
            self.params["InitialHouseholdConsumption"].sum().unsqueeze(0)
        )
        self.state["InputCapacity"] = self.params["InitialGrossOutput"].clone()
        inventory_target = self.params["IntermediateConsumptionMatrix"] * self.params[
            "InventoryTargetDays"
        ].unsqueeze(0)
        self.state["Inventories"] = inventory_target.clone()

        self._x0 = self.params["InitialGrossOutput"].clone()
        self._l0 = self.params["InitialLabourCompensation"].clone()
        self._xcap0 = self._x0.clone()
        self._S_tar = inventory_target
        self._mpc = self.params["InitialHouseholdConsumption"].sum() / self._l0.sum()

    def step(self, t, scenario, params, **kwargs):
        """Execute one daily time step of the model.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters (possibly shock-adjusted).
        **kwargs
            Unused; accepted for compatibility with base class.
        """
        self.hire_fire(t, scenario, params)
        self.productive_capacity(t, scenario, params)
        self.consumption_demand(t, scenario, params)
        self.intermediate_orders(t, scenario, params)
        self.aggregate_demand(t, scenario, params)
        self.compute_production(t, scenario, params)
        self.rationing(t, scenario, params)
        self.inventory_update(t, scenario, params)
        self.accounting(t, scenario, params)

    # ------------------------------------------------------------------
    # Step sub-methods
    # ------------------------------------------------------------------

    def hire_fire(self, t, scenario, params):
        r"""Sluggish labour adjustment towards a target workforce.

        Firms adjust their labour force depending on which production
        constraint is binding.  If capacity is binding, the firm tries to
        hire; if demand or input constraints bind, it fires.  Adjustment is
        sluggish -- firms can only move a fraction of the way toward their
        target each period.  During lockdown, labour is additionally capped
        by the exogenous supply shock. Replicates Eqs. 19-20.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                \Delta l_{i,t} &= \frac{l_{i,0}}{x_{i,0}}
                    \left[\min\{x^{\text{inp}}_{i,t},\, d_{i,t}\}
                          - x^{\text{cap}}_{i,t}\right] \\
                l_{i,t} &= \begin{cases}
                    l_{i,t-1} + \gamma_H \Delta l_{i,t}
                        & \text{if } \Delta l_{i,t} \ge 0 \\
                    l_{i,t-1} + \gamma_F \Delta l_{i,t}
                        & \text{if } \Delta l_{i,t} < 0
                \end{cases}
            \end{align}

        Dependency
        ----------
        - prior: ProductiveCapacity, InputCapacity, AggregateDemand, LabourCompensation
        - params: HiringRate, FiringRate
        - scenario: SupplyShock

        Sets
        ----
        - LabourCompensation

        Notes
        -----
        See :doc:`/models/PichlerEtAl2022DIO/notation` for symbol definitions.
        """
        if not self.hyper["hiring_firing"]:
            supply_shock = scenario.get("SupplyShock", torch.zeros_like(self._l0))
            self.state["LabourCompensation"] = self._l0 * (1.0 - supply_shock)
            return

        supply_shock = scenario.get("SupplyShock", torch.zeros_like(self._l0))
        labor_share = self._l0 / torch.where(
            self._x0 != 0, self._x0, torch.ones_like(self._x0)
        )

        hire_target = torch.min(
            self.prior["InputCapacity"], self.prior["AggregateDemand"]
        )
        delta_labour = labor_share * (hire_target - self.prior["ProductiveCapacity"])

        gamma = torch.where(
            delta_labour >= 0,
            params["HiringRate"],
            params["FiringRate"],
        )

        self.state["LabourCompensation"] = torch.min(
            self._l0 * (1.0 - supply_shock),
            self.prior["LabourCompensation"] + gamma * delta_labour,
        )

    def productive_capacity(self, t, scenario, params):
        r"""Labour-scaled production capacity.

        Each industry has a finite production capacity that scales linearly
        with available labour relative to the pre-pandemic baseline.
        Initially every industry employs :math:`l_{i,0}` workers and
        produces at full capacity :math:`x^{\text{cap}}_{i,0} = x_{i,0}`.
        Replicates Eq. 7.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                x^{\text{cap}}_{i,t} = \frac{l_{i,t}}{l_{i,0}}\,
                    x^{\text{cap}}_{i,0}
            \end{align}

        Dependency
        ----------
        - state: LabourCompensation

        Sets
        ----
        - ProductiveCapacity

        Notes
        -----
        See :doc:`/models/PichlerEtAl2022DIO/notation` for symbol definitions.
        """
        safe_l0 = torch.where(self._l0 != 0, self._l0, torch.ones_like(self._l0))
        self.state["ProductiveCapacity"] = (
            self.state["LabourCompensation"] / safe_l0
        ) * self._xcap0

    def consumption_demand(self, t, scenario, params):
        r"""Muellbauer consumption function with fear-of-infection.

        Total household consumption demand follows an adapted version of
        Muellbauer (2020), combining persistence of past consumption with
        current and permanent labour income.  A fear-of-infection factor
        :math:`(1 - \tilde{\epsilon}^D_t)` scales aggregate demand.
        Consumption is allocated across industries via time-varying
        preference coefficients. Replicates Eqs. 5-6.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                \tilde{c}^d_t &= (1 - \tilde{\epsilon}^D_t)\,
                    \exp\!\left(
                        \rho \log \tilde{c}^d_{t-1}
                        + \frac{1-\rho}{2}\log(m\tilde{l}_t)
                        + \frac{1-\rho}{2}\log(m\tilde{l}^p_t)
                    \right) \\
                c^d_{i,t} &= \theta_{i,t}\,\tilde{c}^d_t
            \end{align}

        Dependency
        ----------
        - state: LabourCompensation
        - prior: TotalConsumptionDemand, ConsumptionDemand
        - params: ConsumptionPersistence, BenefitRate
        - scenario: PermanentIncomeExpectation, FearOfInfection, DemandPreferences

        Sets
        ----
        - TotalConsumptionDemand
        - ConsumptionDemand

        Notes
        -----
        See :doc:`/models/PichlerEtAl2022DIO/notation` for symbol definitions.
        """
        initial_labour_total = self._l0.sum()
        labour_total_effective = (
            params["BenefitRate"] * initial_labour_total
            + (1.0 - params["BenefitRate"]) * self.state["LabourCompensation"].sum()
        )

        total_consumption_demand_prior = self.prior["TotalConsumptionDemand"].squeeze()
        log_total_consumption_demand = torch.log(
            torch.clamp(total_consumption_demand_prior, min=1e-10)
        )
        log_labour = torch.log(
            torch.clamp(self._mpc * labour_total_effective, min=1e-10)
        )
        log_labour_permanent = torch.log(
            torch.clamp(
                self._mpc
                * initial_labour_total
                * scenario.get("PermanentIncomeExpectation", torch.tensor(1.0)),
                min=1e-10,
            )
        )

        total_consumption_demand = torch.exp(
            params["ConsumptionPersistence"] * log_total_consumption_demand
            + (1.0 - params["ConsumptionPersistence"]) / 2.0 * log_labour
            + (1.0 - params["ConsumptionPersistence"]) / 2.0 * log_labour_permanent
        )
        self.state["TotalConsumptionDemand"] = total_consumption_demand.unsqueeze(0)

        self.state["ConsumptionDemand"] = (
            scenario.get(
                "DemandPreferences",
                self.prior["ConsumptionDemand"]
                / torch.clamp(total_consumption_demand_prior, min=1e-10),
            )
            * total_consumption_demand
            * (1.0 - scenario.get("FearOfInfection", torch.tensor(0.0)))
        )

    def intermediate_orders(self, t, scenario, params):
        r"""Inventory-gap ordering of intermediate inputs.

        Each industry places orders for intermediate inputs based on two
        components: (1) a naive expectation that demand will equal last
        period's level, scaled by the technical coefficients; and (2) a
        correction term that moves inventories toward a target of
        :math:`n_i` days of each input, at a speed governed by
        :math:`\tau`. Replicates Eq. 4.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                O_{ji,t} = A_{ji}\,d_{i,t-1}
                    + \frac{1}{\tau}\left(n_i Z_{ji,0} - S_{ji,t-1}\right)
            \end{align}

        Dependency
        ----------
        - prior: AggregateDemand, Inventories
        - params: TechnicalCoefficients, InventoryAdjustmentSpeed

        Sets
        ----
        - IntermediateOrders

        Notes
        -----
        See :doc:`/models/PichlerEtAl2022DIO/notation` for symbol definitions.
        """
        orders = (
            params["TechnicalCoefficients"] * self.prior["AggregateDemand"].unsqueeze(0)
            + (self._S_tar - self.prior["Inventories"])
            / params["InventoryAdjustmentSpeed"]
        )
        self.state["IntermediateOrders"] = torch.clamp(orders, min=0.0)

    def aggregate_demand(self, t, scenario, params):
        r"""Total demand aggregation.

        Total demand for the output of industry *i* is the sum of
        intermediate orders from all other industries, household
        consumption demand, and exogenous other final demand (government,
        exports, investment). Replicates Eq. 3.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                d_{i,t} = \sum_{j=1}^{N} O_{ij,t} + c^d_{i,t} + f^d_{i,t}
            \end{align}

        Dependency
        ----------
        - state: ConsumptionDemand, IntermediateOrders
        - params: InitialOtherFinalDemand
        - scenario: OtherFinalDemand

        Sets
        ----
        - AggregateDemand

        Notes
        -----
        See :doc:`/models/PichlerEtAl2022DIO/notation` for symbol definitions.
        """
        other_final_demand = scenario.get(
            "OtherFinalDemand",
            params["InitialOtherFinalDemand"],
        )
        self.state["AggregateDemand"] = (
            self.state["ConsumptionDemand"]
            + self.state["IntermediateOrders"].sum(dim=1)
            + other_final_demand
        )

    def compute_production(self, t, scenario, params):
        r"""Production function and output-level choice.

        Realized output is the minimum of three constraints: labour
        capacity, input-based capacity (from inventories and the chosen
        production function), and demand.  The input-based capacity depends
        on the ``production_function`` hyperparameter; five functional forms
        range from Leontief (all inputs binding) through partially binding
        Leontief variants to linear (perfect substitutes).  The partially
        binding Leontief distinguishes critical, important, and non-critical
        inputs based on an industry analyst survey. Replicates Eqs. 8-14.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                x_{i,t} &= \min\{x^{\text{cap}}_{i,t},\;
                                 x^{\text{inp}}_{i,t},\;
                                 d_{i,t}\} \\[6pt]
                x^{\text{inp}}_{i,t} &= \begin{cases}
                    \displaystyle\min_{\{j:\,A_{ji}>0\}}
                        \frac{S_{ji,t}}{A_{ji}}
                        & \text{leontief} \\
                    \displaystyle\min_{j \in \mathcal{V}_i \cup \mathcal{U}_i}
                        \frac{S_{ji,t}}{A_{ji}}
                        & \text{strongly\_critical} \\
                    \displaystyle\min\!\left\{
                        \min_{j\in\mathcal{V}_i}\frac{S_{ji,t}}{A_{ji}},\;
                        \tfrac{1}{2}\!\left(
                            \min_{k\in\mathcal{U}_i}\frac{S_{ki,t}}{A_{ki}}
                            + x^{\text{cap}}_{i,0}
                        \right)
                    \right\}
                        & \text{half\_critical} \\
                    \displaystyle\min_{j \in \mathcal{V}_i}
                        \frac{S_{ji,t}}{A_{ji}}
                        & \text{weakly\_critical} \\
                    \displaystyle\sum_j S_{ji,t} \big/ \sum_j A_{ji}
                        & \text{linear}
                \end{cases}
            \end{align}

        Dependency
        ----------
        - state: ProductiveCapacity, AggregateDemand
        - prior: Inventories
        - params: TechnicalCoefficients, CriticalInputMatrix

        Sets
        ----
        - InputCapacity
        - GrossOutput

        Notes
        -----
        The dispatcher selects one of five production-function variants by
        the ``production_function`` hyperparameter, set at construction:

        - ``leontief`` -- all inputs with positive technical coefficient bind.
        - ``strongly_critical`` -- critical and important inputs bind.
        - ``half_critical`` -- critical inputs bind; important inputs at
          half capacity.
        - ``weakly_critical`` -- only critical inputs bind.
        - ``linear`` -- perfect substitution across inputs.

        See :doc:`/models/PichlerEtAl2022DIO/notation` for symbol definitions.
        """
        self.state["InputCapacity"] = self.production(
            self.prior["Inventories"],
            params["TechnicalCoefficients"],
            params["CriticalInputMatrix"],
            self._xcap0,
        )
        self.state["GrossOutput"] = torch.min(
            torch.min(self.state["ProductiveCapacity"], self.state["InputCapacity"]),
            self.state["AggregateDemand"],
        )

    def rationing(self, t, scenario, params):
        r"""Proportional rationing of output across buyers.

        When output falls short of demand, industry *i* rations its output
        proportionally across all customers.  Each buyer receives a share
        of their order equal to the ratio of output to demand.  An optional
        ``firm_priority`` mode gives precedence to intermediate demand over
        final consumption. Replicates Eqs. 15-17.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                Z_{ji,t} &= O_{ji,t}\,\frac{x_{j,t}}{d_{j,t}}, &
                c_{i,t}  &= c^d_{i,t}\,\frac{x_{i,t}}{d_{i,t}}, &
                f_{i,t}  &= f^d_{i,t}\,\frac{x_{i,t}}{d_{i,t}}
            \end{align}

        Dependency
        ----------
        - state: GrossOutput, AggregateDemand, IntermediateOrders, ConsumptionDemand

        Sets
        ----
        - IntermediateConsumption
        - RealizedConsumption

        Notes
        -----
        See :doc:`/models/PichlerEtAl2022DIO/notation` for symbol definitions.
        """
        safe_demand = torch.where(
            self.state["AggregateDemand"] != 0,
            self.state["AggregateDemand"],
            torch.ones_like(self.state["AggregateDemand"]),
        )
        share = self.state["GrossOutput"] / safe_demand

        if not self.hyper["firm_priority"]:
            self.state["IntermediateConsumption"] = self.state[
                "IntermediateOrders"
            ] * share.unsqueeze(1)
            self.state["RealizedConsumption"] = self.state["ConsumptionDemand"] * share
        else:
            row_sum_orders = self.state["IntermediateOrders"].sum(dim=1)
            safe_row_sum = torch.where(
                row_sum_orders != 0,
                row_sum_orders,
                torch.ones_like(self.state["GrossOutput"]),
            )
            share_firm = torch.min(
                torch.ones_like(self.state["GrossOutput"]),
                self.state["GrossOutput"] / safe_row_sum,
            )
            self.state["IntermediateConsumption"] = self.state[
                "IntermediateOrders"
            ] * share_firm.unsqueeze(1)

            demand_remaining = self.state["AggregateDemand"] - row_sum_orders
            safe_remaining = torch.where(
                demand_remaining != 0,
                demand_remaining,
                torch.ones_like(demand_remaining),
            )
            share_final = torch.clamp(
                (
                    self.state["GrossOutput"]
                    - self.state["IntermediateConsumption"].sum(dim=1)
                )
                / safe_remaining,
                min=0.0,
                max=1.0,
            )
            self.state["RealizedConsumption"] = (
                self.state["ConsumptionDemand"] * share_final
            )

    def inventory_update(self, t, scenario, params):
        r"""Inventory accumulation from deliveries minus usage.

        After production and rationing, each industry updates its inventory
        of every input.  Inventories increase by deliveries received and
        decrease by inputs consumed in production (at rates given by the
        technical coefficients).  A floor of zero prevents negative stocks
        for non-critical inputs that may be fully depleted. Replicates
        Eq. 18.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                S_{ji,t+1} = \max\{S_{ji,t} + Z_{ji,t}
                    - A_{ji}\,x_{i,t},\; 0\}
            \end{align}

        Dependency
        ----------
        - state: GrossOutput, IntermediateConsumption
        - prior: Inventories
        - params: TechnicalCoefficients

        Sets
        ----
        - Inventories

        Notes
        -----
        See :doc:`/models/PichlerEtAl2022DIO/notation` for symbol definitions.
        """
        used_inputs = params["TechnicalCoefficients"] * self.state[
            "GrossOutput"
        ].unsqueeze(0)
        self.state["Inventories"] = torch.clamp(
            self.prior["Inventories"]
            + self.state["IntermediateConsumption"]
            - used_inputs,
            min=0.0,
        )

    def accounting(self, t, scenario, params):
        r"""Firm profits and household savings.

        Industry profits are total output minus intermediate purchases,
        labour compensation, and other expenses (taxes, imports).
        Household savings are computed as total income minus total realized
        consumption (including non-modeled import and tax expenditures).
        Replicates Eq. 2.

        Parameters
        ----------
        t : int
            Current timestep index.
        scenario : dict[str, torch.Tensor]
            Scenario variable values for this timestep.
        params : torch.nn.ParameterDict
            Model parameters.

        Equations
        ---------
        .. math::
            :nowrap:

            \begin{align}
                \pi_{i,t} = x_{i,t} - \sum_{j=1}^{N} Z_{ji,t}
                    - l_{i,t} - e_{i,t}
            \end{align}

        Dependency
        ----------
        - state: GrossOutput, IntermediateConsumption, LabourCompensation, RealizedConsumption, Profits
        - params: OtherCostCoefficients, HouseholdOtherCostCoefficient

        Sets
        ----
        - Profits
        - Savings

        Notes
        -----
        See :doc:`/models/PichlerEtAl2022DIO/notation` for symbol definitions.
        """
        self.state["Profits"] = (
            self.state["GrossOutput"]
            - self.state["IntermediateConsumption"].sum(dim=0)
            - self.state["LabourCompensation"]
            - params["OtherCostCoefficients"] * self.state["GrossOutput"]
        )

        household_other_cost_coef = params["HouseholdOtherCostCoefficient"].squeeze()
        extra_expenditure = (
            household_other_cost_coef
            / (1.0 - household_other_cost_coef)
            * self.state["RealizedConsumption"].sum()
        )

        self.state["Savings"] = (
            self.state["Profits"].sum()
            + self.state["LabourCompensation"].sum()
            - self.state["RealizedConsumption"].sum()
            - extra_expenditure
        ).unsqueeze(0)

    # ------------------------------------------------------------------
    # Production functions
    # ------------------------------------------------------------------

    def production_leontief(
        self,
        inventory_matrix,
        technical_coefficients,
        critical_input_matrix,
        capacity_initial,
    ):
        """Standard Leontief: all inputs with positive ``A`` are binding.

        Parameters
        ----------
        inventory_matrix : torch.Tensor
            ``(N, N)`` inventory matrix :math:`S_{ji,t}`.
        technical_coefficients : torch.Tensor
            ``(N, N)`` technical coefficients matrix :math:`A_{ji}`.
        critical_input_matrix : torch.Tensor
            ``(N, N)`` critical-input matrix (unused in this variant).
        capacity_initial : torch.Tensor
            ``(N,)`` baseline productive capacity :math:`x^{\\text{cap}}_{i,0}`.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        n_sectors = inventory_matrix.shape[1]
        input_capacity = torch.full((n_sectors,), float("inf"))
        for k in range(n_sectors):
            mask = technical_coefficients[:, k] > 0
            if mask.any():
                safe_a = torch.where(
                    mask,
                    technical_coefficients[:, k],
                    torch.ones_like(technical_coefficients[:, k]),
                )
                ratios = inventory_matrix[:, k] / safe_a
                ratios = torch.where(mask, ratios, torch.tensor(float("inf")))
                input_capacity[k] = ratios.min()
        return input_capacity

    def production_strongly_critical(
        self,
        inventory_matrix,
        technical_coefficients,
        critical_input_matrix,
        capacity_initial,
    ):
        """Critical and important inputs are binding (score >= 0.5).

        Parameters
        ----------
        inventory_matrix : torch.Tensor
            ``(N, N)`` inventory matrix.
        technical_coefficients : torch.Tensor
            ``(N, N)`` technical coefficients matrix.
        critical_input_matrix : torch.Tensor
            ``(N, N)`` critical-input matrix.
        capacity_initial : torch.Tensor
            ``(N,)`` baseline productive capacity.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        n_sectors = inventory_matrix.shape[1]
        input_capacity = torch.full((n_sectors,), float("inf"))
        for k in range(n_sectors):
            mask = critical_input_matrix[:, k] >= 0.5
            if mask.any():
                safe_a = torch.where(
                    mask,
                    technical_coefficients[:, k],
                    torch.ones_like(technical_coefficients[:, k]),
                )
                ratios = inventory_matrix[:, k] / safe_a
                ratios = torch.where(mask, ratios, torch.tensor(float("inf")))
                input_capacity[k] = ratios.min()
        return input_capacity

    def production_half_critical(
        self,
        inventory_matrix,
        technical_coefficients,
        critical_input_matrix,
        capacity_initial,
    ):
        """Critical inputs fully binding; important inputs half-binding.

        Parameters
        ----------
        inventory_matrix : torch.Tensor
            ``(N, N)`` inventory matrix.
        technical_coefficients : torch.Tensor
            ``(N, N)`` technical coefficients matrix.
        critical_input_matrix : torch.Tensor
            ``(N, N)`` critical-input matrix.
        capacity_initial : torch.Tensor
            ``(N,)`` baseline productive capacity.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        n_sectors = inventory_matrix.shape[1]
        input_capacity = torch.full((n_sectors,), float("inf"))
        for k in range(n_sectors):
            critical = critical_input_matrix[:, k] > 0.5
            important = critical_input_matrix[:, k] == 0.5

            vals = []
            if critical.any():
                safe_a = torch.where(
                    critical,
                    technical_coefficients[:, k],
                    torch.ones_like(technical_coefficients[:, k]),
                )
                ratios = inventory_matrix[:, k] / safe_a
                ratios = torch.where(critical, ratios, torch.tensor(float("inf")))
                vals.append(ratios.min())

            if important.any():
                safe_a = torch.where(
                    important,
                    technical_coefficients[:, k],
                    torch.ones_like(technical_coefficients[:, k]),
                )
                ratios = (inventory_matrix[:, k] / safe_a) * 0.5 + capacity_initial[
                    k
                ] / 2.0
                ratios = torch.where(important, ratios, torch.tensor(float("inf")))
                vals.append(ratios.min())

            if vals:
                input_capacity[k] = torch.stack(vals).min()
        return input_capacity

    def production_weakly_critical(
        self,
        inventory_matrix,
        technical_coefficients,
        critical_input_matrix,
        capacity_initial,
    ):
        """Only critical inputs (score > 0.5) are binding.

        Parameters
        ----------
        inventory_matrix : torch.Tensor
            ``(N, N)`` inventory matrix.
        technical_coefficients : torch.Tensor
            ``(N, N)`` technical coefficients matrix.
        critical_input_matrix : torch.Tensor
            ``(N, N)`` critical-input matrix.
        capacity_initial : torch.Tensor
            ``(N,)`` baseline productive capacity.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        n_sectors = inventory_matrix.shape[1]
        input_capacity = torch.full((n_sectors,), float("inf"))
        for k in range(n_sectors):
            mask = critical_input_matrix[:, k] > 0.5
            if mask.any():
                safe_a = torch.where(
                    mask,
                    technical_coefficients[:, k],
                    torch.ones_like(technical_coefficients[:, k]),
                )
                ratios = inventory_matrix[:, k] / safe_a
                ratios = torch.where(mask, ratios, torch.tensor(float("inf")))
                input_capacity[k] = ratios.min()
        return input_capacity

    def production_linear(
        self,
        inventory_matrix,
        technical_coefficients,
        critical_input_matrix,
        capacity_initial,
    ):
        """Linear production: all inputs are perfect substitutes.

        Parameters
        ----------
        inventory_matrix : torch.Tensor
            ``(N, N)`` inventory matrix.
        technical_coefficients : torch.Tensor
            ``(N, N)`` technical coefficients matrix.
        critical_input_matrix : torch.Tensor
            ``(N, N)`` critical-input matrix (unused in this variant).
        capacity_initial : torch.Tensor
            ``(N,)`` baseline productive capacity.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        col_sum_technical_coefficients = technical_coefficients.sum(dim=0)
        col_sum_inventory_matrix = inventory_matrix.sum(dim=0)
        safe_denom = torch.where(
            col_sum_technical_coefficients != 0,
            col_sum_technical_coefficients,
            torch.ones_like(col_sum_technical_coefficients),
        )
        input_capacity = col_sum_inventory_matrix / safe_denom
        input_capacity = torch.where(
            col_sum_technical_coefficients != 0,
            input_capacity,
            torch.tensor(float("inf")),
        )
        return input_capacity
