"""
Behavior (simulation logic) for the Pichler et al. (2022) Dynamic IO model.

Implements the daily-timestep dynamic disequilibrium input-output model
with labor adjustment, Muellbauer consumption, inventory-based ordering,
partially binding Leontief production, and proportional rationing.
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

    1. ``hire_fire`` -- sluggish labour adjustment (Eq. 19--20)
    2. ``productive_capacity`` -- labour-scaled capacity (Eq. 7)
    3. ``consumption_demand`` -- Muellbauer consumption function (Eq. 5--6)
    4. ``intermediate_orders`` -- inventory-gap ordering (Eq. 4)
    5. ``aggregate_demand`` -- total demand aggregation (Eq. 3)
    6. ``compute_production`` -- min(capacity, input limit, demand) (Eq. 8--14)
    7. ``rationing`` -- proportional rationing of output (Eq. 15--17)
    8. ``inventory_update`` -- inventory accumulation (Eq. 18)
    9. ``accounting`` -- profits and household savings (Eq. 2)

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
        p = self.params

        self.state["GrossOutput"] = p["InitialGrossOutput"].clone()
        self.state["AggregateDemand"] = p["InitialGrossOutput"].clone()
        self.state["LabourCompensation"] = p["InitialLabourCompensation"].clone()
        self.state["ConsumptionDemand"] = p["InitialHouseholdConsumption"].clone()
        self.state["RealizedConsumption"] = p["InitialHouseholdConsumption"].clone()
        self.state["Profits"] = p["InitialProfits"].clone()
        self.state["ProductiveCapacity"] = p["InitialGrossOutput"].clone()

        Z = p["IntermediateConsumptionMatrix"]
        n = p["InventoryTargetDays"]
        self.state["Inventories"] = Z * n.unsqueeze(0)
        self.state["IntermediateConsumption"] = Z.clone()
        self.state["IntermediateOrders"] = Z.clone()

        self.state["TotalConsumptionDemand"] = (
            p["InitialHouseholdConsumption"].sum().unsqueeze(0)
        )

        self.state["InputCapacity"] = p["InitialGrossOutput"].clone()

        x0 = p["InitialGrossOutput"]
        l0 = p["InitialLabourCompensation"]
        self._x0 = x0.clone()
        self._l0 = l0.clone()
        self._xcap0 = x0.clone()
        self._S_tar = Z * n.unsqueeze(0)
        self._mpc = p["InitialHouseholdConsumption"].sum() / l0.sum()

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
        by the exogenous supply shock.

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

        where:

        - :math:`l_{i,t}` is labour compensation in industry *i* at time *t*
        - :math:`l_{i,0}` is the initial (pre-pandemic) labour compensation
        - :math:`x_{i,0}` is the initial output level
        - :math:`x^{\text{inp}}_{i,t}` is the input-based production capacity
        - :math:`x^{\text{cap}}_{i,t}` is the labour-based production capacity
        - :math:`d_{i,t}` is total demand
        - :math:`\gamma_H` is the hiring adjustment rate
        - :math:`\gamma_F` is the firing adjustment rate
        """
        if not self.hyper["hiring_firing"]:
            supply_shock = scenario.get("SupplyShock", torch.zeros_like(self._l0))
            self.state["LabourCompensation"] = self._l0 * (1.0 - supply_shock)
            return

        supply_shock = scenario.get("SupplyShock", torch.zeros_like(self._l0))
        new_lcap = self._l0 * (1.0 - supply_shock)
        labor_share = self._l0 / torch.where(
            self._x0 != 0, self._x0, torch.ones_like(self._x0)
        )

        xcap_prior = self.prior["ProductiveCapacity"]
        xinp_prior = self.prior["InputCapacity"]
        d_prior = self.prior["AggregateDemand"]

        hire_target = torch.min(xinp_prior, d_prior)
        delta_l = labor_share * (hire_target - xcap_prior)

        hiring = delta_l >= 0
        gamma = torch.where(
            hiring,
            params["HiringRate"],
            params["FiringRate"],
        )

        new_l = torch.min(
            new_lcap,
            self.prior["LabourCompensation"] + gamma * delta_l,
        )
        self.state["LabourCompensation"] = new_l

    def productive_capacity(self, t, scenario, params):
        r"""Labour-scaled production capacity.

        Each industry has a finite production capacity that scales linearly
        with available labour relative to the pre-pandemic baseline.
        Initially every industry employs :math:`l_{i,0}` workers and
        produces at full capacity :math:`x^{\text{cap}}_{i,0} = x_{i,0}`.

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

        where:

        - :math:`x^{\text{cap}}_{i,t}` is the labour-based production capacity of industry *i*
        - :math:`l_{i,t}` is the current labour compensation
        - :math:`l_{i,0}` is the initial labour compensation
        - :math:`x^{\text{cap}}_{i,0}` is the initial production capacity (equal to initial output)
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
        preference coefficients.

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

        where:

        - :math:`\tilde{c}^d_t` is aggregate consumption demand
        - :math:`\rho` is the persistence of consumption
        - :math:`m` is the propensity to consume final domestic goods out of labour income
        - :math:`\tilde{l}_t` is current aggregate labour income (effective, accounting for benefits)
        - :math:`\tilde{l}^p_t` is permanent income expectation
        - :math:`\tilde{\epsilon}^D_t` is the aggregate demand shock due to fear of infection
        - :math:`\theta_{i,t}` is the time-varying preference share for industry *i*
        - :math:`c^d_{i,t}` is consumption demand for the output of industry *i*
        """
        rho1 = params["ConsumptionPersistence"]
        rho0 = 1.0 - rho1
        benefits = params["BenefitRate"]
        mpc = self._mpc

        l1 = self._l0.sum()
        lt = self.state["LabourCompensation"].sum()
        lt_eff = benefits * l1 + (1.0 - benefits) * lt

        xi = scenario.get("PermanentIncomeExpectation", torch.tensor(1.0))
        eps = scenario.get("FearOfInfection", torch.tensor(0.0))

        Cdt_prior = self.prior["TotalConsumptionDemand"].squeeze()
        log_Ct = torch.log(torch.clamp(Cdt_prior, min=1e-10))
        log_lt = torch.log(torch.clamp(mpc * lt_eff, min=1e-10))
        log_ltp = torch.log(torch.clamp(mpc * l1 * xi, min=1e-10))

        Cdt = torch.exp(rho1 * log_Ct + rho0 / 2.0 * log_lt + rho0 / 2.0 * log_ltp)
        self.state["TotalConsumptionDemand"] = Cdt.unsqueeze(0)

        theta = scenario.get(
            "DemandPreferences",
            self.prior["ConsumptionDemand"] / torch.clamp(Cdt_prior, min=1e-10),
        )
        self.state["ConsumptionDemand"] = theta * Cdt * (1.0 - eps)

    def intermediate_orders(self, t, scenario, params):
        r"""Inventory-gap ordering of intermediate inputs.

        Each industry places orders for intermediate inputs based on two
        components: (1) a naive expectation that demand will equal last
        period's level, scaled by the technical coefficients; and (2) a
        correction term that moves inventories toward a target of
        :math:`n_i` days of each input, at a speed governed by
        :math:`\tau`.

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

        where:

        - :math:`O_{ji,t}` is the order from industry *i* to industry *j* for input *j*
        - :math:`A_{ji} = Z_{ji,0}/x_{i,0}` is the technical coefficient
        - :math:`d_{i,t-1}` is lagged total demand for industry *i*
        - :math:`n_i` is the target inventory in days for industry *i*
        - :math:`Z_{ji,0}` is the baseline intermediate consumption of input *j* by *i*
        - :math:`S_{ji,t-1}` is the current inventory of input *j* held by *i*
        - :math:`\tau` is the inventory adjustment speed (in days)
        """
        A = params["TechnicalCoefficients"]
        tau = params["InventoryAdjustmentSpeed"]
        d_prior = self.prior["AggregateDemand"]
        S_prior = self.prior["Inventories"]

        orders = A * d_prior.unsqueeze(0) + (self._S_tar - S_prior) / tau
        self.state["IntermediateOrders"] = torch.clamp(orders, min=0.0)

    def aggregate_demand(self, t, scenario, params):
        r"""Total demand aggregation.

        Total demand for the output of industry *i* is the sum of
        intermediate orders from all other industries, household
        consumption demand, and exogenous other final demand (government,
        exports, investment).

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

        where:

        - :math:`d_{i,t}` is total demand for the output of industry *i*
        - :math:`O_{ij,t}` is the intermediate order from industry *j* to industry *i*
        - :math:`c^d_{i,t}` is household consumption demand for good *i*
        - :math:`f^d_{i,t}` is exogenous other final demand for good *i*
        """
        fd_other = scenario.get(
            "OtherFinalDemand",
            params["InitialOtherFinalDemand"],
        )
        self.state["AggregateDemand"] = (
            self.state["ConsumptionDemand"]
            + self.state["IntermediateOrders"].sum(dim=1)
            + fd_other
        )

    def compute_production(self, t, scenario, params):
        r"""Production function and output-level choice.

        Realized output is the minimum of three constraints: labour
        capacity, input-based capacity (from inventories and the chosen
        production function), and demand.  The input-based capacity depends
        on the ``production_function`` hyperparameter.  Five functional
        forms are available, ranging from Leontief (all inputs binding)
        through partially binding Leontief variants to linear (perfect
        substitutes).  The partially binding Leontief distinguishes
        critical, important, and non-critical inputs based on an industry
        analyst survey.

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
        Output choice (Eq. 14):

        .. math::
            :nowrap:

            \begin{align}
                x_{i,t} = \min\{x^{\text{cap}}_{i,t},\;
                                 x^{\text{inp}}_{i,t},\;
                                 d_{i,t}\}
            \end{align}

        Leontief (Eq. 9):

        .. math::
            :nowrap:

            \begin{align}
                x^{\text{inp}}_{i,t} = \min_{\{j:\,A_{ji}>0\}}
                    \frac{S_{ji,t}}{A_{ji}}
            \end{align}

        Strongly-critical (Eq. 10):

        .. math::
            :nowrap:

            \begin{align}
                x^{\text{inp}}_{i,t} = \min_{j \in \mathcal{V}_i
                    \cup \mathcal{U}_i} \frac{S_{ji,t}}{A_{ji}}
            \end{align}

        Half-critical (Eq. 11):

        .. math::
            :nowrap:

            \begin{align}
                x^{\text{inp}}_{i,t} = \min_{\{j \in \mathcal{V}_i,\;
                    k \in \mathcal{U}_i\}} \left\{
                    \frac{S_{ji,t}}{A_{ji}},\;
                    \frac{1}{2}\!\left(
                        \frac{S_{ki,t}}{A_{ki}} + x^{\text{cap}}_{i,0}
                    \right)\right\}
            \end{align}

        Weakly-critical (Eq. 12):

        .. math::
            :nowrap:

            \begin{align}
                x^{\text{inp}}_{i,t} = \min_{j \in \mathcal{V}_i}
                    \frac{S_{ji,t}}{A_{ji}}
            \end{align}

        Linear (Eq. 13):

        .. math::
            :nowrap:

            \begin{align}
                x^{\text{inp}}_{i,t} = \frac{\sum_j S_{ji,t}}
                                            {\sum_j A_{ji}}
            \end{align}

        where:

        - :math:`x_{i,t}` is realized output of industry *i*
        - :math:`x^{\text{cap}}_{i,t}` is labour-based production capacity
        - :math:`x^{\text{inp}}_{i,t}` is input-based production capacity
        - :math:`d_{i,t}` is total demand
        - :math:`S_{ji,t}` is the inventory of input *j* held by industry *i*
        - :math:`A_{ji}` is the technical coefficient (input *j* per unit output of *i*)
        - :math:`\mathcal{V}_i` is the set of critical inputs to industry *i*
        - :math:`\mathcal{U}_i` is the set of important (but not critical) inputs to industry *i*
        - :math:`x^{\text{cap}}_{i,0}` is the initial production capacity
        """
        S = self.prior["Inventories"]
        A = params["TechnicalCoefficients"]
        A_ess = params["CriticalInputMatrix"]

        xinp = self.production(S, A, A_ess, self._xcap0)
        self.state["InputCapacity"] = xinp

        xcap = self.state["ProductiveCapacity"]
        d = self.state["AggregateDemand"]

        self.state["GrossOutput"] = torch.min(torch.min(xcap, xinp), d)

    def rationing(self, t, scenario, params):
        r"""Proportional rationing of output across buyers.

        When output falls short of demand, industry *i* rations its output
        proportionally across all customers.  Each buyer receives a share
        of their order equal to the ratio of output to demand.  An optional
        ``firm_priority`` mode gives precedence to intermediate demand over
        final consumption.

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

        where:

        - :math:`Z_{ji,t}` is the realized intermediate delivery from *j* to *i*
        - :math:`O_{ji,t}` is the intermediate order placed by *i* to *j*
        - :math:`c_{i,t}` is realized household consumption of good *i*
        - :math:`c^d_{i,t}` is household consumption demand for good *i*
        - :math:`f_{i,t}` is realized other final demand for good *i*
        - :math:`f^d_{i,t}` is exogenous other final demand
        - :math:`x_{i,t}` is output of industry *i*
        - :math:`d_{i,t}` is total demand for the output of industry *i*
        """
        x = self.state["GrossOutput"]
        d = self.state["AggregateDemand"]
        orders = self.state["IntermediateOrders"]
        cd = self.state["ConsumptionDemand"]

        safe_d = torch.where(d != 0, d, torch.ones_like(d))
        s = x / safe_d

        if not self.hyper["firm_priority"]:
            self.state["IntermediateConsumption"] = orders * s.unsqueeze(1)
            self.state["RealizedConsumption"] = cd * s
        else:
            safe_rowsum = torch.where(
                orders.sum(dim=1) != 0, orders.sum(dim=1), torch.ones_like(x)
            )
            s_firm = torch.min(torch.ones_like(x), x / safe_rowsum)
            self.state["IntermediateConsumption"] = orders * s_firm.unsqueeze(1)

            Z_total = self.state["IntermediateConsumption"].sum(dim=1)
            remaining = d - orders.sum(dim=1)
            safe_remaining = torch.where(
                remaining != 0, remaining, torch.ones_like(remaining)
            )
            s_final = torch.clamp((x - Z_total) / safe_remaining, min=0.0, max=1.0)
            self.state["RealizedConsumption"] = cd * s_final

    def inventory_update(self, t, scenario, params):
        r"""Inventory accumulation from deliveries minus usage.

        After production and rationing, each industry updates its inventory
        of every input.  Inventories increase by deliveries received and
        decrease by inputs consumed in production (at rates given by the
        technical coefficients).  A floor of zero prevents negative stocks
        for non-critical inputs that may be fully depleted.

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

        where:

        - :math:`S_{ji,t+1}` is the inventory of input *j* held by industry *i* at the start of the next period
        - :math:`S_{ji,t}` is the current inventory
        - :math:`Z_{ji,t}` is the intermediate delivery of *j* received by *i*
        - :math:`A_{ji}` is the technical coefficient
        - :math:`x_{i,t}` is realized output of industry *i*
        """
        A = params["TechnicalCoefficients"]
        x = self.state["GrossOutput"]
        S_prior = self.prior["Inventories"]
        Z = self.state["IntermediateConsumption"]
        used_inputs = A * x.unsqueeze(0)

        self.state["Inventories"] = torch.clamp(S_prior + Z - used_inputs, min=0.0)

    def accounting(self, t, scenario, params):
        r"""Firm profits and household savings.

        Industry profits are total output minus intermediate purchases,
        labour compensation, and other expenses (taxes, imports).
        Household savings are computed as total income minus total realized
        consumption (including non-modeled import and tax expenditures).

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

        where:

        - :math:`\pi_{i,t}` is the profit of industry *i*
        - :math:`x_{i,t}` is total output
        - :math:`Z_{ji,t}` is intermediate purchases of input *j* by industry *i*
        - :math:`l_{i,t}` is labour compensation paid by industry *i*
        - :math:`e_{i,t}` is other expenses (taxes, imports, etc.)
        """
        x = self.state["GrossOutput"]
        Z = self.state["IntermediateConsumption"]
        labour = self.state["LabourCompensation"]
        cost_coef = params["OtherCostCoefficients"]

        self.state["Profits"] = x - Z.sum(dim=0) - labour - cost_coef * x

        c_other_coef = params["HouseholdOtherCostCoefficient"].squeeze()
        c = self.state["RealizedConsumption"]
        pi = self.state["Profits"]
        extra_expenditure = c_other_coef / (1.0 - c_other_coef) * c.sum()

        self.state["Savings"] = (
            pi.sum() + labour.sum() - c.sum() - extra_expenditure
        ).unsqueeze(0)

    # ------------------------------------------------------------------
    # Production functions
    # ------------------------------------------------------------------

    def production_leontief(self, S, A, A_ess, xcap0):
        """Standard Leontief: all inputs with positive ``A`` are binding.

        Parameters
        ----------
        S : torch.Tensor
            ``(N, N)`` inventory matrix.
        A : torch.Tensor
            ``(N, N)`` technical coefficients matrix.
        A_ess : torch.Tensor
            ``(N, N)`` critical-input matrix (unused in this variant).
        xcap0 : torch.Tensor
            ``(N,)`` baseline productive capacity.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        N = S.shape[1]
        xinp = torch.full((N,), float("inf"))
        for k in range(N):
            mask = A[:, k] > 0
            if mask.any():
                safe_a = torch.where(mask, A[:, k], torch.ones_like(A[:, k]))
                ratios = S[:, k] / safe_a
                ratios = torch.where(mask, ratios, torch.tensor(float("inf")))
                xinp[k] = ratios.min()
        return xinp

    def production_strongly_critical(self, S, A, A_ess, xcap0):
        """Critical and important inputs are binding (score >= 0.5).

        Parameters
        ----------
        S : torch.Tensor
            ``(N, N)`` inventory matrix.
        A : torch.Tensor
            ``(N, N)`` technical coefficients matrix.
        A_ess : torch.Tensor
            ``(N, N)`` critical-input matrix.
        xcap0 : torch.Tensor
            ``(N,)`` baseline productive capacity.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        N = S.shape[1]
        xinp = torch.full((N,), float("inf"))
        for k in range(N):
            mask = A_ess[:, k] >= 0.5
            if mask.any():
                safe_a = torch.where(mask, A[:, k], torch.ones_like(A[:, k]))
                ratios = S[:, k] / safe_a
                ratios = torch.where(mask, ratios, torch.tensor(float("inf")))
                xinp[k] = ratios.min()
        return xinp

    def production_half_critical(self, S, A, A_ess, xcap0):
        """Critical inputs fully binding; important inputs half-binding.

        Parameters
        ----------
        S : torch.Tensor
            ``(N, N)`` inventory matrix.
        A : torch.Tensor
            ``(N, N)`` technical coefficients matrix.
        A_ess : torch.Tensor
            ``(N, N)`` critical-input matrix.
        xcap0 : torch.Tensor
            ``(N,)`` baseline productive capacity.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        N = S.shape[1]
        xinp = torch.full((N,), float("inf"))
        for k in range(N):
            critical = A_ess[:, k] > 0.5
            important = A_ess[:, k] == 0.5

            vals = []
            if critical.any():
                safe_a = torch.where(critical, A[:, k], torch.ones_like(A[:, k]))
                ratios = S[:, k] / safe_a
                ratios = torch.where(critical, ratios, torch.tensor(float("inf")))
                vals.append(ratios.min())

            if important.any():
                safe_a = torch.where(important, A[:, k], torch.ones_like(A[:, k]))
                ratios = (S[:, k] / safe_a) * 0.5 + xcap0[k] / 2.0
                ratios = torch.where(important, ratios, torch.tensor(float("inf")))
                vals.append(ratios.min())

            if vals:
                xinp[k] = torch.stack(vals).min()
        return xinp

    def production_weakly_critical(self, S, A, A_ess, xcap0):
        """Only critical inputs (score > 0.5) are binding.

        Parameters
        ----------
        S : torch.Tensor
            ``(N, N)`` inventory matrix.
        A : torch.Tensor
            ``(N, N)`` technical coefficients matrix.
        A_ess : torch.Tensor
            ``(N, N)`` critical-input matrix.
        xcap0 : torch.Tensor
            ``(N,)`` baseline productive capacity.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        N = S.shape[1]
        xinp = torch.full((N,), float("inf"))
        for k in range(N):
            mask = A_ess[:, k] > 0.5
            if mask.any():
                safe_a = torch.where(mask, A[:, k], torch.ones_like(A[:, k]))
                ratios = S[:, k] / safe_a
                ratios = torch.where(mask, ratios, torch.tensor(float("inf")))
                xinp[k] = ratios.min()
        return xinp

    def production_linear(self, S, A, A_ess, xcap0):
        """Linear production: all inputs are perfect substitutes.

        Parameters
        ----------
        S : torch.Tensor
            ``(N, N)`` inventory matrix.
        A : torch.Tensor
            ``(N, N)`` technical coefficients matrix.
        A_ess : torch.Tensor
            ``(N, N)`` critical-input matrix (unused in this variant).
        xcap0 : torch.Tensor
            ``(N,)`` baseline productive capacity.

        Returns
        -------
        torch.Tensor
            ``(N,)`` input-constrained production capacity.
        """
        col_sum_A = A.sum(dim=0)
        col_sum_S = S.sum(dim=0)
        safe_denom = torch.where(col_sum_A != 0, col_sum_A, torch.ones_like(col_sum_A))
        xinp = col_sum_S / safe_denom
        xinp = torch.where(col_sum_A != 0, xinp, torch.tensor(float("inf")))
        return xinp
