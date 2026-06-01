# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
r"""Behavior class for the Mark-0 COVID model.

Reference: Bouchaud, J.-P., Gualdi, S., Tarzia, M., Zamponi, F. (2018);
Gualdi, S., Tarzia, M., Zamponi, F., Bouchaud, J.-P. (2015); Sharma, D.,
Bouchaud, J.-P., Gualdi, S., Tarzia, M., Zamponi, F. (2021).

The model is a closed-economy heterogeneous-agent ABM. Each macro period
runs a 24-phase loop over :math:`N` firms plus one representative
household, one bank, one central bank. The forward pass is differentiable
end-to-end because:

1. All three :math:`U(0,1)` random draws (price-adjustment magnitude, wage-
   adjustment magnitude, firm-revival draw) are pre-drawn in
   :meth:`initialize` as ``(timesteps, N_firms)`` buffers and indexed by
   timestep in the forward loop. The buffers are drawn from a per-instance
   :class:`torch.Generator` seeded from ``hyper["seed"]`` — no global RNG.
2. Branch logic mirrors abmstat: most sites use ``torch.where`` (value
   branches differentiable, condition not). The two gradient-critical sites
   — :meth:`find_surviving_firms` (stay-alive indicator) and
   :meth:`recompute_firm_totals_stayalive` (positive-Y guard) — use
   :meth:`Behavior.diffwhere`, a sigmoid-blended where that is
   differentiable through the condition.
3. Divisor guards use ``torch.where(x != 0, x, 1.0)`` to match abmstat;
   the inflation/aggregation phases also add a small ``epsilon`` where
   abmstat's branching is unnecessary.

The 24 phases are dispatched from :meth:`step` in a single ordered call. Three
of them are post-block aggregation passes:
:meth:`recompute_firm_totals_stayalive`, :meth:`recompute_firm_totals_revival`,
:meth:`recompute_firm_totals_consumption`. Their position in the loop is
load-bearing — the macro accounting identities only close if they fire after
the corresponding firm-mask mutation.

The ``supports_differentiable`` class attribute is ``True``; the default
instance flag ``differentiable=True``. Downstream consumers
(SloppyModels_KLDivergence pipeline) need gradients w.r.t. macro parameters
through the agent loop. The non-differentiable mode (``differentiable=False``)
preserves the contract used by aggregate-only consumers — values match within
tanh-saturation tolerance.
"""

import logging

import torch

from macrostat.core.behavior import Behavior
from macrostat.models.Mark0COVID.parameters import ParametersMark0COVID
from macrostat.models.Mark0COVID.scenarios import ScenariosMark0COVID
from macrostat.models.Mark0COVID.variables import VariablesMark0COVID

logger = logging.getLogger(__name__)


class BehaviorMark0COVID(Behavior):
    r"""Simulation logic for the Mark-0 COVID model."""

    version = "Mark0COVID"
    supports_differentiable = True

    def __init__(
        self,
        parameters: ParametersMark0COVID | None = None,
        scenarios: ScenariosMark0COVID | None = None,
        variables: VariablesMark0COVID | None = None,
        scenario: int = 0,
        differentiable: bool = True,
        debug: bool = False,
    ):
        if parameters is None:
            parameters = ParametersMark0COVID()
        if scenarios is None:
            scenarios = ScenariosMark0COVID(parameters=parameters)
        if variables is None:
            variables = VariablesMark0COVID(parameters=parameters)

        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            scenario=scenario,
            differentiable=differentiable,
            debug=debug,
        )

        self._torch_rng: torch.Generator | None = None
        self._noise_price: torch.Tensor | None = None
        self._noise_wage: torch.Tensor | None = None
        self._noise_revive: torch.Tensor | None = None
        self._noise_revive_y: torch.Tensor | None = None

    @property
    def _dtype(self) -> torch.dtype:
        return self.hyper.get("dtype", torch.float64)

    # ------------------------------------------------------------------
    # Initialization
    # ------------------------------------------------------------------

    def initialize(self):
        """Allocate the per-instance RNG, draw three noise buffers, set
        initial firm and macro state.

        The RNG is created fresh and re-seeded from ``hyper["seed"]`` on
        every ``forward()`` call so two consecutive simulations on the same
        instance produce identical streams.
        """
        self.initialize_rng()
        self.initialize_noise_buffers()
        self.initialize_firms()
        self.initialize_macro()

    def initialize_rng(self):
        """Allocate a per-instance ``torch.Generator`` seeded from
        ``hyper["seed"]``. Never touches the global torch RNG.

        Dependency
        ----------
        - hyper: seed
        """
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(int(self.hyper["seed"]))

    def initialize_noise_buffers(self):
        r"""Pre-draw four independent ``(T, N_firms)`` U(0,1) buffers, in the
        exact interleaved order the reference implementation uses live:
        ``price[t], wage[t], revive[t], revive_y[t]`` per period.

        Notes
        -----
        Uses the global torch RNG (``torch.manual_seed``) so the underlying
        stream is bit-identical to the reference at the same seed. Saves and
        restores the global state around the draws so no outside-state is
        polluted.

        Dependency
        ----------
        - hyper: timesteps
        - hyper: N_firms
        - hyper: seed
        """
        t = int(self.hyper["timesteps"])
        n = int(self.hyper["N_firms"])
        seed = int(self.hyper["seed"])
        dtype = self._dtype

        saved = torch.random.get_rng_state()
        try:
            torch.manual_seed(seed)
            self._noise_price = torch.empty(t, n, dtype=dtype)
            self._noise_wage = torch.empty(t, n, dtype=dtype)
            self._noise_revive = torch.empty(t, n, dtype=dtype)
            self._noise_revive_y = torch.empty(t, n, dtype=dtype)
            for i in range(t):
                self._noise_price[i] = torch.rand(n, dtype=dtype)
                self._noise_wage[i] = torch.rand(n, dtype=dtype)
                self._noise_revive[i] = torch.rand(n, dtype=dtype)
                self._noise_revive_y[i] = torch.rand(n, dtype=dtype)
        finally:
            torch.random.set_rng_state(saved)

    def initialize_firms(self):
        r"""Initialise firm-level state vectors and macro carry-over scalars.

        Firms are indexed :math:`i \in \{0, \ldots, N-1\}`. Initial price and
        production carry a small linear spread across the index, scaled by
        :math:`0.01`. Assets are seeded so the cross-section average equals
        :math:`Y_0 W_0`. The aggregates :math:`\bar P`, :math:`\bar W`,
        :math:`Y_{tot}`, :math:`A_{tot}` are recorded as production-weighted
        moments.

        Equations
        ---------
        .. math::
            \begin{align}
                P_{i,0} &= 1 + 0.01\,(2 i/N - 1), \\
                Y_{i,0} &= Y_0 + 0.01\,(2 i/N - 1), \\
                W_{i,0} &= 1, \quad D_{i,0} = Y_0, \\
                A_{i,0} &= 2\,Y_{i,0}\,W_{i,0}\,(i/N), \\
                \Pi_{i,0} &= P_{i,0}\,\min(D_{i,0}, Y_{i,0}) - W_{i,0}\,Y_{i,0}, \\
                \alpha_{i,0} &= 1.
            \end{align}

        Each firm receives a distinct initial price, production, and asset
        endowment, producing a non-degenerate cross-section before the first
        macro step.

        Dependency
        ----------
        - params: InitialProductionScale
        - hyper: N_firms

        Sets
        ----
        - FirmPrice
        - FirmWage
        - FirmProduction
        - FirmDemand
        - FirmAssets
        - FirmProfits
        - FirmAlive
        - AveragePrice
        - AverageWage
        - MaxWage
        - TotalProduction
        - FirmAssetsTotal
        - FirmStayAlive
        - FirmEnterBankruptcy
        - FirmPayroll
        - FirmExcessDemandQuantity
        - FirmExcessDemandMask
        - FirmExcessSupplyMask
        - FirmRenSolvency
        - FirmUnemployedLabourShare
        """
        n = int(self.hyper["N_firms"])
        kwg = {"dtype": self._dtype}
        y0 = float(self.params["InitialProductionScale"].item())

        ratio = torch.arange(n, **kwg) / n
        spread = 2.0 * ratio - 1.0

        price = torch.ones(n, **kwg) + 0.01 * spread
        production = y0 + 0.01 * spread
        wage = torch.ones(n, **kwg)
        demand = torch.full((n,), y0, **kwg)
        profits = price * torch.min(demand, production) - wage * production
        assets = 2.0 * production * wage * ratio
        alive = torch.ones(n, **kwg)

        self.state["FirmPrice"] = price
        self.state["FirmWage"] = wage
        self.state["FirmProduction"] = production
        self.state["FirmDemand"] = demand
        self.state["FirmAssets"] = assets
        self.state["FirmProfits"] = profits
        self.state["FirmAlive"] = alive

        ytot = production.sum()
        atot = assets.sum()
        pavg = (price * production).sum() / ytot
        wavg = (wage * production).sum() / ytot
        self.state["AveragePrice"] = pavg.unsqueeze(0)
        self.state["AverageWage"] = wavg.unsqueeze(0)
        self.state["MaxWage"] = wage.max().unsqueeze(0)
        self.state["TotalProduction"] = ytot.unsqueeze(0)
        self.state["FirmAssetsTotal"] = atot.unsqueeze(0)

        zero_firm = torch.zeros(n, **kwg)
        self.state["FirmStayAlive"] = alive
        self.state["FirmEnterBankruptcy"] = zero_firm.clone()
        self.state["FirmPayroll"] = wage * production
        self.state["FirmExcessDemandQuantity"] = zero_firm.clone()
        self.state["FirmExcessDemandMask"] = zero_firm.clone()
        self.state["FirmExcessSupplyMask"] = zero_firm.clone()
        self.state["FirmRenSolvency"] = zero_firm.clone()
        self.state["FirmUnemployedLabourShare"] = zero_firm.clone()

    def initialize_macro(self):
        r"""Initialise macro state: savings, interest rates, employment,
        EWMA registers.

        Total money stock is fixed to :math:`N` by rescaling household
        savings :math:`S` and firm assets :math:`A_i` against the prior
        firm endowment. The CB rate, loan rate, and EWMA loan rate are all
        initialised to the baseline :math:`\rho^\star`; deposit-side rates,
        inflation registers, and the bankruptcy / propensity / gamma /
        default accumulators all initialise to zero.

        Equations
        ---------
        .. math::
            \begin{align}
                S_0 &= N\,\frac{Y_{tot}}{A_{tot} + Y_{tot}}, \\
                A_{i,0} &\leftarrow A_{i,0}\,\frac{N}{A_{tot} + Y_{tot}}, \\
                M^0_0 &= N, \\
                \rho^{CB}_0 = \rho^l_0 = \bar\rho^l_0 &= \rho^\star, \\
                e_0 &= Y_{tot}/N, \quad u_0 = 1 - e_0.
            \end{align}

        Rescaling pins :math:`A_{tot} + S = N` exactly at :math:`t=0`, so the
        money-stock identity holds before any phase fires.

        Dependency
        ----------
        - params: InterestRateBaseline
        - hyper: N_firms
        - state: FirmProduction
        - state: FirmWage
        - state: FirmAssets

        Sets
        ----
        - FirmAssets
        - FirmAssetsTotal
        - HouseholdSavings
        - M0Stock
        - CBRate
        - LoanRate
        - LoanRateEWMA
        - DepositRate
        - DepositRateEWMA
        - UnemploymentEWMA
        - Inflation
        - ExpectedInflationEWMA
        - ExpectedInflationUsed
        - BankruptcyRate
        - ConsumptionPropensity
        - Employment
        - Unemployment
        - FirmSavingsTotal
        - FirmDebtTotal
        - TotalPayroll
        - TotalDemand
        - FirmGamma
        - ConsumptionBudget
        - DefaultedTotal
        - LowestPrice
        """
        n = int(self.hyper["N_firms"])
        kwg = {"dtype": self._dtype}
        rho_star = self.params["InterestRateBaseline"]

        production = self.state["FirmProduction"]
        wage = self.state["FirmWage"]
        assets = self.state["FirmAssets"]
        ytot = production.sum()
        atot = assets.sum()

        savings = ytot * (n / (atot + ytot))
        assets = assets * (n / (atot + ytot))
        atot = assets.sum()

        self.state["FirmAssets"] = assets
        self.state["FirmAssetsTotal"] = atot.unsqueeze(0)
        self.state["HouseholdSavings"] = savings.unsqueeze(0)
        self.state["M0Stock"] = torch.tensor([float(n)], **kwg)

        rho_scalar = torch.full((1,), float(rho_star.item()), **kwg)
        zero_scalar = torch.zeros(1, **kwg)

        self.state["CBRate"] = rho_scalar.clone()
        self.state["LoanRate"] = rho_scalar.clone()
        self.state["LoanRateEWMA"] = rho_scalar.clone()
        self.state["DepositRate"] = zero_scalar.clone()
        self.state["DepositRateEWMA"] = zero_scalar.clone()
        self.state["UnemploymentEWMA"] = zero_scalar.clone()
        self.state["Inflation"] = zero_scalar.clone()
        self.state["ExpectedInflationEWMA"] = zero_scalar.clone()
        self.state["ExpectedInflationUsed"] = zero_scalar.clone()
        self.state["BankruptcyRate"] = zero_scalar.clone()
        self.state["ConsumptionPropensity"] = zero_scalar.clone()

        employment = ytot / n
        unemployment = 1.0 - employment
        self.state["Employment"] = employment.unsqueeze(0)
        self.state["Unemployment"] = unemployment.unsqueeze(0)

        firm_savings = torch.maximum(assets, torch.zeros_like(assets)).sum()
        firm_debt = (-torch.minimum(assets, torch.zeros_like(assets))).sum()
        payroll_total = (wage * production).sum()
        self.state["FirmSavingsTotal"] = firm_savings.unsqueeze(0)
        self.state["FirmDebtTotal"] = firm_debt.unsqueeze(0)
        self.state["TotalPayroll"] = payroll_total.unsqueeze(0)
        self.state["TotalDemand"] = ytot.unsqueeze(0)
        self.state["FirmGamma"] = zero_scalar.clone()
        self.state["ConsumptionBudget"] = zero_scalar.clone()
        self.state["DefaultedTotal"] = zero_scalar.clone()
        self.state["LowestPrice"] = torch.ones(1, **kwg)

    # ------------------------------------------------------------------
    # Per-period step dispatcher
    # ------------------------------------------------------------------

    def step(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        r"""Execute one macro period: 24 ordered phase methods.

        The dispatcher seeds ``self.state`` from ``self.prior`` (the recorded
        end-of-(t-1) state) so each phase reads the most recent value through
        ``self.state[k]`` regardless of whether the value was set this period
        or carried over. Phases mutate ``self.state`` in place.

        Three of the calls are ordering-critical post-block aggregation passes
        — see :meth:`recompute_firm_totals_stayalive`,
        :meth:`recompute_firm_totals_revival`,
        :meth:`recompute_firm_totals_consumption`. Their position is
        load-bearing: the macro identities only close if they fire after the
        corresponding firm-mask mutation.

        Parameters
        ----------
        t : int
            Current macro period index.
        scenario : dict
            Time-indexed scenario values.
        params : dict
            Parameters with scenario shocks applied.

        Sets
        ----
        - TotalPayroll
        """
        for k, v in self.prior.items():
            self.state[k] = v.clone()

        self.renormalize_prices(t, scenario, params)
        self.update_averages(t, scenario, params)
        self.find_surviving_firms(t, scenario, params)
        self.demand_production_imbalance(t, scenario, params)
        self.compute_gamma_and_ren(t, scenario, params)
        self.compute_ushare(t, scenario, params)
        self.production_adjustment(t, scenario, params)
        self.price_adjustment(t, scenario, params)
        self.wage_adjustment(t, scenario, params)
        self.expectation_adjustments(t, scenario, params)
        self.recompute_firm_totals_stayalive(t, scenario, params)
        self.compute_lowest_price(t, scenario, params)
        self.bankrupt_firms(t, scenario, params)
        self.compute_moments(t, scenario, params)
        self.compute_employment(t, scenario, params)
        self.solve_rounding_errors(t, scenario, params)
        self.set_interest_rates(t, scenario, params)
        self.household_consumption(t, scenario, params)
        self.firm_accounting(t, scenario, params)
        self.pay_dividends(t, scenario, params)
        self.recompute_firm_totals_consumption(t, scenario, params)
        self.revive_firms(t, scenario, params)
        self.recompute_firm_totals_revival(t, scenario, params)
        self.compute_inflation_and_employment(t, scenario, params)
        self.monetary_policy(t, scenario, params)
        self.state["TotalPayroll"] = self.state["FirmPayroll"].sum().unsqueeze(0)

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def renormalize_prices(self, t, scenario, params):
        r"""Rescale all nominal quantities by the lagged average price so
        that :math:`\bar P_t \equiv 1` going into the wage/price update.

        Every nominal series — firm prices, wages, assets, profits,
        household savings, the wage moments, and the money stock — is
        divided by :math:`\bar P_{t-1}` in lock-step. The carry-over
        :math:`\bar P` then collapses to 1.

        Equations
        ---------
        .. math::
            \begin{align}
                P_{i,t} &\leftarrow P_{i,t} / \bar P_{t-1}, \\
                W_{i,t} &\leftarrow W_{i,t} / \bar P_{t-1}, \\
                A_{i,t} &\leftarrow A_{i,t} / \bar P_{t-1}, \\
                \Pi_{i,t} &\leftarrow \Pi_{i,t} / \bar P_{t-1}, \\
                S_t &\leftarrow S_t / \bar P_{t-1}, \\
                \bar W_t, M_t^{0}, W^{\max}_t &\leftarrow (\cdot) / \bar P_{t-1}, \\
                \bar P_t &\leftarrow 1.
            \end{align}

        Lock-step rescaling preserves every nominal-quantity ratio while
        zeroing out trend inflation in the carry-over level, so subsequent
        wage and price update phases work in unit price units.

        Dependency
        ----------
        - state: AveragePrice
        - state: FirmPrice
        - state: FirmWage
        - state: FirmAssets
        - state: FirmProfits
        - state: HouseholdSavings
        - state: AverageWage
        - state: MaxWage
        - state: M0Stock

        Sets
        ----
        - FirmPrice
        - FirmWage
        - FirmAssets
        - FirmProfits
        - HouseholdSavings
        - AverageWage
        - MaxWage
        - M0Stock
        - AveragePrice
        """
        pavg = self.state["AveragePrice"]
        self.state["FirmPrice"] = self.state["FirmPrice"] / pavg
        self.state["FirmWage"] = self.state["FirmWage"] / pavg
        self.state["FirmAssets"] = self.state["FirmAssets"] / pavg
        self.state["FirmProfits"] = self.state["FirmProfits"] / pavg
        self.state["HouseholdSavings"] = self.state["HouseholdSavings"] / pavg
        self.state["AverageWage"] = self.state["AverageWage"] / pavg
        self.state["MaxWage"] = self.state["MaxWage"] / pavg
        self.state["M0Stock"] = self.state["M0Stock"] / pavg
        self.state["AveragePrice"] = torch.ones_like(pavg)

    def update_averages(self, t, scenario, params):
        r"""EWMA update of inflation, interest rate, and unemployment
        registers; convex combination of EWMA inflation and CB target for
        the expectation used downstream.

        The four EWMA registers share the same memory weight
        :math:`\omega`. The expected-inflation pipe used by every
        downstream phase mixes the CB target with the EWMA term in a
        fixed convex combination.

        Equations
        ---------
        .. math::
            \begin{align}
                \pi^{ema}_t &= \omega\,\pi_{t-1} + (1-\omega)\,\pi^{ema}_{t-1}, \\
                \bar\rho^d_t &= \omega\,\rho^d_{t-1} + (1-\omega)\,\bar\rho^d_{t-1}, \\
                \bar\rho^l_t &= \omega\,\rho^l_{t-1} + (1-\omega)\,\bar\rho^l_{t-1}, \\
                \bar u_t &= \omega\,u_{t-1} + (1-\omega)\,\bar u_{t-1}, \\
                \hat\pi_t &= \tau^T \pi^\star + \tau^R \pi^{ema}_t.
            \end{align}

        Smoothing the four registers in lock-step makes downstream
        bank-side and household-side decisions react to slow-moving
        aggregate signals rather than period noise.

        Dependency
        ----------
        - params: EWMAMemory
        - params: ExpectedInflationEWMAWeight
        - params: ExpectedInflationTargetWeight
        - params: CBInflationTarget
        - state: Inflation
        - state: ExpectedInflationEWMA
        - state: DepositRate
        - state: DepositRateEWMA
        - state: LoanRate
        - state: LoanRateEWMA
        - state: Unemployment
        - state: UnemploymentEWMA

        Sets
        ----
        - ExpectedInflationEWMA
        - DepositRateEWMA
        - LoanRateEWMA
        - UnemploymentEWMA
        - ExpectedInflationUsed
        """
        omega = params["EWMAMemory"]
        tau_r = params["ExpectedInflationEWMAWeight"]
        tau_t = params["ExpectedInflationTargetWeight"]
        pi_star = params["CBInflationTarget"]

        comp = 1.0 - omega
        pi_ema = (
            omega * self.state["Inflation"] + comp * self.state["ExpectedInflationEWMA"]
        )
        rp_avg = (
            omega * self.state["DepositRate"] + comp * self.state["DepositRateEWMA"]
        )
        rm_avg = omega * self.state["LoanRate"] + comp * self.state["LoanRateEWMA"]
        u_avg = (
            omega * self.state["Unemployment"] + comp * self.state["UnemploymentEWMA"]
        )
        pi_used = tau_t * pi_star + tau_r * pi_ema

        self.state["ExpectedInflationEWMA"] = pi_ema
        self.state["DepositRateEWMA"] = rp_avg
        self.state["LoanRateEWMA"] = rm_avg
        self.state["UnemploymentEWMA"] = u_avg
        self.state["ExpectedInflationUsed"] = pi_used

    def find_surviving_firms(self, t, scenario, params):
        r"""Identify firms whose assets plus a fraction :math:`\Theta` of
        payroll are positive; survivors stay alive next period, the rest
        enter bankruptcy.

        The differentiable indicator uses :meth:`Behavior.diffwhere` so
        gradient flows through the solvency threshold. The bankruptcy
        indicator is the squared complement, giving a smoother gradient
        at the boundary.

        Equations
        ---------
        .. math::
            \begin{align}
                \text{stay}_{i,t} &= \alpha_{i,t-1} \cdot \text{diffwhere}\big(A_{i,t} + \Theta\, W_{i,t} Y_{i,t},\, 1,\, 0\big), \\
                \text{enter}_{i,t} &= \alpha_{i,t-1}\,(1 - \text{stay}_{i,t})^2.
            \end{align}

        Firms whose end-of-period assets cover a fraction :math:`\Theta`
        of the next period's wage bill survive; the rest are flagged for
        bankruptcy.

        Dependency
        ----------
        - params: DefaultThreshold
        - state: FirmAlive
        - state: FirmWage
        - state: FirmProduction
        - state: FirmAssets

        Sets
        ----
        - FirmStayAlive
        - FirmEnterBankruptcy
        - FirmPayroll
        """
        theta = params["DefaultThreshold"]
        payroll = self.state["FirmWage"] * self.state["FirmProduction"]
        payroll_affordability = self.state["FirmAssets"] + theta * payroll

        ones = torch.ones_like(payroll_affordability)
        zeros = torch.zeros_like(payroll_affordability)
        stay_alive = self.diffwhere(payroll_affordability, ones, zeros)
        enter_bankruptcy = (1.0 - stay_alive).pow(2)

        alive = self.state["FirmAlive"]
        self.state["FirmStayAlive"] = alive * stay_alive
        self.state["FirmEnterBankruptcy"] = alive * enter_bankruptcy
        self.state["FirmPayroll"] = payroll

    def demand_production_imbalance(self, t, scenario, params):
        r"""Compute the per-firm demand-production gap and the
        corresponding excess-demand and excess-supply indicator masks.

        Notes
        -----
        ``torch.where`` here is differentiable through the value branches
        (constants) but not through the condition. Acceptable because the
        condition does not depend on any parameter.

        Equations
        ---------
        .. math::
            \begin{align}
                \Delta Y_{i,t} &= \text{stay}_{i,t}\,(D_{i,t} - Y_{i,t}), \\
                \mathbf{1}^D_{i,t} &= \mathbb{1}\{\Delta Y_{i,t} > 0\}, \\
                \mathbf{1}^S_{i,t} &= \alpha_{i,t}\,\mathbb{1}\{D_{i,t} - Y_{i,t} \le 0\}.
            \end{align}

        The masks partition the surviving firm cross-section into
        demand-constrained and supply-constrained sub-populations consumed
        by the subsequent production, price, and wage updates.

        Dependency
        ----------
        - state: FirmDemand
        - state: FirmProduction
        - state: FirmAlive
        - state: FirmStayAlive

        Sets
        ----
        - FirmExcessDemandQuantity
        - FirmExcessDemandMask
        - FirmExcessSupplyMask
        """
        one = torch.ones_like(self.state["FirmProduction"])
        zero = torch.zeros_like(self.state["FirmProduction"])

        dY = self.state["FirmStayAlive"] * (
            self.state["FirmDemand"] - self.state["FirmProduction"]
        )
        excess_demand = torch.where(dY > 0, one, zero)
        excess_supply = self.state["FirmAlive"] * torch.where(
            self.state["FirmDemand"] - self.state["FirmProduction"] <= 0, one, zero
        )

        self.state["FirmExcessDemandQuantity"] = dY
        self.state["FirmExcessDemandMask"] = excess_demand
        self.state["FirmExcessSupplyMask"] = excess_supply

    def compute_gamma_and_ren(self, t, scenario, params):
        r"""Bank-side :math:`\Gamma` (real-rate gap above baseline) and
        per-firm :math:`\text{ren}` ratio (:math:`\Gamma` times solvency).

        Equations
        ---------
        .. math::
            \begin{align}
                \Gamma_t &= \Gamma_0 + \text{ReLU}\big(\alpha_\Gamma\,(\bar\rho^l_t - \hat\pi_t) - \Gamma_0\big), \\
                \text{ren}_{i,t} &= \Gamma_t \cdot \frac{A_{i,t}}{W_{i,t} Y_{i,t} + \epsilon}.
            \end{align}

        :math:`\Gamma_t` rises only when the real loan rate exceeds the
        baseline; :math:`\text{ren}_{i,t}` scales each firm's gross
        solvency by that bank-side aggressiveness, driving the hiring,
        wage, and price asymmetries downstream.

        Dependency
        ----------
        - params: LoanRateGammaSensitivity
        - params: GammaBaseline
        - hyper: epsilon
        - state: LoanRateEWMA
        - state: ExpectedInflationUsed
        - state: FirmAssets
        - state: FirmPayroll

        Sets
        ----
        - FirmGamma
        - FirmRenSolvency
        """
        alpha_g = params["LoanRateGammaSensitivity"]
        gamma_0 = params["GammaBaseline"]
        eps = self.hyper["epsilon"]

        gap = (
            alpha_g * (self.state["LoanRateEWMA"] - self.state["ExpectedInflationUsed"])
            - gamma_0
        )
        gamma = gamma_0 + torch.relu(gap)
        ren = gamma * (self.state["FirmAssets"] / (self.state["FirmPayroll"] + eps))

        self.state["FirmGamma"] = gamma
        self.state["FirmRenSolvency"] = ren

    def compute_ushare(self, t, scenario, params):
        r"""Per-firm share of the unemployed labour pool via a logit
        weight on the wage gap to the maximum wage.

        Equations
        ---------
        .. math::
            \begin{align}
                z_{i,t} &= \beta\,(W_{i,t} - W^{\max}_t) / \bar W_t, \\
                w_{i,t} &= \alpha_{i,t}\,e^{z_{i,t}}, \\
                Z_t &= \sum_i w_{i,t}, \\
                u^{\text{share}}_{i,t} &= \alpha_{i,t}\,\frac{u_t\,N\,(1 - \text{bust}_t)\,e^{z_{i,t}}}{Z_t + \epsilon}.
            \end{align}

        Higher-wage live firms attract a larger share of the unemployed
        pool, allowing them to expand production faster when demand
        pressure builds.

        Dependency
        ----------
        - params: HouseholdIntensityOfChoice
        - hyper: N_firms
        - hyper: epsilon
        - state: FirmWage
        - state: MaxWage
        - state: AverageWage
        - state: FirmAlive
        - state: Unemployment
        - state: BankruptcyRate

        Sets
        ----
        - FirmUnemployedLabourShare
        """
        beta = params["HouseholdIntensityOfChoice"]
        n = float(self.hyper["N_firms"])

        arg = (
            beta
            * (self.state["FirmWage"] - self.state["MaxWage"])
            / self.state["AverageWage"]
        )
        weights = self.state["FirmAlive"] * torch.exp(arg)
        wage_norm = weights.sum()
        eps = self.hyper["epsilon"]
        u_share = self.state["FirmAlive"] * (
            self.state["Unemployment"]
            * n
            * (1.0 - self.state["BankruptcyRate"])
            * torch.exp(arg)
            / (wage_norm + eps)
        )

        self.state["FirmUnemployedLabourShare"] = u_share

    def production_adjustment(self, t, scenario, params):
        r"""Adjust production: increase when in excess-demand regime,
        decrease when in excess-supply regime.

        Equations
        ---------
        .. math::
            \begin{align}
                \eta^+_{i,t} &= \mathrm{clamp}\big(\eta_0 r\,(1 + \text{ren}_{i,t}),\,0,\,1\big), \\
                \eta^-_{i,t} &= \mathrm{clamp}\big(\eta_0\,(1 - \text{ren}_{i,t}),\,0,\,1\big), \\
                Y_{i,t} &\leftarrow Y_{i,t}
                    + \text{stay}_{i,t}\,\mathbf{1}^D_{i,t}\,\min(\eta^+_{i,t}\,\Delta Y_{i,t},\,u^{\text{share}}_{i,t}) \\
                    &\phantom{\leftarrow Y_{i,t}}\;
                    + \text{stay}_{i,t}\,\mathbf{1}^S_{i,t}\,\eta^-_{i,t}\,\Delta Y_{i,t}.
            \end{align}

        Hiring is capped by the firm's share of the unemployed labour
        pool, and firing is asymmetric to hiring through the
        :math:`(1 + \text{ren})` versus :math:`(1 - \text{ren})`
        modulation.

        Dependency
        ----------
        - params: FiringPropensity
        - params: HiringFiringRate
        - state: FirmProduction
        - state: FirmStayAlive
        - state: FirmExcessDemandMask
        - state: FirmExcessSupplyMask
        - state: FirmExcessDemandQuantity
        - state: FirmRenSolvency
        - state: FirmUnemployedLabourShare

        Sets
        ----
        - FirmProduction
        """
        eta0m = params["FiringPropensity"]
        r_param = params["HiringFiringRate"]

        eta_plus = torch.clamp(
            eta0m * r_param * (1.0 + self.state["FirmRenSolvency"]), 0.0, 1.0
        )
        eta_minus = torch.clamp(eta0m * (1.0 - self.state["FirmRenSolvency"]), 0.0, 1.0)

        production = self.state["FirmProduction"]
        production = production + self.state["FirmStayAlive"] * self.state[
            "FirmExcessDemandMask"
        ] * torch.minimum(
            eta_plus * self.state["FirmExcessDemandQuantity"],
            self.state["FirmUnemployedLabourShare"],
        )
        production = production + self.state["FirmStayAlive"] * self.state[
            "FirmExcessSupplyMask"
        ] * (eta_minus * self.state["FirmExcessDemandQuantity"])
        self.state["FirmProduction"] = production

    def price_adjustment(self, t, scenario, params):
        r"""Price update from the frozen-noise buffer.

        Excess-demand firms with below-average price scale :math:`P` up
        by :math:`(1 + r_p)`; excess-supply firms with above-average
        price scale down by :math:`(1 - r_p)`.

        Notes
        -----
        Branching uses ``torch.where`` to match the reference; gradient
        flow w.r.t. :math:`\gamma_p` is through :math:`r_p`, gradient
        flow through the condition is suppressed.

        Equations
        ---------
        .. math::
            \begin{align}
                r_p &= \gamma_p\,u^p_{t,i}, \\
                P_{i,t} &\leftarrow P_{i,t}\,(1 + r_p)
                    \quad\text{if}\quad \text{stay}_{i,t}\,\mathbf{1}^D_{i,t}\,(P_{i,t} < \bar P_t), \\
                P_{i,t} &\leftarrow P_{i,t}\,(1 - r_p)
                    \quad\text{if}\quad \mathbf{1}^S_{i,t}\,(P_{i,t} > \bar P_t).
            \end{align}

        Below-average price-setters facing excess demand raise prices and
        above-average price-setters facing excess supply cut prices,
        pulling the cross-section toward the average.

        Dependency
        ----------
        - params: PriceAdjustmentSize
        - state: FirmPrice
        - state: AveragePrice
        - state: FirmStayAlive
        - state: FirmExcessDemandMask
        - state: FirmExcessSupplyMask

        Sets
        ----
        - FirmPrice
        """
        gammap = params["PriceAdjustmentSize"]
        rp = gammap * self._noise_price[t]

        price = self.state["FirmPrice"]
        pavg = self.state["AveragePrice"]
        ones = torch.ones_like(price)
        zeros = torch.zeros_like(price)

        priceup = (
            self.state["FirmStayAlive"]
            * self.state["FirmExcessDemandMask"]
            * torch.where(price < pavg, ones, zeros)
        )
        price = torch.where(priceup > 0.0, price * (1.0 + rp), price)

        pricedown = torch.where(
            self.state["FirmExcessSupplyMask"] * price > pavg, ones, zeros
        )
        price = torch.where(pricedown > 0.0, price * (1.0 - rp), price)

        self.state["FirmPrice"] = price

    def wage_adjustment(self, t, scenario, params):
        r"""Smooth wage update from the frozen-noise buffer.

        Excess-demand profitable firms raise wages; excess-supply
        loss-making firms lower wages. The cashflow-per-production
        ceiling caps wage increases at the firm's affordable level so
        the asset balance does not turn nominally infeasible.

        Equations
        ---------
        .. math::
            \begin{align}
                r_w &= \gamma_p\,r\,u^w_{t,i}, \\
                W_{i,t} &\leftarrow W_{i,t}\,[1 + (1 + \text{ren}_{i,t})\,r_w\,e_t]
                    \quad\text{if}\quad \text{stay}_{i,t}\,\mathbf{1}^D_{i,t}\,(\Pi_{i,t} > 0), \\
                W_{i,t} &\leftarrow \min\big(W_{i,t},\,\text{cashflow}_{i,t}/Y_{i,t}\big)
                    \quad\text{(ceiling on the wage rise)}, \\
                W_{i,t} &\leftarrow W_{i,t}\,[1 - (1 - \text{ren}_{i,t})\,r_w\,u_t]
                    \quad\text{if}\quad \text{stay}_{i,t}\,\mathbf{1}^S_{i,t}\,(\Pi_{i,t} < 0).
            \end{align}

        Wage growth is gated on both regime (demand/supply) and firm
        profitability; the cashflow ceiling prevents wage-driven
        insolvency, and the :math:`(1 \pm \text{ren})` weighting
        introduces the same hiring/firing asymmetry seen in production.

        Dependency
        ----------
        - params: PriceAdjustmentSize
        - params: WagePriceAdjustmentRatio
        - state: FirmWage
        - state: FirmPrice
        - state: FirmDemand
        - state: FirmProduction
        - state: FirmAssets
        - state: FirmProfits
        - state: Employment
        - state: Unemployment
        - state: LoanRate
        - state: DepositRate
        - state: FirmStayAlive
        - state: FirmExcessDemandMask
        - state: FirmExcessSupplyMask
        - state: FirmRenSolvency

        Sets
        ----
        - FirmWage
        """
        gammap = params["PriceAdjustmentSize"]
        r_ratio = params["WagePriceAdjustmentRatio"]
        zero_vec = torch.zeros_like(self.state["FirmWage"])
        ones = torch.ones_like(self.state["FirmWage"])

        rw = gammap * r_ratio * self._noise_wage[t]

        mask_wageplus = torch.where(
            self.state["FirmStayAlive"]
            * self.state["FirmExcessDemandMask"]
            * (self.state["FirmProfits"] > 0).to(zero_vec.dtype)
            > 0.0,
            ones,
            zero_vec,
        )

        employment = self.state["Employment"]
        coeff_up = 1.0 + (1.0 + self.state["FirmRenSolvency"]) * rw * employment
        wage = self.state["FirmWage"]
        wage = torch.where(mask_wageplus > 0.0, wage * coeff_up, wage)

        revenue = self.state["FirmPrice"] * torch.minimum(
            self.state["FirmDemand"], self.state["FirmProduction"]
        )
        loan_interest = self.state["LoanRate"] * torch.minimum(
            self.state["FirmAssets"], torch.zeros_like(self.state["FirmAssets"])
        )
        deposit_interest = self.state["DepositRate"] * torch.maximum(
            self.state["FirmAssets"], torch.zeros_like(self.state["FirmAssets"])
        )
        cashflows = revenue + loan_interest + deposit_interest

        ydiv = torch.where(
            self.state["FirmProduction"] != 0.0, self.state["FirmProduction"], ones
        )
        cashflow_per_prod = torch.where(
            self.state["FirmProduction"] != 0.0, cashflows / ydiv, zero_vec
        )

        wage = torch.where(
            mask_wageplus > 0.0,
            torch.maximum(torch.minimum(wage, cashflow_per_prod), zero_vec),
            wage,
        )
        wage = torch.where(
            mask_wageplus > 0.0,
            torch.maximum(wage, zero_vec),
            wage,
        )

        hard_stay = (self.state["FirmStayAlive"] > 0.5).to(zero_vec.dtype)
        mask_wageminus = torch.where(
            hard_stay * self.state["FirmExcessSupplyMask"] * self.state["FirmProfits"]
            < 0.0,
            ones,
            zero_vec,
        )
        unemp = self.state["Unemployment"]
        coeff_dn = 1.0 - (1.0 - self.state["FirmRenSolvency"]) * rw * unemp
        wage = torch.where(
            mask_wageminus > 0.0,
            torch.maximum(wage * coeff_dn, zero_vec),
            wage,
        )

        self.state["FirmWage"] = wage

    def expectation_adjustments(self, t, scenario, params):
        r"""Anticipated inflation pass-through into prices and wages for
        surviving firms.

        Equations
        ---------
        .. math::
            \begin{align}
                P_{i,t} &\leftarrow P_{i,t}\,(1 + \hat\pi_t) \quad\text{if stay}_{i,t}, \\
                W_{i,t} &\leftarrow W_{i,t}\,(1 + \hat\pi_t\,w_f) \quad\text{if stay}_{i,t}.
            \end{align}

        Surviving firms transmit a fraction :math:`w_f` of the expected
        inflation to wages and the full amount to prices, building in a
        wedge that drives the loan-rate response.

        Dependency
        ----------
        - params: WageInflationFactor
        - state: ExpectedInflationUsed
        - state: FirmPrice
        - state: FirmWage
        - state: FirmStayAlive

        Sets
        ----
        - FirmPrice
        - FirmWage
        """
        wf = params["WageInflationFactor"]
        pi_used = self.state["ExpectedInflationUsed"]
        stay = self.state["FirmStayAlive"]

        self.state["FirmPrice"] = torch.where(
            stay > 0.0,
            self.state["FirmPrice"] * (1.0 + pi_used),
            self.state["FirmPrice"],
        )
        self.state["FirmWage"] = torch.where(
            stay > 0.0,
            self.state["FirmWage"] * (1.0 + pi_used * wf),
            self.state["FirmWage"],
        )

    def recompute_firm_totals_stayalive(self, t, scenario, params):
        r"""Ordering-critical re-aggregation after the wage/price update.

        Enforces non-negative production for surviving firms via
        :meth:`Behavior.diffwhere`, then recomputes macro totals from the
        stay-alive mask. Must fire before :meth:`bankrupt_firms` zeroes
        out the bankrupt slice.

        Equations
        ---------
        .. math::
            \begin{align}
                Y_{i,t} &\leftarrow \text{diffwhere}\big(\text{stay}_{i,t},\,\max(Y_{i,t}, 0),\,Y_{i,t}\big), \\
                Y_{tot} &= \sum_i \text{stay}_{i,t}\,Y_{i,t}, \\
                W_{tot} &= \sum_i \text{stay}_{i,t}\,W_{i,t}\,Y_{i,t}, \\
                \bar W_t &= W_{tot} / (Y_{tot} + \epsilon), \\
                \bar P_t &= \sum_i \text{stay}_{i,t}\,P_{i,t}\,Y_{i,t} / (Y_{tot} + \epsilon), \\
                S^{f+}_t &= \sum_i \text{stay}_{i,t}\,\max(A_{i,t}, 0), \\
                D^{f-}_t &= \sum_i \text{stay}_{i,t}\,\max(-A_{i,t}, 0).
            \end{align}

        The clamp keeps production weakly positive for solvent firms; the
        production-weighted averages and total payroll feed the
        subsequent bank-rate and household-consumption phases.

        Dependency
        ----------
        - hyper: epsilon
        - state: FirmProduction
        - state: FirmWage
        - state: FirmPrice
        - state: FirmAssets
        - state: FirmStayAlive

        Sets
        ----
        - FirmProduction
        - TotalProduction
        - TotalPayroll
        - AverageWage
        - AveragePrice
        - FirmSavingsTotal
        - FirmDebtTotal
        """
        zero_vec = torch.zeros_like(self.state["FirmProduction"])
        production = self.diffwhere(
            self.state["FirmStayAlive"],
            torch.maximum(self.state["FirmProduction"], zero_vec),
            self.state["FirmProduction"],
        )
        self.state["FirmProduction"] = production

        eps = self.hyper["epsilon"]
        ytot = (self.state["FirmStayAlive"] * production).sum()
        wtot = (self.state["FirmStayAlive"] * self.state["FirmWage"] * production).sum()
        wavg = wtot / (ytot + eps)
        pavg = (
            self.state["FirmStayAlive"] * self.state["FirmPrice"] * production
        ).sum() / (ytot + eps)
        firm_savings = (
            self.state["FirmStayAlive"]
            * torch.maximum(self.state["FirmAssets"], zero_vec)
        ).sum()
        firm_debt = (
            self.state["FirmStayAlive"]
            * (-torch.minimum(self.state["FirmAssets"], zero_vec))
        ).sum()

        self.state["TotalProduction"] = ytot.unsqueeze(0)
        self.state["TotalPayroll"] = wtot.unsqueeze(0)
        self.state["AverageWage"] = wavg.unsqueeze(0)
        self.state["AveragePrice"] = pavg.unsqueeze(0)
        self.state["FirmSavingsTotal"] = firm_savings.unsqueeze(0)
        self.state["FirmDebtTotal"] = firm_debt.unsqueeze(0)

    def compute_lowest_price(self, t, scenario, params):
        r"""Lowest price among surviving firms with positive price.

        Equations
        ---------
        .. math::
            \begin{align}
                P^{\min}_t = \min_{i \in \{i: \text{stay}_{i,t}\,P_{i,t} > 0\}} P_{i,t}.
            \end{align}

        Dead firms are masked to a large sentinel so the minimum picks
        up only live, positively-priced firms. The result anchors the
        household demand allocation in the next phase.

        Dependency
        ----------
        - state: FirmPrice
        - state: FirmStayAlive

        Sets
        ----
        - LowestPrice
        """
        alive_p = torch.where(
            self.state["FirmStayAlive"] * self.state["FirmPrice"] > 0,
            self.state["FirmPrice"],
            torch.full_like(self.state["FirmPrice"], 1.0e30),
        )
        self.state["LowestPrice"] = alive_p.min().unsqueeze(0)

    def bankrupt_firms(self, t, scenario, params):
        r"""Zero out production, assets, wage of bankrupt firms; record
        the defaulted asset total.

        Equations
        ---------
        .. math::
            \begin{align}
                D^{\text{def}}_t &= -\sum_i \text{enter}_{i,t}\,A_{i,t}, \\
                Y_{i,t},\,A_{i,t},\,\alpha_{i,t},\,W_{i,t} &\leftarrow 0
                    \quad\text{if}\quad \text{enter}_{i,t} = 1.
            \end{align}

        Bankrupt firms are zeroed simultaneously across production,
        assets, alive flag, and wage; the negative-asset sum surfaces as
        the system-level defaulted total, which the loan rate phase uses
        to set the bankruptcy-adjusted spread.

        Dependency
        ----------
        - state: FirmAssets
        - state: FirmProduction
        - state: FirmAlive
        - state: FirmWage
        - state: FirmStayAlive
        - state: FirmEnterBankruptcy

        Sets
        ----
        - FirmProduction
        - FirmAssets
        - FirmAlive
        - FirmWage
        - DefaultedTotal
        - FirmStayAlive
        """
        eb = self.state["FirmEnterBankruptcy"]
        deftot = (eb * self.state["FirmAssets"] * -1.0).sum()
        zero_vec = torch.zeros_like(self.state["FirmProduction"])

        bankrupt = eb == 1.0
        self.state["FirmProduction"] = torch.where(
            bankrupt, zero_vec, self.state["FirmProduction"]
        )
        self.state["FirmAssets"] = torch.where(
            bankrupt, zero_vec, self.state["FirmAssets"]
        )
        self.state["FirmAlive"] = torch.where(
            bankrupt, zero_vec, self.state["FirmAlive"]
        )
        self.state["FirmWage"] = torch.where(bankrupt, zero_vec, self.state["FirmWage"])
        self.state["DefaultedTotal"] = deftot.unsqueeze(0)
        self.state["FirmStayAlive"] = (
            self.state["FirmStayAlive"] * self.state["FirmAlive"]
        )

    def compute_moments(self, t, scenario, params):
        r"""Recompute wage and price moments after bankruptcy zeroing.

        Equations
        ---------
        .. math::
            \begin{align}
                W_{tot} &= \sum_i \text{stay}_{i,t}\,W_{i,t}\,Y_{i,t}, \\
                \bar W_t &= W_{tot} / (Y_{tot} + \epsilon), \\
                \bar P_t &= \sum_i \alpha_{i,t}\,P_{i,t}\,Y_{i,t} / (Y_{tot} + \epsilon), \\
                W^{\max}_t &= \max_i \alpha_{i,t}\,W_{i,t}.
            \end{align}

        The post-bankruptcy moments anchor the bank-side and
        household-side decisions: maximum wage drives the unemployment
        logit, average price drives the demand allocation, average wage
        feeds the consumption budget.

        Dependency
        ----------
        - hyper: epsilon
        - state: TotalProduction
        - state: FirmAlive
        - state: FirmWage
        - state: FirmPrice
        - state: FirmProduction
        - state: FirmStayAlive

        Sets
        ----
        - TotalPayroll
        - AverageWage
        - AveragePrice
        - MaxWage
        """
        eps = self.hyper["epsilon"]
        ytot = self.state["TotalProduction"]
        wtot = (
            self.state["FirmStayAlive"]
            * self.state["FirmWage"]
            * self.state["FirmProduction"]
        ).sum()
        wavg = wtot / (ytot + eps)
        pavg = (
            self.state["FirmAlive"]
            * self.state["FirmPrice"]
            * self.state["FirmProduction"]
        ).sum() / (ytot + eps)
        wmax = (self.state["FirmWage"] * self.state["FirmAlive"]).max()
        self.state["TotalPayroll"] = wtot.unsqueeze(0)
        self.state["AverageWage"] = wavg.unsqueeze(0)
        self.state["AveragePrice"] = pavg.unsqueeze(0)
        self.state["MaxWage"] = wmax.unsqueeze(0)

    def compute_employment(self, t, scenario, params):
        r"""Update aggregate employment and unemployment scalars.

        Equations
        ---------
        .. math::
            \begin{align}
                e_t &= Y_{tot} / N, \\
                u_t &= 1 - e_t.
            \end{align}

        Aggregate employment is the total production normalised to the
        labour-force ceiling; unemployment is the residual that
        downstream wage and revival phases consume.

        Dependency
        ----------
        - hyper: N_firms
        - state: TotalProduction

        Sets
        ----
        - Employment
        - Unemployment
        """
        n = float(self.hyper["N_firms"])
        employment = self.state["TotalProduction"] / n
        self.state["Employment"] = employment
        self.state["Unemployment"] = 1.0 - employment

    def solve_rounding_errors(self, t, scenario, params):
        r"""Patch the monetary-closure residual to absorb floating-point
        drift from earlier denominator regularisations.

        Notes
        -----
        The reference implementation gates the residual subtraction on a
        finite-residual condition. Without the gate, the
        ``+ epsilon`` leakage from divisor guards accumulates
        monotonically into ``HouseholdSavings``. The gate is preserved
        via ``torch.where`` value branches so the gradient still flows
        through the patched branch.

        Equations
        ---------
        .. math::
            \begin{align}
                R_t &= S_t + S^{f+}_t - D^{f-}_t - D^{\text{def}}_t - M^0_t, \\
                S_t &\leftarrow
                    \begin{cases}
                        S_t - R_t & \text{if}\ |R_t| > 10^{-9}, \\
                        S_t & \text{otherwise}.
                    \end{cases}
            \end{align}

        The patch enforces the money-stock identity
        :math:`M^0 = S + S^{f+} - D^{f-} - D^{\text{def}}` to within
        floating-point tolerance at every period.

        Dependency
        ----------
        - state: HouseholdSavings
        - state: FirmSavingsTotal
        - state: FirmDebtTotal
        - state: DefaultedTotal
        - state: M0Stock

        Sets
        ----
        - HouseholdSavings
        """
        count = (
            self.state["HouseholdSavings"]
            + self.state["FirmSavingsTotal"]
            - self.state["FirmDebtTotal"]
            - self.state["DefaultedTotal"]
            - self.state["M0Stock"]
        )
        patched = self.state["HouseholdSavings"] - count
        self.state["HouseholdSavings"] = torch.where(
            count.abs() > 1e-9, patched, self.state["HouseholdSavings"]
        )

    def set_interest_rates(self, t, scenario, params):
        r"""Set loan rate (CB rate plus bankruptcy-adjusted spread) and
        deposit rate (residual closing the bank's balance sheet); then
        accrue interest on household savings.

        Equations
        ---------
        .. math::
            \begin{align}
                \rho^l_t &=
                    \begin{cases}
                        \rho^{CB}_t + (1 - f)\,D^{\text{def}}_t / D^{f-}_t & \text{if}\ D^{f-}_t > 0, \\
                        \rho^{CB}_t & \text{otherwise},
                    \end{cases} \\
                I_t &= \rho^l_t\,D^{f-}_t, \\
                \rho^d_t &=
                    \begin{cases}
                        (I_t - D^{\text{def}}_t) / (S_t + S^{f+}_t) & \text{if}\ S_t + S^{f+}_t > 0, \\
                        0 & \text{otherwise},
                    \end{cases} \\
                S_t &\leftarrow (1 + \rho^d_t)\,S_t.
            \end{align}

        The loan-rate spread compensates the bank for default losses; the
        deposit rate is whatever residual balances total bank cashflow
        against accrued interest, so the bank's balance sheet closes
        without an exogenous funding source.

        Dependency
        ----------
        - params: BankruptcyInterestEffect
        - state: CBRate
        - state: FirmDebtTotal
        - state: DefaultedTotal
        - state: HouseholdSavings
        - state: FirmSavingsTotal

        Sets
        ----
        - LoanRate
        - DepositRate
        - HouseholdSavings
        """
        f = params["BankruptcyInterestEffect"]

        debt_tot = self.state["FirmDebtTotal"]
        deftot = self.state["DefaultedTotal"]
        ones = torch.ones_like(debt_tot)
        zeros = torch.zeros_like(debt_tot)

        div = torch.where(debt_tot != 0, debt_tot, ones)
        deftodebt = torch.where(debt_tot != 0, deftot / div, zeros)

        loan_rate = torch.where(
            debt_tot > 0,
            self.state["CBRate"] + (1.0 - f) * deftodebt,
            self.state["CBRate"],
        )
        self.state["LoanRate"] = loan_rate
        interests = loan_rate * debt_tot

        total_deposits = self.state["HouseholdSavings"] + self.state["FirmSavingsTotal"]
        deposit_rate = torch.where(
            total_deposits > 0,
            (interests - deftot)
            / torch.where(total_deposits != 0, total_deposits, ones),
            zeros,
        )
        self.state["DepositRate"] = deposit_rate
        self.state["HouseholdSavings"] = (1.0 + deposit_rate) * self.state[
            "HouseholdSavings"
        ]

    def household_consumption(self, t, scenario, params):
        r"""Update the household consumption propensity, set the budget,
        and allocate firm-level demand by a logit weight on the price
        gap to the lowest live price.

        Equations
        ---------
        .. math::
            \begin{align}
                c_t &= \mathrm{clamp}\big(c_0\,[1 + \alpha_c\,(\hat\pi_t - \bar\rho^d_t)],\,0,\,1\big), \\
                B_t &= c_t\,[W_{tot} + \max(S_t, 0)], \\
                z_{i,t} &= \beta\,(P^{\min}_t - P_{i,t}) / \bar P_t, \\
                D_{i,t} &= \alpha_{i,t}\,\frac{B_t\,e^{z_{i,t}}}{(\sum_j \alpha_{j,t}\,e^{z_{j,t}} + \epsilon)\,P_{i,t}}, \\
                D_t^{tot} &= \sum_i \alpha_{i,t}\,D_{i,t}.
            \end{align}

        Real-rate gaps raise the consumption propensity; the logit
        allocation sends a larger share of demand to lower-priced live
        firms, transmitting price competition into the real economy.

        Dependency
        ----------
        - params: ConsumptionPropensityBaseline
        - params: ConsumptionRealRateSensitivity
        - params: HouseholdIntensityOfChoice
        - hyper: epsilon
        - state: ExpectedInflationUsed
        - state: DepositRateEWMA
        - state: HouseholdSavings
        - state: TotalPayroll
        - state: FirmPrice
        - state: AveragePrice
        - state: FirmAlive
        - state: LowestPrice

        Sets
        ----
        - ConsumptionPropensity
        - ConsumptionBudget
        - FirmDemand
        - TotalDemand
        """
        c0 = params["ConsumptionPropensityBaseline"]
        alpha_c = params["ConsumptionRealRateSensitivity"]
        beta = params["HouseholdIntensityOfChoice"]
        eps = self.hyper["epsilon"]

        propensity = torch.clamp(
            c0
            * (
                1.0
                + alpha_c
                * (self.state["ExpectedInflationUsed"] - self.state["DepositRateEWMA"])
            ),
            min=0.0,
            max=1.0,
        )
        zero_scalar = torch.zeros_like(self.state["HouseholdSavings"])
        budget = propensity * (
            self.state["TotalPayroll"]
            + torch.maximum(self.state["HouseholdSavings"], zero_scalar)
        )

        arg = (
            beta
            * (self.state["LowestPrice"] - self.state["FirmPrice"])
            / self.state["AveragePrice"]
        )
        pnorm = (self.state["FirmAlive"] * torch.exp(arg)).sum()
        demand = self.state["FirmAlive"] * (
            budget * torch.exp(arg) / ((pnorm + eps) * self.state["FirmPrice"])
        )
        self.state["FirmDemand"] = demand
        self.state["TotalDemand"] = (
            (self.state["FirmAlive"] * demand).sum().unsqueeze(0)
        )
        self.state["ConsumptionPropensity"] = propensity
        self.state["ConsumptionBudget"] = budget

    def firm_accounting(self, t, scenario, params):
        r"""Realise per-firm EBIT and profits, draw down household savings
        by the aggregate EBIT, and update firm assets by the
        alive-weighted profit stream.

        Equations
        ---------
        .. math::
            \begin{align}
                \text{ebit}_{i,t} &= P_{i,t}\,\min(D_{i,t}, Y_{i,t}) - W_{i,t}\,Y_{i,t}, \\
                \Pi_{i,t} &= \text{ebit}_{i,t}
                    + \rho^l_t\,\min(A_{i,t}, 0)
                    + \rho^d_t\,\max(A_{i,t}, 0), \\
                A_{i,t} &\leftarrow A_{i,t} + \alpha_{i,t}\,\Pi_{i,t}, \\
                S_t &\leftarrow S_t - \sum_i \alpha_{i,t}\,\text{ebit}_{i,t}.
            \end{align}

        The accounting balances aggregate EBIT against the household
        savings drawdown and pushes profit signals into firm assets,
        carrying the previous-period balance-sheet effects into the next
        bankruptcy check.

        Dependency
        ----------
        - state: FirmPrice
        - state: FirmProduction
        - state: FirmDemand
        - state: FirmWage
        - state: FirmAssets
        - state: FirmAlive
        - state: LoanRate
        - state: DepositRate
        - state: HouseholdSavings

        Sets
        ----
        - FirmProfits
        - HouseholdSavings
        - FirmAssets
        """
        zero_vec = torch.zeros_like(self.state["FirmAssets"])
        ebit = (
            self.state["FirmPrice"]
            * torch.minimum(self.state["FirmProduction"], self.state["FirmDemand"])
            - self.state["FirmWage"] * self.state["FirmProduction"]
        )
        profits = ebit + (
            self.state["LoanRate"] * torch.minimum(self.state["FirmAssets"], zero_vec)
            + self.state["DepositRate"]
            * torch.maximum(self.state["FirmAssets"], zero_vec)
        )
        self.state["FirmProfits"] = profits
        self.state["HouseholdSavings"] = self.state["HouseholdSavings"] - (
            self.state["FirmAlive"] * ebit
        ).sum().unsqueeze(0)
        self.state["FirmAssets"] = (
            self.state["FirmAssets"] + self.state["FirmAlive"] * profits
        )

    def pay_dividends(self, t, scenario, params):
        r"""Pay dividends from firms with positive assets and profits;
        credit household savings, debit firm assets.

        Equations
        ---------
        .. math::
            \begin{align}
                m_{i,t} &= \alpha_{i,t}\,\mathbb{1}\{A_{i,t} > 0\}\,\mathbb{1}\{\Pi_{i,t} > 0\}, \\
                d_{i,t} &= m_{i,t}\,A_{i,t}\,\delta, \\
                S_t &\leftarrow S_t + \sum_i d_{i,t}, \\
                A_{i,t} &\leftarrow A_{i,t} - d_{i,t}.
            \end{align}

        The mask gates dividends on both balance-sheet sign and
        profitability, recycling part of the surviving firms' surplus
        back to households as a closed-loop income source.

        Dependency
        ----------
        - params: DividendShare
        - state: FirmAlive
        - state: FirmAssets
        - state: FirmProfits
        - state: HouseholdSavings

        Sets
        ----
        - HouseholdSavings
        - FirmAssets
        """
        delta = params["DividendShare"]
        ones = torch.ones_like(self.state["FirmAssets"])
        zeros = torch.zeros_like(self.state["FirmAssets"])
        mask = (
            self.state["FirmAlive"]
            * torch.where(self.state["FirmAssets"] > 0, ones, zeros)
            * torch.where(self.state["FirmProfits"] > 0, ones, zeros)
        )
        dividends = mask * self.state["FirmAssets"] * delta
        self.state["HouseholdSavings"] = self.state[
            "HouseholdSavings"
        ] + dividends.sum().unsqueeze(0)
        self.state["FirmAssets"] = self.state["FirmAssets"] - dividends

    def recompute_firm_totals_consumption(self, t, scenario, params):
        r"""Ordering-critical re-aggregation after firm accounting and
        dividends.

        Equations
        ---------
        .. math::
            \begin{align}
                D^{tot}_t &= \sum_i \alpha_{i,t}\,D_{i,t}, \\
                S^{f+}_t &= \sum_i \text{stay}_{i,t}\,\max(A_{i,t}, 0), \\
                A^{tot}_t &= \sum_i \alpha_{i,t}\,A_{i,t}.
            \end{align}

        The recompute refreshes the aggregate stocks the revival and
        balance-sheet-adjustment phases will read, after firm-level
        profits and dividends have moved cash around.

        Dependency
        ----------
        - state: FirmAlive
        - state: FirmDemand
        - state: FirmAssets
        - state: FirmStayAlive

        Sets
        ----
        - TotalDemand
        - FirmSavingsTotal
        - FirmAssetsTotal
        """
        zero_vec = torch.zeros_like(self.state["FirmAssets"])
        self.state["TotalDemand"] = (
            (self.state["FirmAlive"] * self.state["FirmDemand"]).sum().unsqueeze(0)
        )
        self.state["FirmSavingsTotal"] = (
            (
                self.state["FirmStayAlive"]
                * torch.maximum(self.state["FirmAssets"], zero_vec)
            )
            .sum()
            .unsqueeze(0)
        )
        self.state["FirmAssetsTotal"] = (
            (self.state["FirmAlive"] * self.state["FirmAssets"]).sum().unsqueeze(0)
        )

    def revive_firms(self, t, scenario, params):
        r"""Bernoulli revival of dead firms; revived firms receive a
        fresh production / price / wage / asset endowment.

        Notes
        -----
        Branch logic uses ``torch.where`` to match the reference;
        gradient w.r.t. :math:`\phi` does not flow through the
        indicator (treated as fixed-noise per period).

        Equations
        ---------
        .. math::
            \begin{align}
                m^{\text{rev}}_{i,t} &= (1 - \alpha_{i,t})\,\mathbb{1}\{u^r_{t,i} < \phi\}, \\
                Y_{i,t} &\leftarrow \max(u_t, 0)\,u^{r,Y}_{t,i}
                    \quad\text{if}\quad m^{\text{rev}}_{i,t} = 1, \\
                P_{i,t} &\leftarrow \bar P_t,\quad
                W_{i,t} \leftarrow \bar W_t
                    \quad\text{if}\quad m^{\text{rev}}_{i,t} = 1, \\
                A_{i,t} &\leftarrow W_{i,t}\,Y_{i,t}
                    \quad\text{if}\quad m^{\text{rev}}_{i,t} = 1, \\
                \Pi_{i,t} &\leftarrow 0
                    \quad\text{if}\quad m^{\text{rev}}_{i,t} = 1, \\
                D^{\text{def}}_t &= \sum_i m^{\text{rev}}_{i,t}\,A_{i,t}, \\
                S^{f+}_t &\leftarrow S^{f+}_t + D^{\text{def}}_t, \\
                \alpha_{i,t} &\leftarrow \alpha_{i,t} + m^{\text{rev}}_{i,t}.
            \end{align}

        Revival keeps the firm count steady and seeds new entrants near
        the average wage and price, preventing the population from
        collapsing.

        Dependency
        ----------
        - params: FirmRevivalProbability
        - state: FirmAlive
        - state: FirmProduction
        - state: FirmPrice
        - state: FirmWage
        - state: FirmAssets
        - state: FirmProfits
        - state: AveragePrice
        - state: AverageWage
        - state: Unemployment
        - state: FirmSavingsTotal

        Sets
        ----
        - FirmAlive
        - FirmProduction
        - FirmPrice
        - FirmWage
        - FirmAssets
        - FirmProfits
        - DefaultedTotal
        - FirmSavingsTotal
        """
        phi = params["FirmRevivalProbability"]
        zero_vec = torch.zeros_like(self.state["FirmProduction"])
        ones = torch.ones_like(self.state["FirmProduction"])
        zero_scalar = torch.zeros_like(self.state["Unemployment"])

        dead = 1.0 - self.state["FirmAlive"]
        draw_below_phi = (self._noise_revive[t] < phi).to(zero_vec.dtype)
        revive_mask = torch.where(dead * draw_below_phi == 1.0, ones, zero_vec)

        new_y_scale = (
            torch.maximum(self.state["Unemployment"], zero_scalar)
            * self._noise_revive_y[t]
        )
        new_y = new_y_scale * ones

        active = revive_mask == 1.0
        self.state["FirmProduction"] = torch.where(
            active, new_y, self.state["FirmProduction"]
        )
        self.state["FirmPrice"] = torch.where(
            active, self.state["AveragePrice"] * ones, self.state["FirmPrice"]
        )
        self.state["FirmWage"] = torch.where(
            active, self.state["AverageWage"] * ones, self.state["FirmWage"]
        )
        new_a = self.state["FirmWage"] * self.state["FirmProduction"]
        self.state["FirmAssets"] = torch.where(active, new_a, self.state["FirmAssets"])
        self.state["FirmProfits"] = torch.where(
            active, zero_vec, self.state["FirmProfits"]
        )

        deftot_revive = (self.state["FirmAssets"] * revive_mask).sum()
        self.state["DefaultedTotal"] = deftot_revive.unsqueeze(0)
        self.state["FirmSavingsTotal"] = self.state[
            "FirmSavingsTotal"
        ] + deftot_revive.unsqueeze(0)
        self.state["FirmAlive"] = self.state["FirmAlive"] + revive_mask

    def recompute_firm_totals_revival(self, t, scenario, params):
        r"""Ordering-critical re-aggregation after revival; absorb the
        default loss into surviving firms' assets, then refresh every
        macro aggregate that downstream phases will read.

        Equations
        ---------
        .. math::
            \begin{align}
                A_{i,t} &\leftarrow A_{i,t} - A_{i,t}\,\frac{D^{\text{def}}_t}{\tilde S^{f+}_t}
                    \quad\text{if}\quad \alpha_{i,t}\,S^{f+}_t\,A_{i,t} > 0, \\
                W_{tot} &= \sum_i \alpha_{i,t}\,W_{i,t}\,Y_{i,t}, \\
                Y_{tot} &= \sum_i \alpha_{i,t}\,Y_{i,t}, \\
                A_{tot} &= \sum_i \alpha_{i,t}\,A_{i,t}, \\
                D^{f-}_t &= \sum_i \alpha_{i,t}\,\max(-A_{i,t}, 0), \\
                W^{\max}_t &= \max_i \alpha_{i,t}\,W_{i,t}, \\
                \text{bust}_t &= (N - \sum_i \alpha_{i,t})/N, \\
                \bar P_t &= \sum_i \alpha_{i,t}\,P_{i,t}\,Y_{i,t} / (Y_{tot} + \epsilon), \\
                \bar W_t &= W_{tot} / (Y_{tot} + \epsilon).
            \end{align}

        The proportional default absorption keeps the aggregate-asset
        identity intact across revival; the refreshed moments feed the
        inflation phase and the next period's wage and price updates.

        Dependency
        ----------
        - hyper: epsilon
        - hyper: N_firms
        - state: FirmAssets
        - state: FirmAlive
        - state: FirmWage
        - state: FirmProduction
        - state: FirmPrice
        - state: FirmSavingsTotal
        - state: DefaultedTotal

        Sets
        ----
        - FirmAssets
        - TotalPayroll
        - TotalProduction
        - FirmAssetsTotal
        - FirmDebtTotal
        - MaxWage
        - BankruptcyRate
        - AveragePrice
        - AverageWage
        """
        eps = self.hyper["epsilon"]
        zero_vec = torch.zeros_like(self.state["FirmAssets"])
        n = float(self.hyper["N_firms"])

        firm_savings = self.state["FirmSavingsTotal"]
        safe_div = torch.where(
            firm_savings > 0.0, firm_savings, torch.ones_like(firm_savings)
        )
        deftot = self.state["DefaultedTotal"]
        adjustment = self.state["FirmAssets"] * deftot / safe_div
        self.state["FirmAssets"] = torch.where(
            self.state["FirmAlive"] * firm_savings * self.state["FirmAssets"] > 0.0,
            self.state["FirmAssets"] - adjustment,
            self.state["FirmAssets"],
        )

        alive = self.state["FirmAlive"]
        wtot = (alive * self.state["FirmWage"] * self.state["FirmProduction"]).sum()
        ytot = (alive * self.state["FirmProduction"]).sum()
        atot = (alive * self.state["FirmAssets"]).sum()
        debt_tot = (alive * (-torch.minimum(self.state["FirmAssets"], zero_vec))).sum()
        wmax = (alive * self.state["FirmWage"]).max()
        bust = (n - alive.sum()) / n

        self.state["TotalPayroll"] = wtot.unsqueeze(0)
        self.state["TotalProduction"] = ytot.unsqueeze(0)
        self.state["FirmAssetsTotal"] = atot.unsqueeze(0)
        self.state["FirmDebtTotal"] = debt_tot.unsqueeze(0)
        self.state["MaxWage"] = wmax.unsqueeze(0)
        self.state["BankruptcyRate"] = bust.unsqueeze(0)

        pavg = (
            alive * self.state["FirmPrice"] * self.state["FirmProduction"]
        ).sum() / (ytot + eps)
        wavg = wtot / (ytot + eps)
        self.state["AveragePrice"] = pavg.unsqueeze(0)
        self.state["AverageWage"] = wavg.unsqueeze(0)

    def compute_inflation_and_employment(self, t, scenario, params):
        r"""Inflation from the current/prior average price; aggregate
        employment update.

        Notes
        -----
        After :meth:`renormalize_prices` the carry-over price level is 1
        (the prior period's average price divided by itself), so
        inflation in the renormalised units is just
        :math:`\bar P_t - 1`.

        Equations
        ---------
        .. math::
            \begin{align}
                \pi_t &= \bar P_t - 1, \\
                e_t &= Y_{tot} / N, \\
                u_t &= 1 - e_t.
            \end{align}

        Re-deriving employment after revival closes the labour-market
        identity used by the next period's wage and consumption phases.

        Dependency
        ----------
        - hyper: N_firms
        - state: AveragePrice
        - state: TotalProduction

        Sets
        ----
        - Inflation
        - Employment
        - Unemployment
        """
        self.state["Inflation"] = self.state["AveragePrice"] - 1.0

        n = float(self.hyper["N_firms"])
        employment = self.state["TotalProduction"] / n
        self.state["Employment"] = employment
        self.state["Unemployment"] = 1.0 - employment

    def monetary_policy(self, t, scenario, params):
        r"""Central bank policy: Taylor-like rule on EWMA inflation only.

        Equations
        ---------
        .. math::
            \begin{align}
                \rho^{CB}_t = \rho^\star + \phi_\pi\,(\pi^{ema}_t - \pi^\star).
            \end{align}

        The CB raises the policy rate when EWMA inflation runs above
        target and lowers it when it runs below, feeding into the next
        period's loan-rate setting and the household propensity gap.

        Dependency
        ----------
        - params: InterestRateBaseline
        - params: CBInflationReaction
        - params: CBInflationTarget
        - state: ExpectedInflationEWMA

        Sets
        ----
        - CBRate
        """
        rho_star = params["InterestRateBaseline"]
        phi_pi = params["CBInflationReaction"]
        pi_star = params["CBInflationTarget"]
        self.state["CBRate"] = rho_star + phi_pi * (
            self.state["ExpectedInflationEWMA"] - pi_star
        )
