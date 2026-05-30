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
        ``hyper["seed"]``. Never touches the global torch RNG."""
        self._torch_rng = torch.Generator()
        self._torch_rng.manual_seed(int(self.hyper["seed"]))

    def initialize_noise_buffers(self):
        r"""Pre-draw four independent ``(T, N_firms)`` U(0,1) buffers, in the
        exact interleaved order abmstat uses live:
        ``price[t], wage[t], revive[t], revive_y[t]`` per period.

        Uses the global torch RNG (``torch.manual_seed``) so the underlying
        stream is bit-identical to abmstat's at the same seed. Saves and
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

        Initial price and production carry a small linear spread across firms
        ``2 * i / N - 1``, scaled by ``0.01``, replicating the abmstat
        reference. Initial assets are ``2 * Y_i * W_i * (i / N)`` and the
        money stock is rescaled to ``N`` after summing.

        Dependency
        ----------
        - params: InitialProductionScale
        - params: InterestRateBaseline
        - hyper: N_firms

        Sets
        ----
        - FirmPrice, FirmWage, FirmProduction, FirmDemand, FirmAssets,
          FirmProfits, FirmAlive
        - AveragePrice, AverageWage, MaxWage
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

    def initialize_macro(self):
        r"""Initialise macro state: savings, interest rates, employment,
        EWMA registers.

        Total money stock is fixed to ``N`` by rescaling the household
        savings :math:`S = N \cdot Y / (A + Y)` and firm assets
        :math:`A_i \leftarrow A_i \cdot N / (A_{tot} + Y_{tot})`. The CB
        rate, loan rate, and EWMA loan rate are all initialised to
        ``rho_star``.

        Dependency
        ----------
        - params: InterestRateBaseline
        - hyper: N_firms
        - prior: FirmAssets, FirmProduction, FirmWage

        Sets
        ----
        - HouseholdSavings, M0Stock, CBRate, LoanRate, DepositRate,
          DepositRateEWMA, LoanRateEWMA, Inflation, ExpectedInflationEWMA,
          UnemploymentEWMA, Employment, Unemployment, BankruptcyRate,
          ConsumptionPropensity, FirmSavingsTotal, FirmDebtTotal,
          TotalPayroll, TotalDemand
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
        self.state["TotalPayroll"] = self._payroll_vec.sum().unsqueeze(0)

    # ------------------------------------------------------------------
    # Phase methods
    # ------------------------------------------------------------------

    def renormalize_prices(self, t, scenario, params):
        r"""Rescale all nominal quantities by the average price so that
        :math:`\bar P_t \equiv 1` going into the wage/price update.

        Sets
        ----
        - FirmPrice, FirmWage, FirmAssets, FirmProfits, HouseholdSavings,
          AverageWage, MaxWage, M0Stock, AveragePrice

        Equations
        ---------
        .. math::
            \begin{align}
                P_{i,t} \leftarrow P_{i,t} / \bar P_{t-1}, \quad
                W_{i,t} \leftarrow W_{i,t} / \bar P_{t-1}, \quad
                A_{i,t} \leftarrow A_{i,t} / \bar P_{t-1}.
            \end{align}
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

        Equations
        ---------
        .. math::
            \begin{align}
                \pi^{ema}_t &= \omega\,\pi_{t-1} + (1-\omega)\,\pi^{ema}_{t-1}, \\
                \hat\pi_t &= \tau^T \pi^\star + \tau^R \pi^{ema}_t.
            \end{align}

        Sets
        ----
        - ExpectedInflationEWMA, DepositRateEWMA, LoanRateEWMA,
          UnemploymentEWMA, ExpectedInflationUsed
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

        Equations
        ---------
        .. math::
            \text{stay}_{i,t} = \alpha_{i,t-1} \cdot \text{diffwhere}\big(A_{i,t} + \Theta\, W_{i,t} Y_{i,t},\, 1,\, 0\big).

        Sets
        ----
        - TotalPayroll (firm-vector temp), stay_alive (scratch), enter_bankruptcy (scratch)
        """
        theta = params["DefaultThreshold"]
        payroll = self.state["FirmWage"] * self.state["FirmProduction"]
        payroll_affordability = self.state["FirmAssets"] + theta * payroll

        ones = torch.ones_like(payroll_affordability)
        zeros = torch.zeros_like(payroll_affordability)
        stay_alive = self.diffwhere(payroll_affordability, ones, zeros)
        enter_bankruptcy = (1.0 - stay_alive).pow(2)

        alive = self.state["FirmAlive"]
        self._stay_alive = alive * stay_alive
        self._enter_bankruptcy = alive * enter_bankruptcy
        self._payroll_vec = payroll

    def demand_production_imbalance(self, t, scenario, params):
        r"""Compute excess-demand and excess-supply masks.

        ``torch.where`` here is differentiable through the value branches
        (constants) but not the condition — fine because the condition does
        not depend on any parameter.

        Sets
        ----
        - excess_demand (scratch), excess_supply (scratch), dY (scratch)
        """
        one = torch.ones_like(self.state["FirmProduction"])
        zero = torch.zeros_like(self.state["FirmProduction"])

        dY = self._stay_alive * (
            self.state["FirmDemand"] - self.state["FirmProduction"]
        )
        excess_demand = torch.where(dY > 0, one, zero)
        excess_supply = self.state["FirmAlive"] * torch.where(
            self.state["FirmDemand"] - self.state["FirmProduction"] <= 0, one, zero
        )

        self._dY = dY
        self._excess_demand = excess_demand
        self._excess_supply = excess_supply

    def compute_gamma_and_ren(self, t, scenario, params):
        r"""Bank-side gamma (real-rate gap above baseline) and per-firm
        ren ratio (gamma times solvency).

        Equations
        ---------
        .. math::
            \begin{align}
                \Gamma_t &= \Gamma_0 + \text{ReLU}\big(\alpha_\Gamma (\bar\rho^l_t - \hat\pi_t) - \Gamma_0\big), \\
                \text{ren}_{i,t} &= \Gamma_t \cdot \frac{A_{i,t}}{W_{i,t} Y_{i,t} + \epsilon}.
            \end{align}

        Sets
        ----
        - FirmGamma, ren (scratch)
        """
        alpha_g = params["LoanRateGammaSensitivity"]
        gamma_0 = params["GammaBaseline"]
        eps = self.hyper["epsilon"]

        gap = (
            alpha_g * (self.state["LoanRateEWMA"] - self.state["ExpectedInflationUsed"])
            - gamma_0
        )
        gamma = gamma_0 + torch.relu(gap)
        ren = gamma * (self.state["FirmAssets"] / (self._payroll_vec + eps))

        self.state["FirmGamma"] = gamma
        self._ren = ren

    def compute_ushare(self, t, scenario, params):
        r"""Per-firm share of unemployed labour pool via logit weight on
        the wage gap to the maximum wage.

        Sets
        ----
        - u_share (scratch), wage_norm (scratch)
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

        self._u_share = u_share
        self._wage_norm = wage_norm
        self._arg = arg

    def production_adjustment(self, t, scenario, params):
        r"""Adjust production: increase when excess demand, decrease when
        excess supply. Eta_plus and eta_minus are clamped to ``[0, 1]``.

        Sets
        ----
        - FirmProduction
        """
        eta0m = params["FiringPropensity"]
        r_param = params["HiringFiringRate"]

        eta_plus = torch.clamp(eta0m * r_param * (1.0 + self._ren), 0.0, 1.0)
        eta_minus = torch.clamp(eta0m * (1.0 - self._ren), 0.0, 1.0)

        production = self.state["FirmProduction"]
        production = (
            production
            + self._stay_alive
            * self._excess_demand
            * torch.minimum(eta_plus * self._dY, self._u_share)
        )
        production = production + self._stay_alive * self._excess_supply * (
            eta_minus * self._dY
        )
        self.state["FirmProduction"] = production
        self._eta_plus = eta_plus
        self._eta_minus = eta_minus

    def price_adjustment(self, t, scenario, params):
        r"""Price update from the frozen-noise buffer.

        Excess-demand firms with below-average price scale ``P`` up by
        ``(1 + rp)``; excess-supply firms with above-average price scale
        down by ``(1 - rp)``. Branching uses ``torch.where`` to match
        abmstat; gradient flow w.r.t. ``gammap`` is through ``rp``.

        Equations
        ---------
        .. math::
            \begin{align}
                r_p &= \gamma_p \, u^p_{t,i}, \\
                P_{i,t} &\leftarrow P_{i,t} (1 + r_p)
                  \text{ if stay}_{i,t}\,\text{excess}^D_{i,t}\,(P_{i,t} < \bar P_t).
            \end{align}

        Sets
        ----
        - FirmPrice, priceup (scratch), pricedown (scratch)
        """
        gammap = params["PriceAdjustmentSize"]
        rp = gammap * self._noise_price[t]

        price = self.state["FirmPrice"]
        pavg = self.state["AveragePrice"]
        ones = torch.ones_like(price)
        zeros = torch.zeros_like(price)

        priceup = (
            self._stay_alive
            * self._excess_demand
            * torch.where(price < pavg, ones, zeros)
        )
        price = torch.where(priceup > 0.0, price * (1.0 + rp), price)

        pricedown = torch.where(self._excess_supply * price > pavg, ones, zeros)
        price = torch.where(pricedown > 0.0, price * (1.0 - rp), price)

        self.state["FirmPrice"] = price
        self._priceup = priceup
        self._pricedown = pricedown

    def wage_adjustment(self, t, scenario, params):
        r"""Smooth wage update from the frozen-noise buffer.

        Excess-demand profitable firms raise wages; excess-supply
        loss-making firms lower wages. The wage ceiling
        (cashflow-per-production) is enforced via ``diffmin``.

        Equations
        ---------
        .. math::
            r_w = \gamma_p \cdot r \cdot u^w_{t,i}.

        Sets
        ----
        - FirmWage, mask_wageplus (scratch), mask_wageminus (scratch)
        """
        gammap = params["PriceAdjustmentSize"]
        r_ratio = params["WagePriceAdjustmentRatio"]
        zero_vec = torch.zeros_like(self.state["FirmWage"])
        ones = torch.ones_like(self.state["FirmWage"])

        rw = gammap * r_ratio * self._noise_wage[t]

        mask_wageplus = torch.where(
            self._stay_alive
            * self._excess_demand
            * (self.state["FirmProfits"] > 0).to(zero_vec.dtype)
            > 0.0,
            ones,
            zero_vec,
        )

        employment = self.state["Employment"]
        coeff_up = 1.0 + (1.0 + self._ren) * rw * employment
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

        hard_stay = (self._stay_alive > 0.5).to(zero_vec.dtype)
        mask_wageminus = torch.where(
            hard_stay * self._excess_supply * self.state["FirmProfits"] < 0.0,
            ones,
            zero_vec,
        )
        unemp = self.state["Unemployment"]
        coeff_dn = 1.0 - (1.0 - self._ren) * rw * unemp
        wage = torch.where(
            mask_wageminus > 0.0,
            torch.maximum(wage * coeff_dn, zero_vec),
            wage,
        )

        self.state["FirmWage"] = wage
        self._mask_wageplus = mask_wageplus
        self._mask_wageminus = mask_wageminus

    def expectation_adjustments(self, t, scenario, params):
        r"""Anticipated inflation pass-through into prices and wages for
        surviving firms.

        Sets
        ----
        - FirmPrice, FirmWage
        """
        wf = params["WageInflationFactor"]
        pi_used = self.state["ExpectedInflationUsed"]
        stay = self._stay_alive

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

        Enforces non-negative production for surviving firms and recomputes
        macro totals from the stay-alive mask. The recompute must happen
        before ``bankrupt_firms`` zeroes out the bankrupt slice.

        Sets
        ----
        - FirmProduction (clamped), TotalProduction, TotalPayroll,
          AverageWage, AveragePrice, FirmSavingsTotal, FirmDebtTotal
        """
        zero_vec = torch.zeros_like(self.state["FirmProduction"])
        production = self.diffwhere(
            self._stay_alive,
            torch.maximum(self.state["FirmProduction"], zero_vec),
            self.state["FirmProduction"],
        )
        self.state["FirmProduction"] = production

        eps = self.hyper["epsilon"]
        ytot = (self._stay_alive * production).sum()
        wtot = (self._stay_alive * self.state["FirmWage"] * production).sum()
        wavg = wtot / (ytot + eps)
        pavg = (self._stay_alive * self.state["FirmPrice"] * production).sum() / (
            ytot + eps
        )
        firm_savings = (
            self._stay_alive * torch.maximum(self.state["FirmAssets"], zero_vec)
        ).sum()
        firm_debt = (
            self._stay_alive * (-torch.minimum(self.state["FirmAssets"], zero_vec))
        ).sum()

        self.state["TotalProduction"] = ytot.unsqueeze(0)
        self.state["TotalPayroll"] = wtot.unsqueeze(0)
        self.state["AverageWage"] = wavg.unsqueeze(0)
        self.state["AveragePrice"] = pavg.unsqueeze(0)
        self.state["FirmSavingsTotal"] = firm_savings.unsqueeze(0)
        self.state["FirmDebtTotal"] = firm_debt.unsqueeze(0)

    def compute_lowest_price(self, t, scenario, params):
        r"""Lowest price among surviving firms.

        Sets
        ----
        - pmin (scratch)
        """
        alive_p = torch.where(
            self._stay_alive * self.state["FirmPrice"] > 0,
            self.state["FirmPrice"],
            torch.full_like(self.state["FirmPrice"], 1.0e30),
        )
        self._pmin = alive_p.min()

    def bankrupt_firms(self, t, scenario, params):
        r"""Zero out production, assets, wage of bankrupt firms; record
        defaulted total.

        Sets
        ----
        - FirmProduction, FirmAssets, FirmAlive, FirmWage, DefaultedTotal
        """
        eb = self._enter_bankruptcy
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
        self._stay_alive = self._stay_alive * self.state["FirmAlive"]

    def compute_moments(self, t, scenario, params):
        r"""Recompute wage and price moments after bankruptcy.

        Sets
        ----
        - TotalPayroll, AverageWage, AveragePrice, MaxWage
        """
        eps = self.hyper["epsilon"]
        ytot = self.state["TotalProduction"]
        wtot = (
            self._stay_alive * self.state["FirmWage"] * self.state["FirmProduction"]
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
        r"""Update employment and unemployment scalars.

        Sets
        ----
        - Employment, Unemployment
        """
        n = float(self.hyper["N_firms"])
        employment = self.state["TotalProduction"] / n
        self.state["Employment"] = employment
        self.state["Unemployment"] = 1.0 - employment

    def solve_rounding_errors(self, t, scenario, params):
        r"""Monetary-closure residual patch.

        abmstat gates ``S -= count`` on ``|count| > 0`` (line 937). Without the
        gate, ``+ eps`` leakage from earlier denominator regularisations
        accumulates monotonically into ``HouseholdSavings``. Gate is preserved
        via ``torch.where`` value branches — gradient still flows through the
        patched branch.

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
        r"""Set loan rate (CB rate + bankruptcy-adjusted spread) and
        deposit rate (residual closing the bank's balance sheet).

        Sets
        ----
        - LoanRate, DepositRate, HouseholdSavings
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
        r"""Logit demand allocation across firms; updates household
        propensity, budget, demand, and total demand.

        Sets
        ----
        - ConsumptionPropensity, ConsumptionBudget, FirmDemand, TotalDemand
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

        arg = beta * (self._pmin - self.state["FirmPrice"]) / self.state["AveragePrice"]
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
        r"""EBIT, profits, household savings drawdown, asset update.

        Equations
        ---------
        .. math::
            \begin{align}
                \text{ebit}_{i,t} &= P_{i,t} \min(D_{i,t}, Y_{i,t}) - W_{i,t} Y_{i,t}, \\
                \Pi_{i,t} &= \text{ebit}_{i,t} + \rho^l_t \min(A_{i,t}, 0) + \rho^d_t \max(A_{i,t}, 0), \\
                A_{i,t} &\leftarrow A_{i,t} + \alpha_{i,t} \Pi_{i,t}, \\
                S_t &\leftarrow S_t - \sum_i \alpha_{i,t} \text{ebit}_{i,t}.
            \end{align}

        Sets
        ----
        - FirmProfits, HouseholdSavings, FirmAssets
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
        r"""Pay dividends from firms with positive assets and profits.

        Sets
        ----
        - HouseholdSavings, FirmAssets
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

        Sets
        ----
        - TotalDemand, FirmProfits (sum aggregated implicitly),
          FirmSavingsTotal, FirmAssetsTotal
        """
        zero_vec = torch.zeros_like(self.state["FirmAssets"])
        self.state["TotalDemand"] = (
            (self.state["FirmAlive"] * self.state["FirmDemand"]).sum().unsqueeze(0)
        )
        self.state["FirmSavingsTotal"] = (
            (self._stay_alive * torch.maximum(self.state["FirmAssets"], zero_vec))
            .sum()
            .unsqueeze(0)
        )
        self.state["FirmAssetsTotal"] = (
            (self.state["FirmAlive"] * self.state["FirmAssets"]).sum().unsqueeze(0)
        )

    def revive_firms(self, t, scenario, params):
        r"""Bernoulli revival of dead firms.

        Inverse-CDF draw:
        :math:`\text{revive}_{i,t} = (1 - \alpha_{i,t})\,\mathbf{1}\{u^r_{t,i} < \phi\}`
        with :math:`u^r_{t,i} \sim U(0,1)` pre-drawn in
        :meth:`initialize_noise_buffers`. Branch logic uses ``torch.where``
        to match abmstat; gradient w.r.t. ``phi`` does not flow through the
        indicator (treated as fixed-noise per period).

        Revived firms receive fresh ``Y, P, W, A``. Defaulted total is
        updated with the new firm assets.

        Sets
        ----
        - FirmAlive, FirmProduction, FirmPrice, FirmWage, FirmAssets,
          FirmProfits, DefaultedTotal, FirmSavingsTotal
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
        self._revive_mask = revive_mask

    def recompute_firm_totals_revival(self, t, scenario, params):
        r"""Ordering-critical re-aggregation after revival and the
        deftot-driven balance-sheet adjustment.

        Adjusts firm assets to absorb the default loss proportional to
        positive assets, then re-aggregates payroll, production, debt, and
        price moments.

        Sets
        ----
        - FirmAssets, TotalPayroll, TotalProduction, FirmAssetsTotal,
          FirmDebtTotal, MaxWage, BankruptcyRate, AveragePrice, AverageWage
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
        self._pavg_for_inflation = pavg
        self.state["AveragePrice"] = pavg.unsqueeze(0)
        self.state["AverageWage"] = wavg.unsqueeze(0)

    def compute_inflation_and_employment(self, t, scenario, params):
        r"""Inflation from current/prior average price; employment update.

        After :meth:`renormalize_prices`, the carry-over price level is 1
        (the prior period's ``AveragePrice`` divided by itself, mirroring
        abmstat's ``Pold = Pold / Pavg``), so inflation in renormalised units
        is ``state["AveragePrice"] - 1``.

        Sets
        ----
        - Inflation, Employment, Unemployment
        """
        self.state["Inflation"] = self.state["AveragePrice"] - 1.0

        n = float(self.hyper["N_firms"])
        employment = self.state["TotalProduction"] / n
        self.state["Employment"] = employment
        self.state["Unemployment"] = 1.0 - employment

    def monetary_policy(self, t, scenario, params):
        r"""Central bank policy: Taylor-like rule on EMA inflation only.

        Equations
        ---------
        .. math::
            \rho^0_t = \rho^\star + \phi_\pi (\pi^{ema}_t - \pi^\star).

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
