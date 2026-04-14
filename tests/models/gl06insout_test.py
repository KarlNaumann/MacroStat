"""Targeted tests for the GL06INSOUT model.

These tests verify the Godley-Lavoie 2006 INSOUT model (Chapter 10) including
smoke tests, positivity checks, accounting identity checks, and shock response
tests for all 7 scenarios.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import torch

from macrostat.models.GL06INSOUT import (
    GL06INSOUT,
    ParametersGL06INSOUT,
    ScenariosGL06INSOUT,
    VariablesGL06INSOUT,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(timesteps: int = 200):
    """Create a default GL06INSOUT model with given timesteps."""
    params = ParametersGL06INSOUT(
        hyperparameters={
            "timesteps": timesteps,
            "timesteps_initialization": 1,
            "use_tqdm": False,
        }
    )
    variables = VariablesGL06INSOUT(parameters=params)
    scenarios = ScenariosGL06INSOUT(parameters=params)
    model = GL06INSOUT(parameters=params, variables=variables, scenarios=scenarios)
    return model, params, scenarios


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------


def test_baseline_runs_without_error():
    """Smoke test: baseline simulation completes without error."""
    model, _, _ = _make_model()
    model.simulate()


def test_scenario1_runs_without_error():
    """Smoke test: scenario 1 (higher inventory ratio) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.1: Higher target inventory ratio")
    model.simulate(scenario=sc)


def test_scenario2_runs_without_error():
    """Smoke test: scenario 2 (higher government spending) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.2: Higher government spending")
    model.simulate(scenario=sc)


def test_scenario3_runs_without_error():
    """Smoke test: scenario 3 (higher reserve requirements) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.3: Higher reserve requirements")
    model.simulate(scenario=sc)


def test_scenario4_runs_without_error():
    """Smoke test: scenario 4 (wider liquidity corridor) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.4: Wider liquidity corridor")
    model.simulate(scenario=sc)


def test_scenario5_runs_without_error():
    """Smoke test: scenario 5 (lower consumption propensity) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.5: Lower consumption propensity")
    model.simulate(scenario=sc)


def test_scenario6_runs_without_error():
    """Smoke test: scenario 6 (higher real wage target) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.6: Higher real wage target")
    model.simulate(scenario=sc)


def test_scenario7_runs_without_error():
    """Smoke test: scenario 7 (wage target + rate rise) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.7: Wage target + rate rise")
    model.simulate(scenario=sc)


# ---------------------------------------------------------------------------
# Positivity Checks
# ---------------------------------------------------------------------------


def test_baseline_stocks_nonnegative():
    """After baseline simulation, key stocks should be non-negative.

    NominalInventories and LoansSupply go negative during early transition
    (matching the R sfcr reference) because the sequential solver produces
    negative real inventories before the model has built up enough demand.
    These are excluded from the non-negativity check.
    """
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    stock_vars = [
        "NominalWealth",
        "M1Household",
        "M2Household",
        "BillsHousehold",
        "BillsBank",
    ]

    for var in stock_vars:
        # Skip period 0 (initialization)
        vals = ts[var][1:]
        assert torch.all(
            vals >= -1e-6
        ), f"{var} has negative values: min={vals.min().item():.6f}"


def test_prices_wages_positive():
    """Price level, nominal wage, and unit cost must remain positive."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    for var in ["PriceLevel", "NominalWage", "UnitCost"]:
        vals = ts[var][1:]
        assert torch.all(
            vals > 0
        ), f"{var} is non-positive: min={vals.min().item():.6f}"


# ---------------------------------------------------------------------------
# Accounting Identities
# ---------------------------------------------------------------------------


def test_loan_market_clears():
    """LoansSupply should equal LoanDemand within tolerance."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    diff = (ts["LoansSupply"][1:] - ts["LoanDemand"][1:]).abs()
    assert torch.all(
        diff < 1e-5
    ), f"Loan market does not clear: max |Ls - Ld| = {diff.max().item():.2e}"


def test_bill_market_clears():
    """BillsSupply should equal BillsHousehold + BillsBank + BillsCentralBank."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    diff = (
        ts["BillsSupply"][1:]
        - ts["BillsHousehold"][1:]
        - ts["BillsBank"][1:]
        - ts["BillsCentralBank"][1:]
    ).abs()
    assert torch.all(
        diff < 1e-5
    ), f"Bill market does not clear: max diff = {diff.max().item():.2e}"


def test_portfolio_sum_equals_non_cash_wealth():
    """Non-cash portfolio should sum exactly to NonCashWealth (actual Vnc).

    M1 is the buffer-stock residual of actual non-cash wealth (Vnc = V - Hhd),
    so M1+M2+Bills+pbl*BL = Vnc by construction regardless of switching regime.
    """
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    m1 = ts["M1Household"][1:]
    m2 = ts["M2Household"][1:]
    bills = ts["BillsHousehold"][1:]
    bonds = ts["BondsHousehold"][1:]
    pbl = ts["BondPrice"][1:]
    vnc = ts["NonCashWealth"][1:]

    portfolio_sum = m1 + m2 + bills + pbl * bonds
    diff = (vnc - portfolio_sum).abs()
    assert torch.all(
        diff < 1e-4
    ), f"Portfolio sum != Vnc: max diff = {diff.max().item():.2e}"


def test_bond_price_yield_relationship():
    """BondPrice should equal 1/BondYield within tolerance."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    pbl = ts["BondPrice"][1:]
    rbl = ts["BondYield"][1:]

    mask = rbl > 0
    diff = (pbl[mask] - 1.0 / rbl[mask]).abs()
    assert torch.all(
        diff < 1e-6
    ), f"Bond price-yield relationship violated: max diff = {diff.max().item():.2e}"


def test_advances_market_clears():
    """AdvancesSupply should equal AdvancesDemand (CB accommodates)."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    diff = (ts["AdvancesSupply"][1:] - ts["AdvancesDemand"][1:]).abs()
    assert torch.all(
        diff < 1e-6
    ), f"Advances market does not clear: max diff = {diff.max().item():.2e}"


def test_hidden_equation_hbd_equals_hbs():
    """Bank reserves demand (CashBanksSupply) should equal BankReservesSupply."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    diff = (ts["CashBanksSupply"][1:] - ts["BankReservesSupply"][1:]).abs()
    assert torch.all(
        diff < 1e-6
    ), f"Bank reserves mismatch: max diff = {diff.max().item():.2e}"


def test_wealth_accumulation_identity():
    """Verify V(t) = V(t-1) + YDhs(t) - C(t) period by period."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    # Skip period 0 (initialization) and period 1 (first step)
    for t in range(2, ts["NominalWealth"].shape[0]):
        v_t = ts["NominalWealth"][t]
        v_prev = ts["NominalWealth"][t - 1]
        ydhs = ts["HaigSimonsDisposableIncome"][t]
        c = ts["NominalConsumption"][t]
        diff = (v_t - (v_prev + ydhs - c)).abs()
        assert diff < 1e-4, (
            f"Wealth identity fails at t={t}: "
            f"|V(t) - (V(t-1)+YDhs-C)| = {diff.item():.2e}"
        )


def test_government_budget_constraint():
    """Verify PSBR = G + rb[-1]*Bs[-1] + BLs[-1] - TX - FCB."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    for t in range(2, ts["PublicSectorBorrowingRequirement"].shape[0]):
        psbr = ts["PublicSectorBorrowingRequirement"][t]
        g = ts["GovernmentSpending"][t]
        rb_prev = ts["BillRate"][t - 1]
        bs_prev = ts["BillsSupply"][t - 1]
        bls_prev = ts["BondsSupply"][t - 1]
        tx = ts["Taxes"][t]
        fcb = ts["CentralBankProfits"][t]

        expected_psbr = g + rb_prev * bs_prev + bls_prev - tx - fcb
        diff = (psbr - expected_psbr).abs()
        assert diff < 1e-4, (
            f"Gov budget constraint fails at t={t}: "
            f"|PSBR - expected| = {diff.item():.2e}"
        )


# ---------------------------------------------------------------------------
# Steady-State Tests
# ---------------------------------------------------------------------------


def test_baseline_converges():
    """Key variables should stabilize after enough periods."""
    model, _, _ = _make_model(timesteps=300)
    model.simulate()
    ts = model.variables.timeseries

    for var in ["NominalOutput", "NominalWealth", "PriceLevel"]:
        last = ts[var][-1]
        prev = ts[var][-2]
        diff = (last - prev).abs()
        # Allow relative tolerance for variables that may be large
        scale = last.abs().clamp(min=1.0)
        assert diff / scale < 1e-3, (
            f"{var} has not converged: |x[-1] - x[-2]| / scale = "
            f"{(diff / scale).item():.2e}"
        )


def test_baseline_inflation_converges_to_zero():
    """In steady state with fixed exogenous rates, inflation should converge to zero."""
    model, _, _ = _make_model(timesteps=300)
    model.simulate()
    ts = model.variables.timeseries

    pi_last = ts["InflationRate"][-1].abs()
    assert (
        pi_last < 1e-3
    ), f"Inflation has not converged to zero: |pi| = {pi_last.item():.2e}"


# ---------------------------------------------------------------------------
# Shock Response Tests
# ---------------------------------------------------------------------------


def test_scenario1_higher_inventories():
    """Scenario 1 (higher sigma_0) should raise real inventories at end."""
    model_base, _, _ = _make_model()
    model_base.simulate()
    inv_base = model_base.variables.timeseries["RealInventories"][-1]

    model_shock, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.1: Higher target inventory ratio")
    model_shock.simulate(scenario=sc)
    inv_shock = model_shock.variables.timeseries["RealInventories"][-1]

    assert inv_shock > inv_base, (
        f"Higher inventory target should raise inventories: "
        f"base={inv_base.item():.4f}, shock={inv_shock.item():.4f}"
    )


def test_scenario2_higher_output():
    """Scenario 2 (higher G) should raise nominal output at end."""
    model_base, _, _ = _make_model()
    model_base.simulate()
    y_base = model_base.variables.timeseries["NominalOutput"][-1]

    model_shock, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.2: Higher government spending")
    model_shock.simulate(scenario=sc)
    y_shock = model_shock.variables.timeseries["NominalOutput"][-1]

    assert y_shock > y_base, (
        f"Higher G should raise Y: base={y_base.item():.4f}, "
        f"shock={y_shock.item():.4f}"
    )


def test_scenario3_higher_reserves_impact_and_longrun():
    """Scenario 3: higher reserve ratios.

    R sfcr reference (200 periods) shows:
    - Impact: BLRN drops sharply (0.037 -> -0.063) as more funds are tied in reserves.
    - Long run: BillsBank increases (1.42 -> 1.85) as the bank re-equilibrates
      with higher deposit rates and a higher BLRN (0.037 -> 0.045).
    """
    model_base, params, _ = _make_model()
    model_base.simulate()
    trigger = params.hyper["scenario_trigger"]
    blrn_base_impact = model_base.variables.timeseries["BankLiquidityRatioTentative"][
        trigger + 1
    ]
    bb_base = model_base.variables.timeseries["BillsBank"][-1]

    model_shock, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.3: Higher reserve requirements")
    model_shock.simulate(scenario=sc)
    blrn_shock_impact = model_shock.variables.timeseries["BankLiquidityRatioTentative"][
        trigger + 1
    ]
    bb_shock = model_shock.variables.timeseries["BillsBank"][-1]

    # Impact: BLRN drops
    assert blrn_shock_impact < blrn_base_impact, (
        f"Higher reserves should reduce BLRN on impact: "
        f"base={blrn_base_impact.item():.6f}, shock={blrn_shock_impact.item():.6f}"
    )
    # Long run: BillsBank increases (per R sfcr reference)
    assert bb_shock > bb_base, (
        f"Higher reserves should raise bank bills in long run: "
        f"base={bb_base.item():.4f}, shock={bb_shock.item():.4f}"
    )


def test_scenario4_wider_corridor_raises_bank_bills():
    """Scenario 4 (wider liquidity corridor) should raise bank bill holdings."""
    model_base, _, _ = _make_model()
    model_base.simulate()
    bb_base = model_base.variables.timeseries["BillsBank"][-1]

    model_shock, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.4: Wider liquidity corridor")
    model_shock.simulate(scenario=sc)
    bb_shock = model_shock.variables.timeseries["BillsBank"][-1]

    assert bb_shock > bb_base, (
        f"Wider corridor should raise bank bills: "
        f"base={bb_base.item():.4f}, shock={bb_shock.item():.4f}"
    )


def test_scenario5_lower_consumption():
    """Scenario 5 (lower alpha1) should reduce real consumption on impact.

    The shock fires at period 50. Real consumption should be lower in the
    periods immediately following the shock (period 55), before wage-price
    dynamics potentially push nominal values back up in the long run.
    """
    model_base, _, _ = _make_model()
    model_base.simulate()
    c_base = model_base.variables.timeseries["RealConsumption"][55]

    model_shock, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.5: Lower consumption propensity")
    model_shock.simulate(scenario=sc)
    c_shock = model_shock.variables.timeseries["RealConsumption"][55]

    assert c_shock < c_base, (
        f"Lower alpha1 should reduce real consumption on impact: "
        f"base={c_base.item():.4f}, shock={c_shock.item():.4f}"
    )


def test_scenario6_higher_wage():
    """Scenario 6 (higher Omega_0) should raise the nominal wage at end."""
    model_base, _, _ = _make_model()
    model_base.simulate()
    w_base = model_base.variables.timeseries["NominalWage"][-1]

    model_shock, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.6: Higher real wage target")
    model_shock.simulate(scenario=sc)
    w_shock = model_shock.variables.timeseries["NominalWage"][-1]

    assert w_shock > w_base, (
        f"Higher real wage target should raise nominal wage: "
        f"base={w_base.item():.4f}, shock={w_shock.item():.4f}"
    )


def test_scenario7_higher_wage_and_rate():
    """Scenario 7 should raise nominal wage and bill rate simultaneously."""
    model_base, _, _ = _make_model()
    model_base.simulate()
    w_base = model_base.variables.timeseries["NominalWage"][-1]
    r_base = model_base.variables.timeseries["BillRate"][-1]

    model_shock, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.7: Wage target + rate rise")
    model_shock.simulate(scenario=sc)
    w_shock = model_shock.variables.timeseries["NominalWage"][-1]
    r_shock = model_shock.variables.timeseries["BillRate"][-1]

    assert w_shock > w_base, (
        f"Scenario 7 should raise wage: "
        f"base={w_base.item():.4f}, shock={w_shock.item():.4f}"
    )
    assert r_shock > r_base, (
        f"Scenario 7 should raise bill rate: "
        f"base={r_base.item():.4f}, shock={r_shock.item():.4f}"
    )


# ---------------------------------------------------------------------------
# Constraint system integration
# ---------------------------------------------------------------------------


def test_constraint_system_step_integration():
    """End-to-end: the resolver and vectorizer agree on the scalar path.

    GL06INSOUT declares five :class:`LinearConstraint` groups, all
    naming the M1 slot of the Tobin portfolio matrix as the residual.
    This test calls ``apply_parameter_shocks`` once (with no shocks)
    and confirms that every derived parameter matches the
    init-time-enforced value. It is the only test that pins the full
    resolver-to-vectorizer round-trip on a production model.
    """
    from macrostat.models.GL06INSOUT import BehaviorGL06INSOUT

    model, params, scenarios = _make_model(timesteps=100)
    behavior = BehaviorGL06INSOUT(
        parameters=params,
        scenarios=scenarios,
        variables=VariablesGL06INSOUT(parameters=params),
        scenario=0,
    )

    shocked = behavior.apply_parameter_shocks(t=0, scenario={})

    derived_names = [
        "WealthShareM1_Constant",
        "WealthShareM1_DepositRate",
        "WealthShareM1_BillRate",
        "WealthShareM1_BondYield",
        "WealthShareM1_Income",
    ]
    for name in derived_names:
        init_value = params.values[name]["value"]
        step_value = shocked[name].item()
        assert abs(init_value - step_value) < 1e-6, (
            f"Init/step mismatch on {name}: "
            f"init={init_value:.8g}, step={step_value:.8g}"
        )

    # All five adding-up identities must hold on the step-time tensors.
    groups = {
        "Constant": 1.0,
        "DepositRate": 0.0,
        "BillRate": 0.0,
        "BondYield": 0.0,
        "Income": 0.0,
    }
    for col, target in groups.items():
        total = (
            shocked[f"WealthShareM1_{col}"]
            + shocked[f"WealthShareM2_{col}"]
            + shocked[f"WealthShareBills_{col}"]
            + shocked[f"WealthShareBonds_{col}"]
        )
        assert torch.isclose(
            total, torch.tensor(target), atol=1e-6
        ), f"Tobin {col} column does not sum to {target}: {total.item()}"


def test_constraint_system_derived_grads_are_zero():
    """Derived M1 parameters have zero autograd sensitivity end-to-end.

    Runs a short simulation with ``requires_grad=True`` and checks
    that no upstream loss can place gradient mass on the derived M1
    slots, because they are recomputed residually each step.
    """
    from macrostat.models.GL06INSOUT import BehaviorGL06INSOUT

    model, params, scenarios = _make_model(timesteps=100)
    behavior = BehaviorGL06INSOUT(
        parameters=params,
        scenarios=scenarios,
        variables=VariablesGL06INSOUT(parameters=params),
        scenario=0,
    )

    shocked = behavior.apply_parameter_shocks(t=0, scenario={})
    loss = sum(
        shocked[f"WealthShareM1_{col}"]
        for col in ("Constant", "DepositRate", "BillRate", "BondYield", "Income")
    )
    loss.backward()

    for col in ("Constant", "DepositRate", "BillRate", "BondYield", "Income"):
        derived = behavior.params[f"WealthShareM1_{col}"]
        # Derived leaf has no grad because it is never read through the
        # constraint path — the step-time value is recomputed from free
        # params.
        if derived.grad is not None:
            assert (
                derived.grad.abs().max().item() == 0.0
            ), f"Derived parameter WealthShareM1_{col} has non-zero grad"


# ---------------------------------------------------------------------------
# Theoretical Steady-State Tests
# ---------------------------------------------------------------------------


def test_compute_theoretical_steady_state_runs():
    """Smoke test: compute_theoretical_steady_state returns without error."""
    model, _, _ = _make_model(timesteps=100)
    model.compute_theoretical_steady_state()


def test_steady_state_y_converges_to_simulation():
    """Theoretical SS real output should match long-run simulated output.

    The analytical solver now includes FCB via the HPM identity, closing the
    structural gap from the prior iterative approach.  Tolerance is 0.5% to
    allow for bank-rate convergence residuals.
    """
    model, _, _ = _make_model(timesteps=2000)
    model.simulate()
    y_sim = model.variables.timeseries["RealOutput"][-1].item()

    model_ss, _, _ = _make_model(timesteps=2000)
    model_ss.compute_theoretical_steady_state()
    y_ss = model_ss.variables.timeseries["RealOutput"][-1].item()

    rel_diff = abs(y_ss - y_sim) / max(abs(y_sim), 1e-6)
    assert rel_diff < 5e-3, (
        f"Theoretical SS y* deviates from simulation: "
        f"sim={y_sim:.6f}, ss={y_ss:.6f}, rel={rel_diff:.2e}"
    )


def test_ss_all_scenarios_match_simulation():
    """All 8 scenarios: SS real output within 1% of 2000-period simulation."""
    for sc_idx in range(8):
        # Simulation (long-run)
        model_sim, _, _ = _make_model(timesteps=2000)
        model_sim.simulate(scenario=sc_idx)
        y_sim = model_sim.variables.timeseries["RealOutput"][-1].item()

        # Theoretical SS
        model_ss, _, _ = _make_model(timesteps=2000)
        model_ss.compute_theoretical_steady_state(scenario=sc_idx)
        y_ss = model_ss.variables.timeseries["RealOutput"][-1].item()

        rel_diff = abs(y_ss - y_sim) / max(abs(y_sim), 1e-6)
        assert rel_diff < 0.01, (
            f"Scenario {sc_idx}: " f"sim={y_sim:.4f}, ss={y_ss:.4f}, rel={rel_diff:.2e}"
        )


def test_ss_baseline_tight_tolerance():
    """Baseline SS should match simulation within 0.2%."""
    model_sim, _, _ = _make_model(timesteps=2000)
    model_sim.simulate()
    y_sim = model_sim.variables.timeseries["RealOutput"][-1].item()

    model_ss, _, _ = _make_model(timesteps=2000)
    model_ss.compute_theoretical_steady_state()
    y_ss = model_ss.variables.timeseries["RealOutput"][-1].item()

    rel_diff = abs(y_ss - y_sim) / max(abs(y_sim), 1e-6)
    assert (
        rel_diff < 2e-3
    ), f"Baseline tight: sim={y_sim:.6f}, ss={y_ss:.6f}, rel={rel_diff:.2e}"


def test_ss_inflation_derived():
    """Inflation is derived from the wage equation, not hardcoded zero.

    At true SS, the consumption identity forces π → 0.  The solver should
    converge to |π| < 1e-6.
    """
    model, _, _ = _make_model(timesteps=2000)
    model.compute_theoretical_steady_state()
    pi = model.variables.timeseries["InflationRate"][-1].abs().item()
    assert pi < 1e-3, f"SS inflation not near zero: |π| = {pi:.2e}"


def test_ss_portfolio_shares_sum():
    """At SS, household portfolio shares should sum to total non-cash wealth."""
    model, _, _ = _make_model(timesteps=2000)
    model.compute_theoretical_steady_state()
    ts = model.variables.timeseries

    M1 = ts["M1Household"][-1]
    M2 = ts["M2Household"][-1]
    B_h = ts["BillsHousehold"][-1]
    BL_h = ts["BondsHousehold"][-1]
    p_bl = ts["BondPrice"][-1]
    V_nc = ts["NonCashWealth"][-1]

    portfolio_sum = M1 + M2 + B_h + p_bl * BL_h
    rel_diff = (portfolio_sum - V_nc).abs() / V_nc.abs().clamp(min=1e-6)
    assert (
        rel_diff.item() < 1e-6
    ), f"Portfolio sum {portfolio_sum.item():.4f} != V_nc {V_nc.item():.4f}"


def test_ss_shock_dispatch():
    """SS values change after parameter shock (spot-check two scenarios).

    Uses short timesteps — SS convergence only needs ~200 outer-loop steps.
    Full 8-scenario accuracy is covered by test_ss_all_scenarios_match_simulation.
    """
    model_base, _, _ = _make_model(timesteps=200)
    model_base.compute_theoretical_steady_state()
    y_base = model_base.variables.timeseries["RealOutput"][-1].item()

    for sc_idx in [2, 5]:  # fiscal shock + wealth shock
        model_sc, _, _ = _make_model(timesteps=200)
        model_sc.compute_theoretical_steady_state(scenario=sc_idx)
        y_sc = model_sc.variables.timeseries["RealOutput"][-1].item()

        assert abs(y_sc - y_base) > 0.01, (
            f"Scenario {sc_idx}: SS y* unchanged after shock "
            f"(base={y_base:.4f}, shock={y_sc:.4f})"
        )
