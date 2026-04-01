"""Targeted tests for the GL06LP model.

These tests verify the Godley-Lavoie 2006 LP model (Long-term Bonds,
Capital Gains, and Liquidity Preference) including smoke tests,
positivity checks, redundant equation checks, and shock response tests.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import torch

from macrostat.models.GL06LP import (
    GL06LP,
    ParametersGL06LP,
    ScenariosGL06LP,
    VariablesGL06LP,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(timesteps: int = 50, init_periods: int = 1):
    """Create a default GL06LP model with given timesteps."""
    params = ParametersGL06LP(
        hyperparameters={
            "timesteps": timesteps,
            "timesteps_initialization": init_periods,
            "use_tqdm": False,
        }
    )
    variables = VariablesGL06LP(parameters=params)
    scenarios = ScenariosGL06LP(parameters=params)
    model = GL06LP(parameters=params, variables=variables, scenarios=scenarios)
    return model, params, scenarios


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------


def test_baseline_runs_without_error():
    """Smoke test: baseline simulation completes without error."""
    model, _, _ = _make_model()
    model.simulate()


def test_scenario1_runs_without_error():
    """Smoke test: scenario 1 (interest rates rise) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.1: Rise in interest rates")
    model.simulate(scenario=sc)


def test_scenario2_runs_without_error():
    """Smoke test: scenario 2 (drop in alpha1) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.2: Drop in alpha1")
    model.simulate(scenario=sc)


# ---------------------------------------------------------------------------
# Positivity Checks
# ---------------------------------------------------------------------------


def test_baseline_stocks_nonnegative():
    """After baseline simulation, all stocks should be non-negative."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    stock_vars = [
        "Wealth",
        "HouseholdBillStock",
        "GovernmentBillStock",
        "CentralBankBillStock",
        "HouseholdBondStock",
        "GovernmentBondSupply",
        "HouseholdCashStock",
        "CentralBankMoneyStock",
    ]

    for var in stock_vars:
        # Skip initialization period (index 0)
        vals = ts[var][1:]
        assert torch.all(
            vals >= -1e-6
        ), f"{var} has negative values: min={vals.min().item()}"


def test_baseline_flows_nonnegative():
    """After baseline simulation, key flows should be non-negative."""
    model, _, _ = _make_model()
    model.simulate()
    ts = model.variables.timeseries

    flow_vars = [
        "ConsumptionHousehold",
        "ConsumptionGovernment",
        "NationalIncome",
        "Taxes",
        "DisposableIncome",
    ]

    for var in flow_vars:
        vals = ts[var][1:]
        assert torch.all(
            vals >= -1e-6
        ), f"{var} has negative values: min={vals.min().item()}"


# ---------------------------------------------------------------------------
# Redundant Equation Checks
# ---------------------------------------------------------------------------


def test_redundant_equation_baseline():
    """The redundant equation H_h = H_s should hold within tolerance."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    hh = ts["HouseholdCashStock"][1:]  # skip init
    hs = ts["CentralBankMoneyStock"][1:]
    diff = (hh - hs).abs()

    assert torch.all(
        diff < 0.01
    ), f"Redundant equation violated: max |H_h - H_s| = {diff.max().item()}"


def test_redundant_equation_scenario1():
    """Redundant equation holds under scenario 1 (bill rate rise)."""
    model, _, scenarios = _make_model(timesteps=100)
    sc = scenarios.get_scenario_index("Scenario.1: Rise in interest rates")
    model.simulate(scenario=sc)
    ts = model.variables.timeseries

    hh = ts["HouseholdCashStock"][1:]
    hs = ts["CentralBankMoneyStock"][1:]
    diff = (hh - hs).abs()

    assert torch.all(
        diff < 0.01
    ), f"Redundant equation violated: max |H_h - H_s| = {diff.max().item()}"


# ---------------------------------------------------------------------------
# Steady State Tests
# ---------------------------------------------------------------------------


def test_steady_state_consumption_equals_disposable_income():
    """In steady state, consumption should equal disposable income."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    c_last = ts["ConsumptionHousehold"][-1]
    yd_last = ts["DisposableIncome"][-1]

    assert torch.isclose(
        c_last, yd_last, atol=1e-3
    ), f"Steady state violation: C={c_last.item()}, YDr={yd_last.item()}"


def test_steady_state_income_identity():
    """In steady state, Y = C + G should hold exactly."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    y = ts["NationalIncome"][-1]
    c = ts["ConsumptionHousehold"][-1]
    g = ts["ConsumptionGovernment"][-1]

    assert torch.isclose(
        y, c + g, atol=1e-6
    ), f"Income identity violated: Y={y.item()}, C+G={(c+g).item()}"


def test_steady_state_capital_gains_zero():
    """In baseline (constant pbl), capital gains should be zero in steady state."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    cg = ts["CapitalGains"][-1]
    assert torch.isclose(
        cg, torch.zeros(1), atol=1e-8
    ), f"Capital gains should be zero: CG={cg.item()}"


def test_steady_state_convergence():
    """Key variables should converge (last two values nearly equal)."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    vars_to_check = [
        "NationalIncome",
        "Wealth",
        "ConsumptionHousehold",
        "HouseholdBillStock",
        "HouseholdBondStock",
    ]

    for var in vars_to_check:
        v_last = ts[var][-1]
        v_prev = ts[var][-2]
        assert torch.isclose(
            v_last, v_prev, atol=1e-4
        ), f"{var} not converged: {v_prev.item()} -> {v_last.item()}"


# ---------------------------------------------------------------------------
# Shock Response Tests
# ---------------------------------------------------------------------------


def test_higher_interest_rates_raises_income():
    """A combined interest rate increase should raise steady-state Y."""
    model_base, _, _ = _make_model(timesteps=100)
    model_base.simulate()
    y_base = model_base.variables.timeseries["NationalIncome"][-1]

    model_shock, _, scenarios = _make_model(timesteps=100)
    sc = scenarios.get_scenario_index("Scenario.1: Rise in interest rates")
    model_shock.simulate(scenario=sc)
    y_shock = model_shock.variables.timeseries["NationalIncome"][-1]

    assert y_shock > y_base, (
        f"Higher rb should raise Y: Y_base={y_base.item()}, "
        f"Y_shock={y_shock.item()}"
    )


def test_lower_alpha1_raises_income_long_run():
    """With exogenous G, a drop in α1 should raise long-run Y via higher debt service."""
    model_base, _, _ = _make_model(timesteps=200)
    model_base.simulate()
    y_base = model_base.variables.timeseries["NationalIncome"][-1]

    model_shock, _, scenarios = _make_model(timesteps=200)
    sc = scenarios.get_scenario_index("Scenario.2: Drop in alpha1")
    model_shock.simulate(scenario=sc)
    y_shock = model_shock.variables.timeseries["NationalIncome"][-1]

    assert y_shock > y_base, (
        f"Lower alpha1 should raise long-run Y (paradox of thrift): "
        f"Y_base={y_base.item()}, Y_shock={y_shock.item()}"
    )


def test_higher_interest_rates_shifts_portfolio():
    """A combined interest rate increase should shift portfolio shares.

    With both rb rising (3%→4%) and pbl falling (20→15), the bond yield
    increases from 5% to 6.67%.  The portfolio allocation equations in LP
    weight both rates, and the larger absolute increase in bond yield means
    the bond share should rise while the bill share falls.
    """
    model_base, _, _ = _make_model(timesteps=100)
    model_base.simulate()
    ts_base = model_base.variables.timeseries
    blh_base = ts_base["HouseholdBondStock"][-1]
    pbl_base = ts_base["BondPrice"][-1]
    v_base = ts_base["Wealth"][-1]
    bond_share_base = (blh_base * pbl_base) / v_base

    model_shock, _, scenarios = _make_model(timesteps=100)
    sc = scenarios.get_scenario_index("Scenario.1: Rise in interest rates")
    model_shock.simulate(scenario=sc)
    ts_shock = model_shock.variables.timeseries
    blh_shock = ts_shock["HouseholdBondStock"][-1]
    pbl_shock = ts_shock["BondPrice"][-1]
    v_shock = ts_shock["Wealth"][-1]
    bond_share_shock = (blh_shock * pbl_shock) / v_shock

    assert bond_share_shock > bond_share_base, (
        f"Higher interest rates should increase bond share: "
        f"base={bond_share_base.item():.4f}, "
        f"shock={bond_share_shock.item():.4f}"
    )


def test_wealth_accounting_identity():
    """Verify V = H_h + B_h + pbl * BL_h at each step."""
    model, _, _ = _make_model(timesteps=50)
    model.simulate()
    ts = model.variables.timeseries

    v = ts["Wealth"][1:]
    hh = ts["HouseholdCashStock"][1:]
    bh = ts["HouseholdBillStock"][1:]
    blh = ts["HouseholdBondStock"][1:]
    pbl = ts["BondPrice"][1:]

    reconstructed = hh + bh + pbl * blh
    diff = (v - reconstructed).abs()

    assert torch.all(
        diff < 1e-4
    ), f"Wealth identity violated: max diff = {diff.max().item()}"


def test_bond_yield_equals_inverse_price():
    """Bond yield should equal 1/pbl."""
    model, _, _ = _make_model(timesteps=50)
    model.simulate()
    ts = model.variables.timeseries

    pbl = ts["BondPrice"][1:]
    rbl = ts["BondYield"][1:]

    # Only check where pbl > 0
    mask = pbl > 0
    expected = 1.0 / pbl[mask]
    actual = rbl[mask]

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"Bond yield != 1/pbl: max diff = " f"{(actual - expected).abs().max().item()}"
    )
