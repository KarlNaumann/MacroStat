"""Targeted tests for the GL06LP2 model.

These tests verify the Godley-Lavoie 2006 LP2 model (Endogenous Bond
Price via Target-Proportion Mechanism with Adaptive Expectations)
including smoke tests, positivity checks, redundant equation checks,
bond-price mechanism tests, and shock response tests.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import torch

from macrostat.models.GL06LP2 import (
    GL06LP2,
    ParametersGL06LP2,
    ScenariosGL06LP2,
    VariablesGL06LP2,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(timesteps: int = 50, init_periods: int = 1):
    """Create a default GL06LP2 model with given timesteps."""
    params = ParametersGL06LP2(
        hyperparameters={
            "timesteps": timesteps,
            "timesteps_initialization": init_periods,
            "use_tqdm": False,
        }
    )
    variables = VariablesGL06LP2(parameters=params)
    scenarios = ScenariosGL06LP2(parameters=params)
    model = GL06LP2(parameters=params, variables=variables, scenarios=scenarios)
    return model, params, scenarios


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------


def test_baseline_runs_without_error():
    """Smoke test: baseline simulation completes without error."""
    model, _, _ = _make_model()
    model.simulate()


def test_scenario1_runs_without_error():
    """Smoke test: scenario 1 (bill rate rise) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.1: Rise in bill rate")
    model.simulate(scenario=sc)


def test_scenario2_runs_without_error():
    """Smoke test: scenario 2 (expected bond price fall) completes."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.2: Expected bond price fall")
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

    hh = ts["HouseholdCashStock"][1:]
    hs = ts["CentralBankMoneyStock"][1:]
    diff = (hh - hs).abs()

    assert torch.all(
        diff < 0.01
    ), f"Redundant equation violated: max |H_h - H_s| = {diff.max().item()}"


# ---------------------------------------------------------------------------
# Bond Price Mechanism Tests
# ---------------------------------------------------------------------------


def test_target_proportion_within_bounds():
    """In baseline, target proportion should settle within [bot, top]."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    tp = ts["TargetProportion"][-1]
    assert (
        0.47 <= tp.item() <= 0.52
    ), f"TargetProportion outside [bot, top]: {tp.item()}"


def test_bond_price_stays_positive():
    """Bond price should remain positive throughout the simulation."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    pbl = ts["BondPrice"][1:]
    assert torch.all(pbl > 0), f"BondPrice went non-positive: min={pbl.min().item()}"


def test_bond_price_near_initial():
    """In baseline, bond price should not deviate wildly from initial."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    pbl_last = ts["BondPrice"][-1]
    # Default initial is 20; should stay within a reasonable range
    assert 15 < pbl_last.item() < 25, f"BondPrice drifted too far: {pbl_last.item()}"


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
        "BondPrice",
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


def test_higher_bill_rate_raises_income():
    """A higher bill rate should increase steady-state national income."""
    model_base, _, _ = _make_model(timesteps=100)
    model_base.simulate()
    y_base = model_base.variables.timeseries["NationalIncome"][-1]

    model_shock, _, scenarios = _make_model(timesteps=100)
    sc = scenarios.get_scenario_index("Scenario.1: Rise in bill rate")
    model_shock.simulate(scenario=sc)
    y_shock = model_shock.variables.timeseries["NationalIncome"][-1]

    assert y_shock > y_base, (
        f"Higher rb should raise Y: Y_base={y_base.item()}, "
        f"Y_shock={y_shock.item()}"
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
        diff < 0.1
    ), f"Wealth identity violated: max diff = {diff.max().item()}"


def test_bond_yield_equals_inverse_price():
    """Bond yield should equal 1/pbl."""
    model, _, _ = _make_model(timesteps=50)
    model.simulate()
    ts = model.variables.timeseries

    pbl = ts["BondPrice"][2:].flatten()
    rbl = ts["BondYield"][2:].flatten()

    mask = pbl > 0
    expected = 1.0 / pbl[mask]
    actual = rbl[mask]

    assert torch.allclose(actual, expected, atol=1e-6), (
        f"Bond yield != 1/pbl: max diff = " f"{(actual - expected).abs().max().item()}"
    )
