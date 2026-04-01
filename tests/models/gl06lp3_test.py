"""Targeted tests for the GL06LP3 model.

These tests verify the Godley-Lavoie 2006 LP3 model (Endogenous
Government Spending via Fiscal Rule) including smoke tests,
positivity checks, redundant equation checks, fiscal rule tests,
and shock response tests.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import torch

from macrostat.models.GL06LP3 import (
    GL06LP3,
    ParametersGL06LP3,
    ScenariosGL06LP3,
    VariablesGL06LP3,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_model(timesteps: int = 50, init_periods: int = 1):
    """Create a default GL06LP3 model with given timesteps."""
    params = ParametersGL06LP3(
        hyperparameters={
            "timesteps": timesteps,
            "timesteps_initialization": init_periods,
            "use_tqdm": False,
        }
    )
    variables = VariablesGL06LP3(parameters=params)
    scenarios = ScenariosGL06LP3(parameters=params)
    model = GL06LP3(parameters=params, variables=variables, scenarios=scenarios)
    return model, params, scenarios


# ---------------------------------------------------------------------------
# Smoke Tests
# ---------------------------------------------------------------------------


def test_baseline_runs_without_error():
    """Smoke test: baseline simulation completes without error."""
    model, _, _ = _make_model()
    model.simulate()


def test_scenario1_runs_without_error():
    """Smoke test: scenario 1 (drop in alpha1) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.1: Drop in alpha1")
    model.simulate(scenario=sc)


def test_scenario2_runs_without_error():
    """Smoke test: scenario 2 (bill rate rise) completes without error."""
    model, _, scenarios = _make_model()
    sc = scenarios.get_scenario_index("Scenario.2: Rise in bill rate")
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
# Fiscal Rule Tests
# ---------------------------------------------------------------------------


def test_government_spending_adjusts():
    """G should differ from the initial value due to fiscal rule."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    g_last = ts["ConsumptionGovernment"][-1].item()
    # Initial G is 20; fiscal rule should adjust it
    assert g_last != 20.0, f"G should have been adjusted by fiscal rule: {g_last}"


def test_psbr_converges_near_zero():
    """In steady state baseline, PSBR should converge near zero."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    psbr_last = ts["PublicSectorBorrowingRequirement"][-1]
    assert (
        abs(psbr_last.item()) < 0.1
    ), f"PSBR should converge near zero: {psbr_last.item()}"


def test_psbr_positive_initially():
    """PSBR should be positive in early periods (government runs deficit)."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    # Check a few early periods (skip init at 0, check periods 2-5)
    psbr_early = ts["PublicSectorBorrowingRequirement"][2:6]
    assert torch.any(psbr_early > 0), f"PSBR should be positive early on: {psbr_early}"


# ---------------------------------------------------------------------------
# Bond Price Mechanism Tests (inherited from LP2)
# ---------------------------------------------------------------------------


def test_bond_price_stays_positive():
    """Bond price should remain positive throughout the simulation."""
    model, _, _ = _make_model(timesteps=100)
    model.simulate()
    ts = model.variables.timeseries

    pbl = ts["BondPrice"][1:]
    assert torch.all(pbl > 0), f"BondPrice went non-positive: min={pbl.min().item()}"


# ---------------------------------------------------------------------------
# Steady State Tests
# ---------------------------------------------------------------------------


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
        "ConsumptionGovernment",
    ]

    for var in vars_to_check:
        v_last = ts[var][-1]
        v_prev = ts[var][-2]
        assert torch.isclose(
            v_last, v_prev, atol=1e-3
        ), f"{var} not converged: {v_prev.item()} -> {v_last.item()}"


# ---------------------------------------------------------------------------
# Shock Response Tests
# ---------------------------------------------------------------------------


def test_drop_alpha1_changes_government_spending():
    """A drop in propensity to consume should trigger fiscal adjustment."""
    model_base, _, _ = _make_model(timesteps=100)
    model_base.simulate()
    g_base = model_base.variables.timeseries["ConsumptionGovernment"][-1]

    model_shock, _, scenarios = _make_model(timesteps=100)
    sc = scenarios.get_scenario_index("Scenario.1: Drop in alpha1")
    model_shock.simulate(scenario=sc)
    g_shock = model_shock.variables.timeseries["ConsumptionGovernment"][-1]

    # The fiscal rule should cause G to differ
    assert not torch.isclose(
        g_base, g_shock, atol=0.01
    ), f"G should differ: G_base={g_base.item()}, G_shock={g_shock.item()}"


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
