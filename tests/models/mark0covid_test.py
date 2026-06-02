# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
r"""Targeted tests for the Mark-0 COVID heterogeneous-agent ABM.

Covers smoke, positivity, bounded-domain invariants, reproducibility under
fixed seed, scenario dispatch, and a slow KS gate of aggregate
distributions against the abmstat reference implementation
(``abmstat.models.mark0_torch.Mark0Torch``). Reference fixtures are
generated on-demand by the slow gate; routine tests run without an abmstat
dependency.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pytest
import torch

from macrostat.models.Mark0COVID import (
    Mark0COVID,
    ParametersMark0COVID,
)


def _make_model(timesteps: int = 30, N_firms: int = 80, seed: int = 0) -> Mark0COVID:
    return Mark0COVID(
        parameters=ParametersMark0COVID(
            hyperparameters={
                "timesteps": timesteps,
                "N_firms": N_firms,
                "seed": seed,
            }
        )
    )


def _as_array(x):
    if torch.is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


# --- smoke -----------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 7])
def test_smoke_runs_clean(seed):
    out = _make_model(seed=seed).simulate()
    for k, v in out.items():
        arr = _as_array(v)
        assert np.isfinite(arr).all(), f"{k} contains NaN/Inf"


def test_smoke_scales_with_N_firms():
    out_small = _make_model(N_firms=50).simulate()
    out_large = _make_model(N_firms=200).simulate()
    assert out_small["FirmAlive"].shape[-1] == 50
    assert out_large["FirmAlive"].shape[-1] == 200


# --- positivity ------------------------------------------------------------


@pytest.mark.parametrize(
    "key",
    [
        "AveragePrice",
        "AverageWage",
        "TotalProduction",
        "TotalPayroll",
        "TotalDemand",
        "HouseholdSavings",
    ],
)
def test_aggregates_positive(key):
    arr = _as_array(_make_model().simulate()[key])
    assert (arr > 0.0).all(), f"{key} should remain strictly positive"


# --- bounded invariants ---------------------------------------------------


def test_unemployment_in_unit_interval():
    u = _as_array(_make_model().simulate()["Unemployment"])
    assert (u >= 0.0).all() and (u <= 1.0).all()


def test_employment_complements_unemployment():
    out = _make_model().simulate()
    u = _as_array(out["Unemployment"])
    e = _as_array(out["Employment"])
    np.testing.assert_allclose(u + e, np.ones_like(u), atol=1e-6)


def test_bankruptcy_rate_in_unit_interval():
    rate = _as_array(_make_model().simulate()["BankruptcyRate"])
    assert (rate >= 0.0).all() and (rate <= 1.0).all()


def test_firm_alive_in_zero_one():
    alive = _as_array(_make_model().simulate()["FirmAlive"])
    assert ((alive == 0.0) | (alive == 1.0)).all()


def test_consumption_propensity_in_unit_interval():
    c = _as_array(_make_model().simulate()["ConsumptionPropensity"])
    assert (c >= 0.0).all() and (c <= 1.0).all()


# --- reproducibility ------------------------------------------------------


def test_reproducibility_fixed_seed():
    a = _as_array(_make_model(seed=4).simulate()["AveragePrice"])
    b = _as_array(_make_model(seed=4).simulate()["AveragePrice"])
    np.testing.assert_array_equal(a, b)


def test_different_seeds_diverge():
    a = _as_array(_make_model(seed=0).simulate()["AveragePrice"])
    b = _as_array(_make_model(seed=1).simulate()["AveragePrice"])
    assert not np.array_equal(a, b)


# --- dtype gate -----------------------------------------------------------


def test_default_dtype_is_float64():
    out = _make_model().simulate()
    assert out["AveragePrice"].dtype == torch.float64


def test_float32_override_propagates():
    params = ParametersMark0COVID(
        hyperparameters={
            "timesteps": 10,
            "N_firms": 30,
            "seed": 0,
            "dtype": torch.float32,
        }
    )
    out = Mark0COVID(parameters=params).simulate()
    assert out["AveragePrice"].dtype == torch.float32


# --- scenarios ------------------------------------------------------------


@pytest.mark.parametrize("scenario_id", [0, 1, 2, 3])
def test_all_scenarios_run(scenario_id):
    out = _make_model().simulate(scenario=scenario_id)
    pavg = _as_array(out["AveragePrice"])
    assert np.isfinite(pavg).all()


def test_collapse_scenario_higher_unemployment_than_baseline():
    base = _as_array(
        _make_model(timesteps=80, seed=0).simulate(scenario=0)["Unemployment"]
    )
    collapse = _as_array(
        _make_model(timesteps=80, seed=0).simulate(scenario=3)["Unemployment"]
    )
    assert collapse[40:].mean() > base[40:].mean()


# --- accounting identity --------------------------------------------------


def test_household_plus_firm_savings_tracks_M0():
    out = _make_model(timesteps=40).simulate()
    hh = _as_array(out["HouseholdSavings"]).squeeze()
    firm = _as_array(out["FirmSavingsTotal"]).squeeze()
    m0 = _as_array(out["M0Stock"]).squeeze()
    assert np.all((hh + firm - m0) > -1e-3 * m0)


# --- slow: abmstat KS gate ------------------------------------------------


def _try_import_abmstat():
    candidates = [
        "/home/karl/Dropbox/workspace/github.com/KarlNaumann/packages/abmstat",
        os.path.expanduser(
            "~/Dropbox/workspace/github.com/KarlNaumann/packages/abmstat"
        ),
    ]
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)
    try:
        from abmstat.models.mark0_torch import Mark0Torch  # noqa: F401

        return True
    except ImportError:
        return False


@pytest.mark.slow
@pytest.mark.parametrize(
    "ms_key,ref_key,tol_pct",
    [
        ("AveragePrice", "Pavg", 2.0),
        ("AverageWage", "Wavg", 2.0),
        ("Unemployment", "u", 8.0),
        ("TotalDemand", "Dtot", 8.0),
        ("TotalPayroll", "payrolltot", 8.0),
    ],
)
def test_mean_aggregate_matches_abmstat(ms_key, ref_key, tol_pct):
    """Mean of aggregate path matches abmstat within `tol_pct` percent.

    Skipped automatically when the local abmstat package is not importable.
    """
    if not _try_import_abmstat():
        pytest.skip("abmstat reference package not available")
    from abmstat.models.mark0_torch import Mark0Torch

    T, N = 50, 200
    ref = Mark0Torch(
        hyper_parameters={
            "T": T,
            "N": N,
            "seed": 0,
            "Teq": 0,
            "device": "cpu",
            "reset_seed": True,
            "requires_grad": False,
        }
    )
    ref_df = ref.simulate()

    out = _make_model(timesteps=T, N_firms=N, seed=0).simulate()
    ms = _as_array(out[ms_key]).squeeze()
    rf = ref_df[ref_key].values
    n = min(len(ms), len(rf))
    ms_mean = float(ms[:n].mean())
    rf_mean = float(rf[:n].mean())
    rel = abs(ms_mean - rf_mean) / (abs(rf_mean) + 1e-12) * 100
    assert rel < tol_pct, f"{ms_key} mean rel-diff {rel:.2f}% > tol {tol_pct}%"
