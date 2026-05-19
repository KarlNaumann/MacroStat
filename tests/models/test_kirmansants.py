# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
"""Targeted tests for the KirmansAnts SDE model.

The model integrates the large-N continuous-limit Kirman SDE via the
Lamperti transform :math:`\\phi = \\arcsin(2 x - 1)` (Moran et al. 2020),
which yields constant diffusion :math:`\\sqrt{2\\mu}` and recovers
:math:`O(dt)` weak convergence. Tests exercise construction validation, the
differentiability gate, reproducibility under fixed seed, the Lamperti map
roundtrip and analytical helpers, and the stationary distribution match
against an analytical Beta sample via ``scipy.stats.ks_2samp`` /
``scipy.stats.ks_1samp``.
"""

import math

import numpy as np
import pytest
from scipy import stats

from macrostat.models.KirmansAnts import (
    BehaviorKirmansAnts,
    KirmansAnts,
    ParametersKirmansAnts,
    ScenariosKirmansAnts,
    VariablesKirmansAnts,
)


def _make_model(timesteps: int = 10, record_inner: bool = False, **hyper):
    extra = {"timesteps": timesteps, "record_inner": record_inner}
    extra.update(hyper)
    parameters = ParametersKirmansAnts(hyperparameters=extra)
    return KirmansAnts(parameters=parameters)


def _stationary_sample(scenario_id: int, timesteps: int = 100):
    """Run a single trajectory and return a thinned float64 stationary sample.

    Thinning at stride 1000 keeps the autocorrelation of the returned sample
    low enough that ``scipy.stats.ks_2samp`` against an analytical Beta draw
    is calibrated; thinner strides at this budget leave residual serial
    correlation that inflates the KS statistic.
    """
    model = _make_model(timesteps=timesteps, record_inner=True)
    model.simulate(scenario=scenario_id)
    micro = model.behavior_instance._micro_trajectory.astype(np.float64)
    burn_in = micro.size // 10
    return micro[burn_in::1000], model


# --- smoke / shape ---------------------------------------------------------


def test_smoke_all_scenarios():
    model = _make_model(timesteps=5)
    for scenario_id in (0, 1, 2):
        out = model.simulate(scenario=scenario_id)
        assert out["density"].shape == (6, 1)


def test_density_in_unit_interval():
    model = _make_model(timesteps=20, record_inner=True)
    model.simulate()
    micro = model.behavior_instance._micro_trajectory
    assert (micro > 0.0).all()
    assert (micro < 1.0).all()


# --- stationary distribution -----------------------------------------------


def test_stationary_unimodal_ks():
    """Beta(2, 2): Lamperti recovers the analytical distribution."""
    sample, _ = _stationary_sample(scenario_id=2)
    reference = stats.beta(2.0, 2.0).rvs(size=sample.size, random_state=20260511)
    pvalue = stats.ks_2samp(sample, reference).pvalue
    assert pvalue > 0.01, f"KS pvalue {pvalue:.4f} too small"


def test_stationary_bimodal_ks():
    """Beta(0.5, 0.5): Lamperti removes the boundary bias seen under rejection."""
    sample, _ = _stationary_sample(scenario_id=0)
    reference = stats.beta(0.5, 0.5).rvs(size=sample.size, random_state=20260511)
    pvalue = stats.ks_2samp(sample, reference).pvalue
    assert pvalue > 0.01, f"KS pvalue {pvalue:.4f} too small"


def test_stationary_uniform_ks():
    """Beta(1, 1): the uniform regime."""
    sample, _ = _stationary_sample(scenario_id=1)
    reference = stats.beta(1.0, 1.0).rvs(size=sample.size, random_state=20260511)
    pvalue = stats.ks_2samp(sample, reference).pvalue
    assert pvalue > 0.01, f"KS pvalue {pvalue:.4f} too small"


@pytest.mark.slow
@pytest.mark.parametrize(
    "scenario_id,shape",
    [(0, 0.5), (1, 1.0), (2, 2.0)],
    ids=["bimodal", "uniform", "unimodal"],
)
def test_stationary_paper_budget(scenario_id, shape):
    """KS-1samp against analytical Beta CDF at paper-figure budget.

    Mirrors the regime panels in ``paper/figures/fig_ant_dynamics.pdf``.
    ``timesteps=1000`` plus ``substeps=10_000`` gives the same trajectory
    length the figure uses; thinning at stride 1000 produces ~10k
    near-independent samples. Skipped by default (``pytest -m slow`` to
    enable) because each scenario is ~minutes of wall-clock.
    """
    sample, _ = _stationary_sample(scenario_id=scenario_id, timesteps=1000)
    cdf = stats.beta(shape, shape).cdf
    pvalue = stats.ks_1samp(sample, cdf).pvalue
    assert pvalue > 0.01, f"scenario={scenario_id} KS pvalue {pvalue:.4f} too small"


# --- reproducibility -------------------------------------------------------


def test_reproducibility():
    a = _make_model(timesteps=10, record_inner=True)
    b = _make_model(timesteps=10, record_inner=True)
    a.simulate()
    b.simulate()
    np.testing.assert_array_equal(
        a.behavior_instance._micro_trajectory,
        b.behavior_instance._micro_trajectory,
    )


# --- Lamperti map ---------------------------------------------------------


@pytest.mark.parametrize(
    "x", [1.0e-12, 1.0e-6, 0.25, 0.5, 0.75, 1.0 - 1.0e-6, 1.0 - 1.0e-12]
)
def test_lamperti_map_stable_near_boundaries(x):
    phi = BehaviorKirmansAnts._lamperti_forward(x)
    assert math.isfinite(phi)
    assert -0.5 * math.pi <= phi <= 0.5 * math.pi
    x_back = BehaviorKirmansAnts._lamperti_inverse(phi)
    assert math.isfinite(x_back)
    assert math.isclose(x_back, x, rel_tol=1.0e-9, abs_tol=1.0e-14)


@pytest.mark.parametrize("mu", [0.5, 1.0, 2.5])
@pytest.mark.parametrize("rho", [0.1, 0.5, 1.0, 2.0])
def test_lamperti_drift_matches_finite_difference(rho, mu):
    """Confirm the analytical Lamperti drift equals the Itô-derived FD form."""
    h = 1.0e-5
    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        f_plus = BehaviorKirmansAnts._lamperti_forward(x + h)
        f_minus = BehaviorKirmansAnts._lamperti_forward(x - h)
        f_center = BehaviorKirmansAnts._lamperti_forward(x)
        fp = (f_plus - f_minus) / (2.0 * h)
        fpp = (f_plus - 2.0 * f_center + f_minus) / (h * h)
        drift_x = rho * (1.0 - 2.0 * x)
        sigma_sq = 2.0 * mu * x * (1.0 - x)
        expected = drift_x * fp + 0.5 * sigma_sq * fpp
        got = BehaviorKirmansAnts._lamperti_drift(x, rho, mu)
        assert math.isclose(got, expected, rel_tol=1.0e-4, abs_tol=1.0e-4)


def test_lamperti_drift_closed_form_in_phi():
    """In phi-space the drift simplifies to -(2 rho - mu) * tan(phi)."""
    rho, mu = 0.5, 1.0
    for x in (0.1, 0.3, 0.5, 0.7, 0.9):
        phi = BehaviorKirmansAnts._lamperti_forward(x)
        expected = -(2.0 * rho - mu) * math.tan(phi)
        got = BehaviorKirmansAnts._lamperti_drift(x, rho, mu)
        assert math.isclose(got, expected, rel_tol=1.0e-9, abs_tol=1.0e-12)


# --- drift symmetry -------------------------------------------------------


def test_drift_symmetry_stationary_mean():
    """Symmetric Beta has mean 1/2; empirical mean stays in tolerance."""
    sample, _ = _stationary_sample(scenario_id=2, timesteps=200)
    empirical_mean = float(np.mean(sample))
    assert (
        abs(empirical_mean - 0.5) < 0.05
    ), f"mean {empirical_mean:.4f} too far from 0.5"


# --- legacy validation -----------------------------------------------------


def test_differentiable_blocked_at_construction():
    parameters = ParametersKirmansAnts()
    scenarios = ScenariosKirmansAnts(parameters=parameters)
    variables = VariablesKirmansAnts(parameters=parameters)
    with pytest.raises(RuntimeError, match="supports_differentiable=False"):
        BehaviorKirmansAnts(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            differentiable=True,
        )


def test_dt_validation_rejects_non_integer_inverse():
    with pytest.raises(ValueError, match="1/dt must be integer-valued"):
        ParametersKirmansAnts(hyperparameters={"dt": 0.123, "substeps": 8})


def test_dt_validation_rejects_inconsistent_substeps():
    with pytest.raises(ValueError, match="substeps must equal"):
        ParametersKirmansAnts(hyperparameters={"dt": 1e-4, "substeps": 5000})


def test_x0_validation_rejects_boundary():
    with pytest.raises(ValueError, match="x0 must be strictly in"):
        ParametersKirmansAnts(hyperparameters={"x0": 0.0})


def test_record_inner_buffer_shape():
    model = _make_model(timesteps=7, record_inner=True)
    model.simulate()
    expected = 7 * model.parameters.hyper["substeps"]
    assert model.behavior_instance._micro_trajectory.shape == (expected,)
    assert model.behavior_instance._micro_trajectory.dtype == np.float32


def test_record_inner_off_leaves_buffer_none():
    model = _make_model(timesteps=5, record_inner=False)
    model.simulate()
    assert model.behavior_instance._micro_trajectory is None
