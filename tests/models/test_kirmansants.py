# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
"""Targeted tests for the KirmansAnts SDE model.

The model is a numpy Euler-Maruyama clone of the abmstat reference
implementation (``packages/abmstat/abmstat/models/kirmanants.py``). Tests
exercise: construction validation, the differentiability gate, the
boundary-rejection guard, reproducibility under fixed seed, and the
stationary distribution match for the unimodal regime.
"""

import numpy as np
import pytest
from scipy.stats import kstest

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


def test_stationary_unimodal_ks():
    model = _make_model(timesteps=100, record_inner=True)
    model.simulate(scenario=2)  # unimodal: rho=2.0, mu=1.0 -> Beta(2, 2)
    micro = model.behavior_instance._micro_trajectory.astype(np.float64)
    burn_in = micro.size // 10
    sample = micro[burn_in::100]
    statistic, _ = kstest(sample, "beta", args=(2.0, 2.0))
    assert statistic < 0.05, f"KS statistic {statistic:.4f} too large"


@pytest.mark.xfail(
    reason=(
        "Boundary-rejection bias suppresses density near x in {0, 1}; the "
        "bimodal Beta(0.5, 0.5) diverges at the boundaries. Phase-2 dispatch "
        "addresses the bias."
    ),
    strict=False,
)
def test_stationary_bimodal_ks():
    model = _make_model(timesteps=100, record_inner=True)
    model.simulate(scenario=0)  # bimodal: rho=0.5, mu=1.0 -> Beta(0.5, 0.5)
    micro = model.behavior_instance._micro_trajectory.astype(np.float64)
    burn_in = micro.size // 10
    sample = micro[burn_in::100]
    statistic, _ = kstest(sample, "beta", args=(0.5, 0.5))
    assert statistic < 0.05


def test_reproducibility():
    model_a = _make_model(timesteps=10, record_inner=True)
    model_b = _make_model(timesteps=10, record_inner=True)
    model_a.simulate()
    model_b.simulate()
    np.testing.assert_array_equal(
        model_a.behavior_instance._micro_trajectory,
        model_b.behavior_instance._micro_trajectory,
    )


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


def test_no_exhaustion_at_defaults():
    model = _make_model(timesteps=20)
    model.simulate()
    assert model.behavior_instance.exhaustion_count == 0


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
