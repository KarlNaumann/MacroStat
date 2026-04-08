"""Targeted tests for the NK3E model.

These tests avoid optional dependencies by running only the NK3E pieces and
assert on closed-form one-step implications of the 3-equation system.

Timing convention (base class Behavior.forward with timesteps_initialization=1):
  indices 0-1: initialization (steady state recorded at t=0, t=1)
  index 2:     first simulated step (t=1 in sim loop)
  index 3:     second simulated step (t=2 in sim loop)
"""

import torch

from macrostat.models.NK3E import (
    NK3E,
    ParametersNK3E,
    ScenariosNK3E,
    VariablesNK3E,
)


def _make_model(timesteps: int = 3):
    params = ParametersNK3E(
        hyperparameters={
            "timesteps": timesteps,
            "timesteps_initialization": 1,
            "use_tqdm": False,
        }
    )
    variables = VariablesNK3E(parameters=params)
    scenarios = ScenariosNK3E(parameters=params)
    model = NK3E(parameters=params, variables=variables, scenarios=scenarios)
    return model, params, scenarios


def test_baseline_steady_state_holds_one_step():
    model, params, _ = _make_model()

    model.simulate()
    ts = model.variables.timeseries

    a1 = params["a1"]
    A = params["A"]
    y_e = params["y_e"]
    pi_T = params["pi_T"]
    r_s = (A - y_e) / a1

    # First simulated step (index 2) should preserve steady state with no shocks
    assert torch.isclose(ts["y"][2, 0], torch.tensor(y_e), atol=1e-6)
    assert torch.isclose(ts["pi"][2, 0], torch.tensor(pi_T), atol=1e-6)
    assert torch.isclose(ts["r"][2, 0], torch.tensor(r_s), atol=1e-6)


def test_scenario1_increase_A_raises_y_and_r_first_step():
    model, params, scenarios = _make_model()
    sc1 = scenarios.get_scenario_index("Scenario.1: Rise in A")

    model.simulate(scenario=sc1)
    ts = model.variables.timeseries

    a1 = params["a1"]
    a2 = params["a2"]
    b = params["b"]
    A = params["A"] + 2.0  # shock
    y_e = params["y_e"]
    pi_T = params["pi_T"]
    r_s = (A - y_e) / a1
    a3 = 1.0 / (a1 * (1.0 / (a2 * b) + a2))

    # First simulated step (index 2) from steady state with r_{t-1}=r_s_baseline
    r_prev = (params["A"] - params["y_e"]) / a1
    y_t = A - a1 * r_prev
    pi_t = pi_T + a2 * (y_t - y_e)
    r_t = r_s + a3 * (pi_t - pi_T)

    assert y_t > y_e
    assert r_t >= r_prev
    assert torch.isclose(ts["y"][2, 0], torch.tensor(y_t), atol=1e-6)
    assert torch.isclose(ts["r"][2, 0], torch.tensor(r_t), atol=1e-6)


def test_scenario2_higher_piT_raises_r_first_step():
    model, params, scenarios = _make_model()
    sc2 = scenarios.get_scenario_index("Scenario.2: Higher pi_T")

    model.simulate(scenario=sc2)
    ts = model.variables.timeseries

    a1 = params["a1"]
    a2 = params["a2"]
    b = params["b"]
    A = params["A"]
    y_e = params["y_e"]
    pi_T = params["pi_T"] + 1.0
    r_s = (A - y_e) / a1
    a3 = 1.0 / (a1 * (1.0 / (a2 * b) + a2))

    # First simulated step (index 2) from steady state
    r_prev = (params["A"] - params["y_e"]) / a1
    y_t = A - a1 * r_prev  # equals y_e
    pi_t = params["pi_T"] + a2 * (y_t - y_e)  # equals pi_T_baseline
    r_t = r_s + a3 * (pi_t - pi_T)  # pi_t - higher target is negative -> r_t < r_s

    assert r_t <= r_prev
    assert torch.isclose(ts["r"][2, 0], torch.tensor(r_t), atol=1e-6)


def test_initialize_preserves_autograd_graph():
    """Verify that initialize() keeps the computation graph from parameters to state.

    If initialize() detaches from the graph (e.g. via torch.tensor()), the
    autograd Jacobian at initialization timesteps will be zero and diverge
    from the correct numerical Jacobian for all subsequent timesteps.
    """
    model, params, _ = _make_model()
    behavior = model.get_model_training_instance()
    behavior.forward()

    ts = model.variables.timeseries
    # First simulated step (index 2): r depends on a1 via r_s = (A - y_e)/a1,
    # so dr/da1 = -(A - y_e)/a1^2 which is non-zero.
    # (y = y_e at steady state regardless of a1, so dy/da1 = 0 by cancellation)
    r_1 = ts["r"][2, 0]
    r_1.backward()

    # Expected: dr/da1 = -(A - y_e) / a1^2
    a1_val = params["a1"]
    A_val = params["A"]
    y_e_val = params["y_e"]
    expected_grad = -(A_val - y_e_val) / (a1_val**2)

    grad_a1 = behavior.params["a1"].grad
    assert grad_a1 is not None, "No gradient computed for a1"
    assert grad_a1.abs() > 1e-10, f"Gradient is effectively zero: {grad_a1.item()}"
    assert torch.isclose(
        grad_a1, torch.tensor(expected_grad), rtol=1e-4
    ), f"Gradient {grad_a1.item():.4f} != expected {expected_grad:.4f}"
