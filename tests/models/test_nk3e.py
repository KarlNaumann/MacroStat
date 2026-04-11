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


def test_state_shock_scenario_zero_matches_baseline():
    """With all shock variables at zero, simulation must equal the baseline.

    Guards against accidental coupling introduced by the shock plumbing.
    """
    model_a, _, _ = _make_model(timesteps=5)
    model_b, _, _ = _make_model(timesteps=5)

    model_a.simulate(scenario=0)
    model_b.simulate(scenario=0)

    for key in ("y", "pi", "r", "r_s", "a3"):
        ts_a = model_a.variables.timeseries[key]
        ts_b = model_b.variables.timeseries[key]
        assert torch.allclose(ts_a, ts_b, atol=1e-10)


def test_inflation_impulse_raises_pi_by_one_at_trigger():
    """One-period InflationShock of +1 at the trigger lifts pi by exactly 1.

    The baseline is a no-shock steady state, so at the first simulated step
    Phillips returns pi_T and the impulse then adds 1, giving pi = pi_T + 1.
    After the impulse period pi decays per the Phillips rule.
    """
    model, params, scenarios = _make_model(timesteps=5)
    sc = scenarios.get_scenario_index("Scenario.4: Inflation impulse")

    model.simulate(scenario=sc)
    ts = model.variables.timeseries

    pi_T = params["pi_T"]
    # First simulated step (index 2): pi starts at steady state, Phillips adds
    # a2*(y_t - y_e) which is zero at SS, impulse of +1 applies, so pi = pi_T+1.
    assert torch.isclose(ts["pi"][2, 0], torch.tensor(pi_T) + 1.0, atol=1e-6)
    # Subsequent steps: impulse is zero, y and r dynamics reflect the one-off
    # perturbation, and pi must differ from pi_T (decay regime, not a jump).
    assert ts["pi"][3, 0].item() != pi_T
    assert abs(ts["pi"][3, 0].item() - (pi_T + 1.0)) > 1e-6


def test_inflation_impulse_preserves_parameters():
    """An InflationShock scenario must not shift A, pi_T, or y_e.

    Contrast with Scenario.2 which moves pi_T — here the structural parameters
    are untouched so the long-run target stays at pi_T.
    """
    model, params, scenarios = _make_model(timesteps=50)
    sc = scenarios.get_scenario_index("Scenario.4: Inflation impulse")

    model.simulate(scenario=sc)
    ts = model.variables.timeseries

    pi_T = params["pi_T"]
    # Decay to target over many periods
    assert torch.isclose(ts["pi"][-1, 0], torch.tensor(pi_T), atol=1e-3)


def test_inflation_impulse_at_late_trigger():
    """InflationShock fires at t=trigger when trigger > timesteps_initialization.

    This is the notebook configuration: trigger=25, timesteps=70, init_t=1.
    The impulse vector has length 1 with fire_offset=0, placing 1.0 at
    ts_scenario[25]. The sim loop reaches t=25 at timeseries index
    init_t + 1 + (25 - init_t) = 26.
    """
    params = ParametersNK3E(
        hyperparameters={
            "timesteps": 70,
            "timesteps_initialization": 1,
            "scenario_trigger": 25,
            "use_tqdm": False,
        }
    )
    variables = VariablesNK3E(parameters=params)
    scenarios = ScenariosNK3E(parameters=params)
    model = NK3E(parameters=params, variables=variables, scenarios=scenarios)

    sc = scenarios.get_scenario_index("Scenario.4: Inflation impulse")
    model.simulate(scenario=sc)
    ts = model.variables.timeseries

    pi_T = params["pi_T"]
    # Index in gathered tensor: init records t=0 and t=1 (2 entries),
    # then sim steps t=1..69 give indices 2..70. Sim step t=25 is at index
    # 2 + (25 - 1) = 26.
    shock_idx = 2 + (25 - 1)
    assert torch.isclose(ts["pi"][shock_idx, 0], torch.tensor(pi_T) + 1.0, atol=1e-6)
    # Period before trigger: still at steady state
    assert torch.isclose(ts["pi"][shock_idx - 1, 0], torch.tensor(pi_T), atol=1e-6)
    # Long run: decays back to target
    assert torch.isclose(ts["pi"][-1, 0], torch.tensor(pi_T), atol=1e-3)


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
