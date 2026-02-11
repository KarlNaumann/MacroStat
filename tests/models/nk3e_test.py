"""Targeted tests for the NK3E model.

These tests avoid optional dependencies by running only the NK3E pieces and
assert on closed-form one-step implications of the 3-equation system.
"""

import torch

from macrostat.models.NK3E import (
    NK3E,
    ParametersNK3E,
    ScenariosNK3E,
    VariablesNK3E,
)


def test_baseline_steady_state_holds_one_step():
    params = ParametersNK3E(
        hyperparameters={
            "timesteps": 3,
            "timesteps_initialization": 1,
            "use_tqdm": False,
        }
    )
    variables = VariablesNK3E(parameters=params)
    scenarios = ScenariosNK3E(parameters=params)
    model = NK3E(parameters=params, variables=variables, scenarios=scenarios)

    # Initialization places the system at steady state (y_e, pi_T, r_s).
    # After one simulated step with no shocks, the state should stay there.
    model.simulate()
    ts = model.variables.timeseries

    a1 = params["a1"]
    A = params["A"]
    y_e = params["y_e"]
    pi_T = params["pi_T"]
    r_s = (A - y_e) / a1

    # Check last recorded values (t=2 since timesteps=3 with init at t=0..1)
    assert torch.isclose(ts["y"][2, 0], torch.tensor(y_e), atol=1e-6)
    assert torch.isclose(ts["pi"][2, 0], torch.tensor(pi_T), atol=1e-6)
    assert torch.isclose(ts["r"][2, 0], torch.tensor(r_s), atol=1e-6)


def test_scenario1_increase_A_raises_y_and_r_first_step():
    params = ParametersNK3E(
        hyperparameters={
            "timesteps": 3,
            "timesteps_initialization": 1,
            "use_tqdm": False,
        }
    )
    variables = VariablesNK3E(parameters=params)
    scenarios = ScenariosNK3E(parameters=params)
    sc1 = scenarios.get_scenario_index("Scenario.1: Rise in A")
    model = NK3E(parameters=params, variables=variables, scenarios=scenarios)

    # Simulate scenario 1
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

    # First simulated step (t=2) from steady state with r_{t-1}=r_s_baseline
    r_prev = (params["A"] - params["y_e"]) / a1
    y_t = A - a1 * r_prev
    pi_t = pi_T + a2 * (y_t - y_e)
    r_t = r_s + a3 * (pi_t - pi_T)

    assert y_t > y_e
    assert r_t >= r_prev
    assert torch.isclose(ts["y"][2, 0], torch.tensor(y_t), atol=1e-6)
    assert torch.isclose(ts["r"][2, 0], torch.tensor(r_t), atol=1e-6)


def test_scenario2_higher_piT_raises_r_first_step():
    params = ParametersNK3E(
        hyperparameters={
            "timesteps": 3,
            "timesteps_initialization": 1,
            "use_tqdm": False,
        }
    )
    variables = VariablesNK3E(parameters=params)
    scenarios = ScenariosNK3E(parameters=params)
    sc2 = scenarios.get_scenario_index("Scenario.2: Higher pi_T")
    model = NK3E(parameters=params, variables=variables, scenarios=scenarios)

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

    # First simulated step (t=2) from steady state
    r_prev = (params["A"] - params["y_e"]) / a1
    y_t = A - a1 * r_prev  # equals y_e
    pi_t = params["pi_T"] + a2 * (y_t - y_e)  # equals pi_T_baseline
    r_t = r_s + a3 * (pi_t - pi_T)  # pi_t - higher target is negative -> r_t < r_s

    assert r_t <= r_prev
    assert torch.isclose(ts["r"][2, 0], torch.tensor(r_t), atol=1e-6)
