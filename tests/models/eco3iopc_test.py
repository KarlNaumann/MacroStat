"""Targeted tests for the ECO3IOPC model.

Covers the climate-channel propensity-to-consume term introduced via the
``PropensityToConsumeIncomeTemperature`` parameter and the ``InitialTemperature``
state snapshot, plus the ``torch.clamp(min=0)`` guard.
"""

# Copyright (c) 2026 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import torch

from macrostat.models.ECO3IOPC import (
    ECO3IOPC,
    ParametersECO3IOPC,
    ScenariosECO3IOPC,
    VariablesECO3IOPC,
)

ALPHA12_MONOTONIC_SWEEP = (0.0, 0.001, 0.01)


def _make_model(alpha12: float = 0.0, timesteps: int = 50):
    """Create a default ECO3IOPC model with given timesteps and alpha12."""
    params = ParametersECO3IOPC(
        hyperparameters={
            "timesteps": timesteps,
            "timesteps_initialization": 1,
            "use_tqdm": False,
        }
    )
    params["PropensityToConsumeIncomeTemperature"] = alpha12
    return ECO3IOPC(
        parameters=params,
        variables=VariablesECO3IOPC(parameters=params),
        scenarios=ScenariosECO3IOPC(parameters=params),
    )


def test_alpha12_positive_lowers_propensity():
    """Positive alpha12 lowers terminal propensity vs the alpha12=0 path.

    With the default emissions trajectory, cumulative CO2 grows and
    Temperature(t) > InitialTemperature, so the alpha12 contribution is
    strictly negative and the propensity sits below the alpha12=0 baseline.
    """
    baseline = _make_model(alpha12=0.0)
    baseline.simulate()
    perturbed = _make_model(alpha12=0.01)
    perturbed.simulate()

    baseline_terminal = baseline.variables.timeseries[
        "PropensityToConsumeIncome"
    ].squeeze()[-1]
    perturbed_terminal = perturbed.variables.timeseries[
        "PropensityToConsumeIncome"
    ].squeeze()[-1]

    assert perturbed_terminal < baseline_terminal


def test_alpha12_monotonic_sweep():
    """Terminal propensity is monotone non-increasing in alpha12.

    Holds the emissions / temperature path effectively fixed at the default
    scenario (alpha12 enters only the propensity equation; the climate block
    does not depend on alpha12) and sweeps alpha12 across three values.
    """
    terminal_propensities = []
    for alpha12 in ALPHA12_MONOTONIC_SWEEP:
        model = _make_model(alpha12=alpha12)
        model.simulate()
        terminal_propensities.append(
            float(model.variables.timeseries["PropensityToConsumeIncome"].squeeze()[-1])
        )

    for previous, current in zip(terminal_propensities[:-1], terminal_propensities[1:]):
        assert current <= previous


def test_alpha12_clamp_at_zero():
    """Extreme alpha12 drives the unclamped propensity below zero; the
    ``torch.clamp(min=0)`` guard caps it at exactly zero and consumption
    remains non-negative.
    """
    model = _make_model(alpha12=100.0)
    model.simulate()
    ts = model.variables.timeseries

    terminal_propensity = ts["PropensityToConsumeIncome"].squeeze()[-1]
    terminal_consumption = ts["RealConsumptionHousehold"].squeeze()[-1]

    assert torch.allclose(terminal_propensity, torch.zeros(()), rtol=0.0, atol=0.0)
    assert terminal_consumption >= 0.0
