"""End-to-end synthetic non-SFC heterogeneous-agent fixture.

Defines a minimal toy model with two variables of *different* tensor shapes
driven by hyperparameter axis sizes:

- ``FirmCapital`` shape ``(N_firms,)``
- ``HouseholdWealth`` shape ``(N_households,)``

with ``N_firms = 20`` and ``N_households = 100``. The model has one scalar
parameter ``decay`` so the autograd path can be exercised end-to-end.

Exercises init → simulate → to_pandas → autograd-Jacobian, defending the
het-agent infrastructure against the variable-shape, non-SFC, and gather-once
contracts simultaneously.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Karl Naumann-Woleske

import pandas as pd
import torch

from macrostat.core import Model, Parameters, Scenarios, Variables
from macrostat.core.behavior import Behavior


class ToyHetParams(Parameters):
    def get_default_hyperparameters(self):
        base = super().get_default_hyperparameters()
        base.update(
            {
                "N_firms": 20,
                "N_households": 100,
                "timesteps": 5,
                "timesteps_initialization": 0,
            }
        )
        return base

    def get_default_parameters(self):
        return {
            "decay": {
                "value": 0.1,
                "lower bound": 0.0,
                "upper bound": 1.0,
                "unit": "1/period",
                "notation": r"\delta",
            }
        }


class ToyHetVariables(Variables):
    def get_default_variables(self):
        return {
            "FirmCapital": {
                "sectors": ["N_firms"],
                "history": 0,
                "unit": "currency",
                "notation": "K_f",
            },
            "HouseholdWealth": {
                "sectors": ["N_households"],
                "history": 0,
                "unit": "currency",
                "notation": "W_h",
            },
        }


class ToyHetBehavior(Behavior):
    def initialize(self):
        self.state["FirmCapital"] = torch.ones(self.hyper["N_firms"])
        self.state["HouseholdWealth"] = torch.ones(self.hyper["N_households"])

    def step(self, t, scenario, params=None, **kwargs):
        decay = params["decay"] if params is not None else self.parameters["decay"]
        self.state["FirmCapital"] = self.prior["FirmCapital"] * (1.0 - decay)
        self.state["HouseholdWealth"] = self.prior["HouseholdWealth"] * (
            1.0 - decay * 0.5
        )


def _make_toy_model():
    p = ToyHetParams()
    v = ToyHetVariables(parameters=p)
    s = Scenarios(parameters=p)
    return Model(parameters=p, variables=v, scenarios=s, behavior=ToyHetBehavior)


def test_init_simulate_to_pandas_end_to_end():
    """Init → simulate → to_pandas runs without error and produces tensors
    of the expected per-variable shape and a wide-format DataFrame.

    The exact time dimension depends on the init+main loop interaction
    inside :meth:`Behavior.forward`; what matters here is that the two
    variables share a single time axis and carry their distinct agent-axis
    sizes through to the final ``self.timeseries`` dict.
    """
    model = _make_toy_model()
    model.simulate()

    ts = model.variables.timeseries
    assert ts["FirmCapital"].shape[1] == 20
    assert ts["HouseholdWealth"].shape[1] == 100
    assert ts["FirmCapital"].shape[0] == ts["HouseholdWealth"].shape[0]
    assert ts["FirmCapital"].shape[0] >= model.parameters["timesteps"]

    df = model.variables.to_pandas()
    assert isinstance(df, pd.DataFrame)
    assert set(df.columns.get_level_values(0)) == {"FirmCapital", "HouseholdWealth"}
    assert df.index.name == "time"


def test_autograd_path_yields_finite_gradient():
    """Autograd path produces finite, non-zero gradients for the toy
    parameter ``decay`` — proves the gather-once contract keeps the
    autograd graph intact end-to-end through the variable-shape state.
    """
    model = _make_toy_model()
    behavior = model.get_model_training_instance()

    out = behavior.forward()
    loss = out["FirmCapital"].sum() + out["HouseholdWealth"].sum()
    grad = torch.autograd.grad(loss, [behavior.params["decay"]])[0]

    assert torch.isfinite(grad).all()
    assert grad.abs().item() > 0.0
