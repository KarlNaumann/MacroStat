"""
Behavior for the 2D linear test model.

The model evolves according to:

.. math::
    x_{t+1} = A x_t,

where :math:`A` is a 2x2 matrix of parameters and :math:`x_0` is an
initial state parameter.
"""

from __future__ import annotations

import logging

import torch

from macrostat.core.behavior import Behavior
from macrostat.models.LINEAR2D.parameters import ParametersLINEAR2D
from macrostat.models.LINEAR2D.scenarios import ScenariosLINEAR2D
from macrostat.models.LINEAR2D.variables import VariablesLINEAR2D

logger = logging.getLogger(__name__)


class BehaviorLINEAR2D(Behavior):
    """Behavior for the 2D linear test model."""

    version = "LINEAR2D"

    def __init__(
        self,
        parameters: ParametersLINEAR2D | None = None,
        scenarios: ScenariosLINEAR2D | None = None,
        variables: VariablesLINEAR2D | None = None,
        scenario: int = 0,
        debug: bool = False,
    ):
        if parameters is None:
            parameters = ParametersLINEAR2D()
        if scenarios is None:
            scenarios = ScenariosLINEAR2D(parameters=parameters)
        if variables is None:
            variables = VariablesLINEAR2D(parameters=parameters)

        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            scenario=scenario,
            debug=debug,
        )

    def initialize(self):
        """Initialize the 2D state from the x0 parameters."""
        x0_1 = self.params["x0_1"]
        x0_2 = self.params["x0_2"]
        self.state["State"] = torch.stack([x0_1, x0_2])

    def step(self, t: int, scenario: dict, params: dict | None = None):
        """Single-step update: x_{t+1} = A x_t."""
        a11 = self.params["a11"]
        a12 = self.params["a12"]
        a21 = self.params["a21"]
        a22 = self.params["a22"]
        A = torch.stack([a11, a12, a21, a22]).reshape(2, 2)

        x_prev = self.prior["State"]
        self.state["State"] = A @ x_prev
