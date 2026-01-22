"""
Parameters for the simple 2D linear model used for Jacobian testing.
"""

from __future__ import annotations

import logging

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersLINEAR2D(Parameters):
    """Parameters for the 2D linear test model.

    The model is:

    .. math::
        x_{t+1} = A x_t,

    where :math:`x_t \\in \\mathbb{R}^2` and :math:`A \\in \\mathbb{R}^{2\\times 2}`.

    We parameterize:

    - ``a11, a12, a21, a22``: entries of :math:`A`
    - ``x0_1, x0_2``: entries of the initial state :math:`x_0`
    """

    version = "LINEAR2D"

    def get_default_parameters(self):
        return {
            "a11": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"a_{11}",
                "unit": ".",
                "value": 0.9,
            },
            "a12": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"a_{12}",
                "unit": ".",
                "value": 0.1,
            },
            "a21": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"a_{21}",
                "unit": ".",
                "value": -0.2,
            },
            "a22": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"a_{22}",
                "unit": ".",
                "value": 0.8,
            },
            "x0_1": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"x_{0,1}",
                "unit": ".",
                "value": 1.0,
            },
            "x0_2": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"x_{0,2}",
                "unit": ".",
                "value": 0.0,
            },
        }

    def get_default_hyperparameters(self):
        hyper = super().get_default_hyperparameters()
        # Keep this model tiny and cheap
        hyper["timesteps"] = 5
        hyper["timesteps_initialization"] = 0
        # Minimal sector info to keep core utilities happy
        hyper.setdefault("sectors", ["Linear2D"])
        hyper.setdefault("vector_sectors", [])
        return hyper
