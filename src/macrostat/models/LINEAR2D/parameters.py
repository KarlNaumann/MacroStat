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

    Parameters follow the sector-indexed naming convention so that
    :meth:`Parameters.vectorize_parameters` assembles them once at
    construction rather than on every :meth:`step`. With
    ``vector_sectors = ["S1", "S2"]``:

    - ``S1.S1.a, S1.S2.a, S2.S1.a, S2.S2.a``: entries of the matrix
      :math:`A`, assembled into a single ``(2, 2)`` tensor keyed ``a``.
    - ``S1.x0, S2.x0``: entries of the initial state :math:`x_0`,
      assembled into a single ``(2,)`` tensor keyed ``x0``.

    The four ``a`` entries and two ``x0`` entries remain independent scalar
    parameters, so the Jacobian tooling still reports one column per entry.
    """

    version = "LINEAR2D"

    def get_default_parameters(self):
        return {
            "S1.S1.a": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"a_{11}",
                "unit": ".",
                "value": 0.9,
            },
            "S1.S2.a": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"a_{12}",
                "unit": ".",
                "value": 0.1,
            },
            "S2.S1.a": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"a_{21}",
                "unit": ".",
                "value": -0.2,
            },
            "S2.S2.a": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"a_{22}",
                "unit": ".",
                "value": 0.8,
            },
            "S1.x0": {
                "lower bound": -10.0,
                "upper bound": 10.0,
                "notation": r"x_{0,1}",
                "unit": ".",
                "value": 1.0,
            },
            "S2.x0": {
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
        hyper.setdefault("vector_sectors", ["S1", "S2"])
        return hyper
