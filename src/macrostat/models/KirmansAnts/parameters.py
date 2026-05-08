# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
"""Parameters class for the Kirman ants SDE model.

Reference: Kirman, A. (1993). "Ants, Rationality, and Recruitment".
The Quarterly Journal of Economics, 108(1), 137-156.

Continuous-time large-N limit (Moran et al. 2020) with a stationary
:math:`\\text{Beta}(\\rho/\\mu, \\rho/\\mu)` distribution on :math:`x \\in [0, 1]`.
"""

import logging

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersKirmansAnts(Parameters):
    """Parameters for the Kirman ants SDE model.

    Economic meaning:
    - rho (>0): rate of spontaneous opinion switching ("self-conversion").
    - mu (>0): herding strength (recruitment intensity).

    The ratio rho/mu determines the regime: rho/mu > 1 unimodal at x=1/2,
    rho/mu < 1 bimodal with mass near x=0 and x=1.

    Hyperparameters control the simulation environment:
    - timesteps: number of macro periods (each = 1 time unit of SDE evolution).
    - dt: SDE micro-step size (Euler-Maruyama discretisation).
    - substeps: number of micro-steps per macro period; must equal int(1/dt).
    - x0: initial density.
    - max_attempts: rejection-sampling cap inside one micro-step.
    - record_inner: if True, every micro-step is written to a side-buffer
      (size timesteps * substeps).
    """

    version = "KirmansAnts"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            parameters=parameters,
            hyperparameters=hyperparameters,
            *args,
            **kwargs,
        )

    def get_default_parameters(self):
        return {
            "rho": {
                "lower bound": 1e-6,
                "upper bound": 100.0,
                "notation": r"\rho",
                "unit": "1 / time",
                "value": 0.5,
            },
            "mu": {
                "lower bound": 1e-6,
                "upper bound": 100.0,
                "notation": r"\mu",
                "unit": "1 / time",
                "value": 1.0,
            },
        }

    def get_default_hyperparameters(self):
        hyper = super().get_default_hyperparameters()
        hyper["timesteps"] = 1000
        hyper["timesteps_initialization"] = 0
        hyper["sectors"] = ["Ants"]
        hyper["dt"] = 1e-4
        hyper["substeps"] = 10000
        hyper["x0"] = 0.5
        hyper["max_attempts"] = 1000
        hyper["record_inner"] = False
        return hyper

    def verify_parameters(self):
        """Enforce KirmansAnts-specific consistency on top of the base bounds check.

        Raises
        ------
        ValueError
            If ``dt`` is not in (0, 1), if ``substeps != int(1/dt)``, or if
            ``x0`` is not strictly in (0, 1).
        """
        super().verify_parameters()

        dt = self.hyper["dt"]
        if not (0.0 < dt < 1.0):
            raise ValueError(f"dt must be in (0, 1); got {dt!r}.")
        if abs(round(1.0 / dt) - 1.0 / dt) > 1e-9:
            raise ValueError(f"1/dt must be integer-valued; got 1/{dt} = {1.0 / dt}.")
        if self.hyper["substeps"] != int(round(1.0 / dt)):
            raise ValueError(
                f"substeps must equal int(1/dt) = {int(round(1.0 / dt))}; "
                f"got {self.hyper['substeps']}."
            )

        x0 = self.hyper["x0"]
        if not (0.0 < x0 < 1.0):
            raise ValueError(f"x0 must be strictly in (0, 1); got {x0!r}.")

        if self.hyper["record_inner"]:
            n_floats = self.hyper["timesteps"] * self.hyper["substeps"]
            n_bytes = n_floats * 4  # float32
            if n_bytes > 1_000_000_000:
                logger.warning(
                    "record_inner=True will allocate %.2f GB "
                    "(timesteps=%d, substeps=%d, float32).",
                    n_bytes / 1e9,
                    self.hyper["timesteps"],
                    self.hyper["substeps"],
                )
