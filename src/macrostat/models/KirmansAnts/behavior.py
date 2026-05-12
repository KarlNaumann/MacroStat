# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
r"""Behavior class for the Kirman ants SDE model.

Reference: Kirman, A. (1993). "Ants, Rationality, and Recruitment".
The Quarterly Journal of Economics, 108(1), 137-156.

Continuous-time large-N limit (see Moran et al. 2020) of the binary-choice
recruitment dynamics: the fraction :math:`x_t \in [0, 1]` of ants at source A
evolves under spontaneous switching at rate :math:`\rho` and herding at rate
:math:`\mu`.

The macro-period ``step`` runs ``substeps = int(1/dt)`` Euler-Maruyama
micro-steps with boundary rejection, mirroring the abmstat reference
implementation in ``packages/abmstat/abmstat/models/kirmanants.py``. Phase 1
of the migration replicates that implementation faithfully — including the
boundary-rejection bias that suppresses the empirical density near
:math:`x \in \{0, 1\}` in the bimodal regime.

The model is non-differentiable: the rejection while-loop has no autograd
path. ``BehaviorKirmansAnts.supports_differentiable = False`` so the base
class refuses ``differentiable=True`` at construction.
"""

import logging

import numpy as np
import torch

from macrostat.core.behavior import Behavior
from macrostat.models.KirmansAnts.parameters import ParametersKirmansAnts
from macrostat.models.KirmansAnts.scenarios import ScenariosKirmansAnts
from macrostat.models.KirmansAnts.variables import VariablesKirmansAnts

logger = logging.getLogger(__name__)


class BehaviorKirmansAnts(Behavior):
    """Simulation logic for the Kirman ants SDE model.

    Each ``step()`` call runs ``substeps`` Euler-Maruyama micro-steps using
    a boundary-rejection scheme to keep the density inside the open unit
    interval. ``self.numpy_rng`` (set by the base class from the ``seed``
    hyperparameter) is the per-instance ``numpy.random.Generator``; the
    global ``np.random`` state is never touched.

    With ``record_inner=True`` every micro-step is written to
    ``self._micro_trajectory`` (float32, length ``timesteps * substeps``).
    The base ``record_state`` continues to record one end-of-macro-period
    value per outer step, so ``model.output["density"]`` keeps its standard
    shape ``(timesteps, 1)``.
    """

    version = "KirmansAnts"
    supports_differentiable = False

    def __init__(
        self,
        parameters: ParametersKirmansAnts | None = None,
        scenarios: ScenariosKirmansAnts | None = None,
        variables: VariablesKirmansAnts | None = None,
        scenario: int = 0,
        differentiable: bool = False,
        debug: bool = False,
    ):
        if parameters is None:
            parameters = ParametersKirmansAnts()
        if scenarios is None:
            scenarios = ScenariosKirmansAnts(parameters=parameters)
        if variables is None:
            variables = VariablesKirmansAnts(parameters=parameters)

        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            scenario=scenario,
            differentiable=differentiable,
            debug=debug,
        )

        self._exhaustion_count: int = 0
        self._micro_trajectory: np.ndarray | None = None

    @property
    def exhaustion_count(self) -> int:
        """Number of micro-steps where rejection sampling hit ``max_attempts``."""
        return self._exhaustion_count

    def initialize(self):
        """Set the initial density and (re)allocate the micro-trajectory buffer.

        Resets the exhaustion counter on every ``forward()`` call so multiple
        runs of the same model instance report independent statistics.
        """
        self._exhaustion_count = 0

        if self.hyper["record_inner"]:
            self._micro_trajectory = np.empty(
                self.hyper["timesteps"] * self.hyper["substeps"],
                dtype=np.float32,
            )
        else:
            self._micro_trajectory = None

        self.state["density"] = torch.full_like(
            self.state["density"], float(self.hyper["x0"])
        )

    def step(self, t: int, scenario: dict, params: dict | None = None, **kwargs):
        """Advance the SDE by one macro-period of ``substeps`` Euler-Maruyama micro-steps.

        Parameters
        ----------
        t : int
            Current macro-period index.
        scenario : dict
            Vectorized scenario values at time t (unused; SDE is autonomous).
        params : dict
            Parameter values for time t with scenario shocks already applied.

        Raises
        ------
        RuntimeError
            If invoked while ``self.differentiable`` is True. The base class
            normally blocks this at construction; this guard catches mutation
            of the flag after init.
        """
        if self.differentiable:
            raise RuntimeError(
                "BehaviorKirmansAnts is non-differentiable; "
                "self.differentiable was mutated to True after construction."
            )
        self.euler_maruyama_advance(t=t, scenario=scenario, params=params)

    def euler_maruyama_advance(
        self, t: int, scenario: dict, params: dict | None = None
    ):
        r"""Run ``substeps`` Euler-Maruyama micro-steps with boundary rejection.

        Parameters
        ----------
        t : int
            Current macro-period index (used as the offset into the optional
            micro-trajectory side-buffer).
        scenario : dict
            Scenario dictionary (not used).
        params : dict | None
            Parameter values for time t with scenario shocks already applied.

        Notes
        -----
        Boundary rejection resamples only the noise draw (drift and diffusion
        held fixed) until the proposed increment keeps the density strictly
        inside :math:`(0, 1)`. On exhaustion the increment is set to zero and
        ``self._exhaustion_count`` is incremented; this matches the abmstat
        reference where the unbounded ``while True`` loop simply never
        terminates if the geometry is degenerate.

        Equations
        ---------
        .. math::
            \begin{align}
                dx_t = \rho\,(1 - 2 x_t)\, dt
                       + \sqrt{2 \mu\, x_t (1 - x_t)}\, dW_t
            \end{align}

        Dependency
        ----------
        - parameters: rho
        - parameters: mu
        - hyperparameters: dt
        - hyperparameters: substeps
        - hyperparameters: max_attempts
        - hyperparameters: record_inner
        - prior: density

        Sets
        -----
        - density
        """
        x = float(self.prior["density"].item())

        for k in range(self.hyper["substeps"]):
            drift = params["rho"].item() * (1.0 - 2.0 * x) * self.hyper["dt"]
            diffusion = np.sqrt(2.0 * params["mu"].item() * x * (1.0 - x))
            for _ in range(self.hyper["max_attempts"]):
                dx = drift + diffusion * self.numpy_rng.standard_normal() * np.sqrt(
                    self.hyper["dt"]
                )
                if 0.0 < x + dx < 1.0:
                    break
            else:
                self._exhaustion_count += 1
                logger.warning("max_attempts exhausted at t=%d, k=%d, x=%.6f", t, k, x)
                dx = 0.0
            x += dx
            if self.hyper["record_inner"]:
                self._micro_trajectory[t * self.hyper["substeps"] + k] = x

        self.state["density"] = torch.full_like(self.state["density"], x)
