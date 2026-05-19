# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
r"""Behavior class for the Kirman ants SDE model.

Reference: Kirman, A. (1993). "Ants, Rationality, and Recruitment".
The Quarterly Journal of Economics, 108(1), 137-156.

Continuous-time large-N limit (see Moran et al. 2020) of the binary-choice
recruitment dynamics: the fraction :math:`x_t \in [0, 1]` of ants at source A
evolves under spontaneous switching at rate :math:`\rho` and herding at rate
:math:`\mu`. The :math:`x`-space Itô SDE is

.. math::
    dx_t = \rho\,(1 - 2 x_t)\, dt + \sqrt{2\,\mu\, x_t\,(1 - x_t)}\, dW_t.

Naive Euler-Maruyama integration of this SDE has a worst-case
:math:`O(\sqrt{dt})` weak bias near the boundary because the diffusion
coefficient is only Hölder-1/2 at :math:`x \in \{0, 1\}`. The bias is most
visible in the bimodal regime (:math:`\rho/\mu < 1`) where the stationary
:math:`\text{Beta}(\rho/\mu, \rho/\mu)` density diverges at the endpoints.

This module integrates the Lamperti-transformed SDE (Moran, Fosset,
Benzaquen, Bouchaud 2020). Under :math:`\phi = \arcsin(2 x - 1)` the
diffusion becomes constant :math:`= \sqrt{2\mu}`, so Euler-Maruyama in
:math:`\phi` is Lipschitz and recovers the standard :math:`O(dt)` weak rate.
Reflective boundary conditions are applied in :math:`\phi`-space at
:math:`[-\pi/2,\,+\pi/2]`; the inverse map :math:`x = (1 + \sin\phi)/2`
returns the density to its native interval before recording.

The macro-period ``step`` runs ``substeps = int(1/dt)`` Lamperti micro-steps
internally. The model is non-differentiable;
``BehaviorKirmansAnts.supports_differentiable = False`` so the base class
refuses ``differentiable=True`` at construction.
"""

import logging
import math

import numpy as np
import torch

from macrostat.core.behavior import Behavior
from macrostat.models.KirmansAnts.parameters import ParametersKirmansAnts
from macrostat.models.KirmansAnts.scenarios import ScenariosKirmansAnts
from macrostat.models.KirmansAnts.variables import VariablesKirmansAnts

logger = logging.getLogger(__name__)


class BehaviorKirmansAnts(Behavior):
    r"""Simulation logic for the Kirman ants SDE model.

    Each ``step()`` call runs ``substeps`` Lamperti micro-steps via
    :meth:`advance_lamperti`. ``self.numpy_rng`` (set by the base class from
    ``hyper["seed"]``) is the per-instance primary
    :class:`numpy.random.Generator`. Each macro-step consumes exactly
    ``substeps`` Gaussians from this stream, so the noise stream is aligned
    across the 5-point parameter stencil used by the downstream Fisher
    information pipeline (paired-seed CRN).

    With ``record_inner=True`` every micro-step :math:`x` value is written
    to ``self._micro_trajectory`` (float32, length
    ``timesteps * substeps``). The Lamperti integrator integrates in
    :math:`\phi`-space but inverse-maps to :math:`x` before recording, so
    downstream KDE / FIM consumers stay method-agnostic.
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

        self._micro_trajectory: np.ndarray | None = None

    def initialize(self):
        """Set initial density and allocate the micro-trajectory side-buffer.

        Resets the per-run state on every ``forward()`` call so multiple
        runs of the same model instance are independent.
        """
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
        """Advance the SDE by one macro-period.

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
        self.advance_lamperti(t=t, params=params)

    # --- Lamperti map helpers ----------------------------------------------

    @staticmethod
    def _lamperti_forward(x: float) -> float:
        r"""Forward Lamperti map :math:`\phi = \arcsin(2 x - 1)`."""
        return math.asin(2.0 * x - 1.0)

    @staticmethod
    def _lamperti_inverse(phi: float) -> float:
        r"""Inverse Lamperti map :math:`x = (1 + \sin\phi)/2`."""
        return 0.5 * (1.0 + math.sin(phi))

    @staticmethod
    def _lamperti_drift(x: float, rho: float, mu: float) -> float:
        r"""Drift of the Lamperti-transformed SDE evaluated at the cached :math:`x`.

        Parameters
        ----------
        x : float
            Current :math:`x`-space density (cached from previous inverse map).
        rho : float
            Spontaneous switching rate.
        mu : float
            Herding strength.

        Returns
        -------
        float
            :math:`\mu_\Phi(\phi)` evaluated at :math:`x`.

        Notes
        -----
        Using :math:`\phi = f(x) = \arcsin(2 x - 1)` (Moran et al.\ 2020),
        the Itô transform gives
        :math:`\mu_\Phi = a(x) f'(x) + \tfrac{1}{2}\sigma^2(x) f''(x)`,
        with constant-diffusion :math:`\sigma_\Phi \equiv \sqrt{2\mu}`. The
        :math:`x`-form below is evaluated directly because :math:`x` is
        already cached from the previous inverse map.

        Equations
        ---------
        .. math::
            \begin{align}
                \mu_\Phi(\phi)
                  = -(2\rho - \mu)\tan(\phi)
                  = -\,\frac{(2\rho - \mu)\,(2 x - 1)}{2\sqrt{x(1 - x)}}.
            \end{align}
        """
        return -(2.0 * rho - mu) * (2.0 * x - 1.0) / (2.0 * math.sqrt(x * (1.0 - x)))

    # --- integrator --------------------------------------------------------

    def advance_lamperti(self, t: int, params: dict):
        r"""Run ``substeps`` Euler-Maruyama micro-steps in Lamperti space.

        Parameters
        ----------
        t : int
            Macro-period index.
        params : dict
            Parameter values at time t.

        Notes
        -----
        Uses the Lamperti transform :math:`\phi = \arcsin(2 x - 1)` from
        Moran, Fosset, Benzaquen, Bouchaud (2020). The diffusion in
        :math:`\phi` is constant :math:`= \sqrt{2\mu}`, so Euler-Maruyama in
        :math:`\phi` is Lipschitz and recovers the standard :math:`O(dt)`
        weak rate. The :math:`\phi`-domain is :math:`[-\pi/2,\,+\pi/2]`;
        reflective BC in :math:`\phi`-space is the geometrically correct
        boundary condition.

        :math:`x` is clipped to :math:`[\epsilon, 1 - \epsilon]` with
        :math:`\epsilon = 10^{-7}` before the forward map and after the
        inverse map. The clip sits above the float32 precision of the
        side-buffer near the boundary and far below any KDE bandwidth used
        downstream, so the clip is invisible to consumers.

        ``_micro_trajectory`` stores :math:`x` (not :math:`\phi`); the
        inverse map is applied at every substep so downstream consumers stay
        method-agnostic.

        Equations
        ---------
        .. math::
            \begin{align}
                \phi &= \arcsin(2 x - 1), \\
                d\phi_t &= \mu_\Phi(\phi_t)\, dt + \sigma_\Phi\, dW_t,
                  \quad \sigma_\Phi = \sqrt{2\mu}, \\
                \mu_\Phi(\phi) &= -(2\rho - \mu)\tan(\phi)
                  = -\,\frac{(2\rho - \mu)\,(2 x - 1)}{2\sqrt{x(1-x)}}, \\
                x &= \tfrac{1}{2}\bigl(1 + \sin\phi\bigr).
            \end{align}

        Dependency
        ----------
        - params: rho
        - params: mu
        - hyper: dt
        - hyper: substeps
        - hyper: record_inner
        - prior: density

        Sets
        -----
        - density
        """
        rho = float(params["rho"].item())
        mu = float(params["mu"].item())
        dt = self.hyper["dt"]
        sqrt_dt = math.sqrt(dt)
        sigma_phi = math.sqrt(2.0 * mu)
        substeps = self.hyper["substeps"]
        record_inner = self.hyper["record_inner"]
        buffer = self._micro_trajectory
        eps_x = 1.0e-7
        phi_min = -0.5 * math.pi
        phi_max = +0.5 * math.pi

        x = float(self.prior["density"].item())
        x = min(max(x, eps_x), 1.0 - eps_x)
        phi = self._lamperti_forward(x)
        noise = self.numpy_rng.standard_normal(substeps)

        for k in range(substeps):
            drift_phi = self._lamperti_drift(x, rho, mu)
            phi = phi + drift_phi * dt + sigma_phi * noise[k] * sqrt_dt
            for _ in range(2):
                if phi < phi_min:
                    phi = 2.0 * phi_min - phi
                elif phi > phi_max:
                    phi = 2.0 * phi_max - phi
                else:
                    break
            if phi <= phi_min or phi >= phi_max:
                phi = min(max(phi, phi_min + 1.0e-15), phi_max - 1.0e-15)
            x = self._lamperti_inverse(phi)
            x = min(max(x, eps_x), 1.0 - eps_x)
            if record_inner:
                buffer[t * substeps + k] = x

        self.state["density"] = torch.full_like(self.state["density"], x)
