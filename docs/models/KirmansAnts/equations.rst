================================
Behavioral Equations KirmansAnts
================================
--------------
Step Equations
--------------
Run ``substeps`` Euler-Maruyama micro-steps in Lamperti space.
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

.. math::
	:label: step
	:nowrap:

	\begin{align}
	\phi &= \arcsin(2 x - 1), \\
	d\phi_t &= \mu_\Phi(\phi_t)\, dt + \sigma_\Phi\, dW_t,
	\quad \sigma_\Phi = \sqrt{2\mu}, \\
	\mu_\Phi(\phi) &= -(2\rho - \mu)\tan(\phi)
	= -\,\frac{(2\rho - \mu)\,(2 x - 1)}{2\sqrt{x(1-x)}}, \\
	x &= \tfrac{1}{2}\bigl(1 + \sin\phi\bigr).
	\end{align}


1. Step

Run ``substeps`` Euler-Maruyama micro-steps in Lamperti space.
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

.. math::
	:label: step
	:nowrap:

	\begin{align}
	\phi &= \arcsin(2 x - 1), \\
	d\phi_t &= \mu_\Phi(\phi_t)\, dt + \sigma_\Phi\, dW_t,
	\quad \sigma_\Phi = \sqrt{2\mu}, \\
	\mu_\Phi(\phi) &= -(2\rho - \mu)\tan(\phi)
	= -\,\frac{(2\rho - \mu)\,(2 x - 1)}{2\sqrt{x(1-x)}}, \\
	x &= \tfrac{1}{2}\bigl(1 + \sin\phi\bigr).
	\end{align}
