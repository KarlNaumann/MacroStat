================================
Behavioral Equations KirmansAnts
================================
--------------
Step Equations
--------------
1. Euler Maruyama Advance

Run ``substeps`` Euler-Maruyama micro-steps with boundary rejection.

.. math::
	:label: euler_maruyama_advance
	:nowrap:

	\begin{align}
	dx_t = \rho\,(1 - 2 x_t)\, dt
	+ \sqrt{2 \mu\, x_t (1 - x_t)}\, dW_t
	\end{align}
