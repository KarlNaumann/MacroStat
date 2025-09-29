=========================
Behavioral Equations NK3E
=========================
--------------
Step Equations
--------------
1. Central Bank Slope

Compute the monetary policy reaction slope a3 from structural parameters.

.. math::
	:label: central_bank_slope
	:nowrap:

	\begin{align}
	a_3 = \frac{1}{a_1\left(\frac{1}{a_2 b} + a_2\right)}
	\end{align}




2. Is Curve Output

IS curve: output as a function of demand shifter and lagged real rate.

.. math::
	:label: is_curve_output
	:nowrap:

	\begin{align}
	y_t = A - a_1 r_{t-1}
	\end{align}




3. Monetary Policy Rate

Monetary policy rule: real rate reacts to inflation deviations.

.. math::
	:label: monetary_policy_rate
	:nowrap:

	\begin{align}
	r_t = r_s + a_3 (\pi_t - \pi^T)
	\end{align}




4. Phillips Curve Inflation

Phillips curve: inflation responds to the output gap.

.. math::
	:label: phillips_curve_inflation
	:nowrap:

	\begin{align}
	\pi_t = \pi_{t-1} + a_2 (y_t - y_e)
	\end{align}




5. Stabilizing Real Rate

Compute the stabilizing real rate r_s consistent with output at potential.

.. math::
	:label: stabilizing_real_rate
	:nowrap:

	\begin{align}
	r_s = \frac{A - y_e}{a_1}
	\end{align}
