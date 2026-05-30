===============================
Behavioral Equations Mark0COVID
===============================
--------------
Step Equations
--------------
1. Compute Gamma And Ren

Bank-side gamma (real-rate gap above baseline) and per-firm
ren ratio (gamma times solvency).

.. math::
	:label: compute_gamma_and_ren
	:nowrap:

	\begin{align}
	\Gamma_t &= \Gamma_0 + \text{ReLU}\big(\alpha_\Gamma (\bar\rho^l_t - \hat\pi_t) - \Gamma_0\big), \\
	\text{ren}_{i,t} &= \Gamma_t \cdot \frac{A_{i,t}}{W_{i,t} Y_{i,t} + \epsilon}.
	\end{align}


2. Find Surviving Firms

Identify firms whose assets plus a fraction :math:`\Theta` of
payroll are positive; survivors stay alive next period, the rest
enter bankruptcy.

.. math::
	:label: find_surviving_firms
	:nowrap:

	\begin{align}
	\text{stay}_{i,t} = \alpha_{i,t-1} \cdot \text{diffwhere}\big(A_{i,t} + \Theta\, W_{i,t} Y_{i,t},\, 1,\, 0\big).
	\end{align}


3. Firm Accounting

EBIT, profits, household savings drawdown, asset update.

.. math::
	:label: firm_accounting
	:nowrap:

	\begin{align}
	\text{ebit}_{i,t} &= P_{i,t} \min(D_{i,t}, Y_{i,t}) - W_{i,t} Y_{i,t}, \\
	\Pi_{i,t} &= \text{ebit}_{i,t} + \rho^l_t \min(A_{i,t}, 0) + \rho^d_t \max(A_{i,t}, 0), \\
	A_{i,t} &\leftarrow A_{i,t} + \alpha_{i,t} \Pi_{i,t}, \\
	S_t &\leftarrow S_t - \sum_i \alpha_{i,t} \text{ebit}_{i,t}.
	\end{align}


4. Monetary Policy

Central bank policy: Taylor-like rule on EMA inflation only.

.. math::
	:label: monetary_policy
	:nowrap:

	\begin{align}
	\rho^0_t = \rho^\star + \phi_\pi (\pi^{ema}_t - \pi^\star).
	\end{align}


5. Price Adjustment

Price update from the frozen-noise buffer.
Excess-demand firms with below-average price scale ``P`` up by
``(1 + rp)``; excess-supply firms with above-average price scale
down by ``(1 - rp)``. Branching uses ``torch.where`` to match
abmstat; gradient flow w.r.t. ``gammap`` is through ``rp``.

.. math::
	:label: price_adjustment
	:nowrap:

	\begin{align}
	r_p &= \gamma_p \, u^p_{t,i}, \\
	P_{i,t} &\leftarrow P_{i,t} (1 + r_p)
	\text{ if stay}_{i,t}\,\text{excess}^D_{i,t}\,(P_{i,t} < \bar P_t).
	\end{align}


6. Renormalize Prices

Rescale all nominal quantities by the average price so that
:math:`\bar P_t \equiv 1` going into the wage/price update.

.. math::
	:label: renormalize_prices
	:nowrap:

	\begin{align}
	P_{i,t} \leftarrow P_{i,t} / \bar P_{t-1}, \quad
	W_{i,t} \leftarrow W_{i,t} / \bar P_{t-1}, \quad
	A_{i,t} \leftarrow A_{i,t} / \bar P_{t-1}.
	\end{align}


7. Update Averages

EWMA update of inflation, interest rate, and unemployment
registers; convex combination of EWMA inflation and CB target for
the expectation used downstream.

.. math::
	:label: update_averages
	:nowrap:

	\begin{align}
	\pi^{ema}_t &= \omega\,\pi_{t-1} + (1-\omega)\,\pi^{ema}_{t-1}, \\
	\hat\pi_t &= \tau^T \pi^\star + \tau^R \pi^{ema}_t.
	\end{align}


8. Wage Adjustment

Smooth wage update from the frozen-noise buffer.
Excess-demand profitable firms raise wages; excess-supply
loss-making firms lower wages. The wage ceiling
(cashflow-per-production) is enforced via ``diffmin``.

.. math::
	:label: wage_adjustment
	:nowrap:

	\begin{align}
	r_w = \gamma_p \cdot r \cdot u^w_{t,i}.
	\end{align}
