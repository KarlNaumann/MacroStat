=======================================
Behavioral Equations PichlerEtAl2022DIO
=======================================
--------------
Step Equations
--------------
1. Accounting

Firm profits and household savings.
Industry profits are total output minus intermediate purchases,
labour compensation, and other expenses (taxes, imports).
Household savings are computed as total income minus total realized
consumption (including non-modeled import and tax expenditures).
Replicates Eq. 2.

.. math::
	:label: accounting
	:nowrap:

	\begin{align}
	\pi_{i,t} = x_{i,t} - \sum_{j=1}^{N} Z_{ji,t}
	- l_{i,t} - e_{i,t}
	\end{align}


2. Aggregate Demand

Total demand aggregation.
Total demand for the output of industry *i* is the sum of
intermediate orders from all other industries, household
consumption demand, and exogenous other final demand (government,
exports, investment). Replicates Eq. 3.

.. math::
	:label: aggregate_demand
	:nowrap:

	\begin{align}
	d_{i,t} = \sum_{j=1}^{N} O_{ij,t} + c^d_{i,t} + f^d_{i,t}
	\end{align}


3. Compute Production

Production function and output-level choice.
Realized output is the minimum of three constraints: labour
capacity, input-based capacity (from inventories and the chosen
production function), and demand.  The input-based capacity depends
on the ``production_function`` hyperparameter; five functional forms
range from Leontief (all inputs binding) through partially binding
Leontief variants to linear (perfect substitutes).  The partially
binding Leontief distinguishes critical, important, and non-critical
inputs based on an industry analyst survey. Replicates Eqs. 8-14.

.. math::
	:label: compute_production
	:nowrap:

	\begin{align}
	x_{i,t} &= \min\{x^{\text{cap}}_{i,t},\;
	x^{\text{inp}}_{i,t},\;
	d_{i,t}\} \\[6pt]
	x^{\text{inp}}_{i,t} &= \begin{cases}
	\displaystyle\min_{\{j:\,A_{ji}>0\}}
	\frac{S_{ji,t}}{A_{ji}}
	& \text{leontief} \\
	\displaystyle\min_{j \in \mathcal{V}_i \cup \mathcal{U}_i}
	\frac{S_{ji,t}}{A_{ji}}
	& \text{strongly\_critical} \\
	\displaystyle\min\!\left\{
	\min_{j\in\mathcal{V}_i}\frac{S_{ji,t}}{A_{ji}},\;
	\tfrac{1}{2}\!\left(
	\min_{k\in\mathcal{U}_i}\frac{S_{ki,t}}{A_{ki}}
	+ x^{\text{cap}}_{i,0}
	\right)
	\right\}
	& \text{half\_critical} \\
	\displaystyle\min_{j \in \mathcal{V}_i}
	\frac{S_{ji,t}}{A_{ji}}
	& \text{weakly\_critical} \\
	\displaystyle\sum_j S_{ji,t} \big/ \sum_j A_{ji}
	& \text{linear}
	\end{cases}
	\end{align}


4. Consumption Demand

Muellbauer consumption function with fear-of-infection.
Total household consumption demand follows an adapted version of
Muellbauer (2020), combining persistence of past consumption with
current and permanent labour income.  A fear-of-infection factor
:math:`(1 - \tilde{\epsilon}^D_t)` scales aggregate demand.
Consumption is allocated across industries via time-varying
preference coefficients. Replicates Eqs. 5-6.

.. math::
	:label: consumption_demand
	:nowrap:

	\begin{align}
	\tilde{c}^d_t &= (1 - \tilde{\epsilon}^D_t)\,
	\exp\!\left(
	\rho \log \tilde{c}^d_{t-1}
	+ \frac{1-\rho}{2}\log(m\tilde{l}_t)
	+ \frac{1-\rho}{2}\log(m\tilde{l}^p_t)
	\right) \\
	c^d_{i,t} &= \theta_{i,t}\,\tilde{c}^d_t
	\end{align}


5. Hire Fire

Sluggish labour adjustment towards a target workforce.
Firms adjust their labour force depending on which production
constraint is binding.  If capacity is binding, the firm tries to
hire; if demand or input constraints bind, it fires.  Adjustment is
sluggish -- firms can only move a fraction of the way toward their
target each period.  During lockdown, labour is additionally capped
by the exogenous supply shock. Replicates Eqs. 19-20.

.. math::
	:label: hire_fire
	:nowrap:

	\begin{align}
	\Delta l_{i,t} &= \frac{l_{i,0}}{x_{i,0}}
	\left[\min\{x^{\text{inp}}_{i,t},\, d_{i,t}\}
	- x^{\text{cap}}_{i,t}\right] \\
	l_{i,t} &= \begin{cases}
	l_{i,t-1} + \gamma_H \Delta l_{i,t}
	& \text{if } \Delta l_{i,t} \ge 0 \\
	l_{i,t-1} + \gamma_F \Delta l_{i,t}
	& \text{if } \Delta l_{i,t} < 0
	\end{cases}
	\end{align}


6. Intermediate Orders

Inventory-gap ordering of intermediate inputs.
Each industry places orders for intermediate inputs based on two
components: (1) a naive expectation that demand will equal last
period's level, scaled by the technical coefficients; and (2) a
correction term that moves inventories toward a target of
:math:`n_i` days of each input, at a speed governed by
:math:`\tau`. Replicates Eq. 4.

.. math::
	:label: intermediate_orders
	:nowrap:

	\begin{align}
	O_{ji,t} = A_{ji}\,d_{i,t-1}
	+ \frac{1}{\tau}\left(n_i Z_{ji,0} - S_{ji,t-1}\right)
	\end{align}


7. Inventory Update

Inventory accumulation from deliveries minus usage.
After production and rationing, each industry updates its inventory
of every input.  Inventories increase by deliveries received and
decrease by inputs consumed in production (at rates given by the
technical coefficients).  A floor of zero prevents negative stocks
for non-critical inputs that may be fully depleted. Replicates
Eq. 18.

.. math::
	:label: inventory_update
	:nowrap:

	\begin{align}
	S_{ji,t+1} = \max\{S_{ji,t} + Z_{ji,t}
	- A_{ji}\,x_{i,t},\; 0\}
	\end{align}


8. Productive Capacity

Labour-scaled production capacity.
Each industry has a finite production capacity that scales linearly
with available labour relative to the pre-pandemic baseline.
Initially every industry employs :math:`l_{i,0}` workers and
produces at full capacity :math:`x^{\text{cap}}_{i,0} = x_{i,0}`.
Replicates Eq. 7.

.. math::
	:label: productive_capacity
	:nowrap:

	\begin{align}
	x^{\text{cap}}_{i,t} = \frac{l_{i,t}}{l_{i,0}}\,
	x^{\text{cap}}_{i,0}
	\end{align}


9. Rationing

Proportional rationing of output across buyers.
When output falls short of demand, industry *i* rations its output
proportionally across all customers.  Each buyer receives a share
of their order equal to the ratio of output to demand.  An optional
``firm_priority`` mode gives precedence to intermediate demand over
final consumption. Replicates Eqs. 15-17.

.. math::
	:label: rationing
	:nowrap:

	\begin{align}
	Z_{ji,t} &= O_{ji,t}\,\frac{x_{j,t}}{d_{j,t}}, &
	c_{i,t}  &= c^d_{i,t}\,\frac{x_{i,t}}{d_{i,t}}, &
	f_{i,t}  &= f^d_{i,t}\,\frac{x_{i,t}}{d_{i,t}}
	\end{align}
