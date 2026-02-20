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

.. math::
	:label: accounting
	:nowrap:

	\begin{align}
	\pi_{i,t} = x_{i,t} - \sum_{j=1}^{N} Z_{ji,t}
	- l_{i,t} - e_{i,t}
	\end{align}


where:
- :math:`\pi_{i,t}` is the profit of industry *i*
- :math:`x_{i,t}` is total output
- :math:`Z_{ji,t}` is intermediate purchases of input *j* by industry *i*
- :math:`l_{i,t}` is labour compensation paid by industry *i*
- :math:`e_{i,t}` is other expenses (taxes, imports, etc.)




2. Aggregate Demand

Total demand aggregation.
Total demand for the output of industry *i* is the sum of
intermediate orders from all other industries, household
consumption demand, and exogenous other final demand (government,
exports, investment).

.. math::
	:label: aggregate_demand
	:nowrap:

	\begin{align}
	d_{i,t} = \sum_{j=1}^{N} O_{ij,t} + c^d_{i,t} + f^d_{i,t}
	\end{align}


where:
- :math:`d_{i,t}` is total demand for the output of industry *i*
- :math:`O_{ij,t}` is the intermediate order from industry *j* to industry *i*
- :math:`c^d_{i,t}` is household consumption demand for good *i*
- :math:`f^d_{i,t}` is exogenous other final demand for good *i*




3. Compute Production

Production function and output-level choice.
Realized output is the minimum of three constraints: labour
capacity, input-based capacity (from inventories and the chosen
production function), and demand.  The input-based capacity depends
on the ``production_function`` hyperparameter.  Five functional
forms are available, ranging from Leontief (all inputs binding)
through partially binding Leontief variants to linear (perfect
substitutes).  The partially binding Leontief distinguishes
critical, important, and non-critical inputs based on an industry
analyst survey.

.. math::
	:label: compute_production
	:nowrap:

	\begin{align}
	Output choice (Eq. 14):
	x_{i,t} = \min\{x^{\text{cap}}_{i,t},\;
	x^{\text{inp}}_{i,t},\;
	d_{i,t}\}
	Leontief (Eq. 9):
	x^{\text{inp}}_{i,t} = \min_{\{j:\,A_{ji}>0\}}
	\frac{S_{ji,t}}{A_{ji}}
	Strongly-critical (Eq. 10):
	x^{\text{inp}}_{i,t} = \min_{j \in \mathcal{V}_i
	\cup \mathcal{U}_i} \frac{S_{ji,t}}{A_{ji}}
	Half-critical (Eq. 11):
	x^{\text{inp}}_{i,t} = \min_{\{j \in \mathcal{V}_i,\;
	k \in \mathcal{U}_i\}} \left\{
	\frac{S_{ji,t}}{A_{ji}},\;
	\frac{1}{2}\!\left(
	\frac{S_{ki,t}}{A_{ki}} + x^{\text{cap}}_{i,0}
	\right)\right\}
	Weakly-critical (Eq. 12):
	x^{\text{inp}}_{i,t} = \min_{j \in \mathcal{V}_i}
	\frac{S_{ji,t}}{A_{ji}}
	Linear (Eq. 13):
	x^{\text{inp}}_{i,t} = \frac{\sum_j S_{ji,t}}
	{\sum_j A_{ji}}
	\end{align}


where:
- :math:`x_{i,t}` is realized output of industry *i*
- :math:`x^{\text{cap}}_{i,t}` is labour-based production capacity
- :math:`x^{\text{inp}}_{i,t}` is input-based production capacity
- :math:`d_{i,t}` is total demand
- :math:`S_{ji,t}` is the inventory of input *j* held by industry *i*
- :math:`A_{ji}` is the technical coefficient (input *j* per unit output of *i*)
- :math:`\mathcal{V}_i` is the set of critical inputs to industry *i*
- :math:`\mathcal{U}_i` is the set of important (but not critical) inputs to industry *i*
- :math:`x^{\text{cap}}_{i,0}` is the initial production capacity




4. Consumption Demand

Muellbauer consumption function with fear-of-infection.
Total household consumption demand follows an adapted version of
Muellbauer (2020), combining persistence of past consumption with
current and permanent labour income.  A fear-of-infection factor
:math:`(1 - \tilde{\epsilon}^D_t)` scales aggregate demand.
Consumption is allocated across industries via time-varying
preference coefficients.

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


where:
- :math:`\tilde{c}^d_t` is aggregate consumption demand
- :math:`\rho` is the persistence of consumption
- :math:`m` is the propensity to consume final domestic goods out of labour income
- :math:`\tilde{l}_t` is current aggregate labour income (effective, accounting for benefits)
- :math:`\tilde{l}^p_t` is permanent income expectation
- :math:`\tilde{\epsilon}^D_t` is the aggregate demand shock due to fear of infection
- :math:`\theta_{i,t}` is the time-varying preference share for industry *i*
- :math:`c^d_{i,t}` is consumption demand for the output of industry *i*




5. Hire Fire

Sluggish labour adjustment towards a target workforce.
Firms adjust their labour force depending on which production
constraint is binding.  If capacity is binding, the firm tries to
hire; if demand or input constraints bind, it fires.  Adjustment is
sluggish -- firms can only move a fraction of the way toward their
target each period.  During lockdown, labour is additionally capped
by the exogenous supply shock.

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


where:
- :math:`l_{i,t}` is labour compensation in industry *i* at time *t*
- :math:`l_{i,0}` is the initial (pre-pandemic) labour compensation
- :math:`x_{i,0}` is the initial output level
- :math:`x^{\text{inp}}_{i,t}` is the input-based production capacity
- :math:`x^{\text{cap}}_{i,t}` is the labour-based production capacity
- :math:`d_{i,t}` is total demand
- :math:`\gamma_H` is the hiring adjustment rate
- :math:`\gamma_F` is the firing adjustment rate




6. Intermediate Orders

Inventory-gap ordering of intermediate inputs.
Each industry places orders for intermediate inputs based on two
components: (1) a naive expectation that demand will equal last
period's level, scaled by the technical coefficients; and (2) a
correction term that moves inventories toward a target of
:math:`n_i` days of each input, at a speed governed by
:math:`\tau`.

.. math::
	:label: intermediate_orders
	:nowrap:

	\begin{align}
	O_{ji,t} = A_{ji}\,d_{i,t-1}
	+ \frac{1}{\tau}\left(n_i Z_{ji,0} - S_{ji,t-1}\right)
	\end{align}


where:
- :math:`O_{ji,t}` is the order from industry *i* to industry *j* for input *j*
- :math:`A_{ji} = Z_{ji,0}/x_{i,0}` is the technical coefficient
- :math:`d_{i,t-1}` is lagged total demand for industry *i*
- :math:`n_i` is the target inventory in days for industry *i*
- :math:`Z_{ji,0}` is the baseline intermediate consumption of input *j* by *i*
- :math:`S_{ji,t-1}` is the current inventory of input *j* held by *i*
- :math:`\tau` is the inventory adjustment speed (in days)




7. Inventory Update

Inventory accumulation from deliveries minus usage.
After production and rationing, each industry updates its inventory
of every input.  Inventories increase by deliveries received and
decrease by inputs consumed in production (at rates given by the
technical coefficients).  A floor of zero prevents negative stocks
for non-critical inputs that may be fully depleted.

.. math::
	:label: inventory_update
	:nowrap:

	\begin{align}
	S_{ji,t+1} = \max\{S_{ji,t} + Z_{ji,t}
	- A_{ji}\,x_{i,t},\; 0\}
	\end{align}


where:
- :math:`S_{ji,t+1}` is the inventory of input *j* held by industry *i* at the start of the next period
- :math:`S_{ji,t}` is the current inventory
- :math:`Z_{ji,t}` is the intermediate delivery of *j* received by *i*
- :math:`A_{ji}` is the technical coefficient
- :math:`x_{i,t}` is realized output of industry *i*




8. Productive Capacity

Labour-scaled production capacity.
Each industry has a finite production capacity that scales linearly
with available labour relative to the pre-pandemic baseline.
Initially every industry employs :math:`l_{i,0}` workers and
produces at full capacity :math:`x^{\text{cap}}_{i,0} = x_{i,0}`.

.. math::
	:label: productive_capacity
	:nowrap:

	\begin{align}
	x^{\text{cap}}_{i,t} = \frac{l_{i,t}}{l_{i,0}}\,
	x^{\text{cap}}_{i,0}
	\end{align}


where:
- :math:`x^{\text{cap}}_{i,t}` is the labour-based production capacity of industry *i*
- :math:`l_{i,t}` is the current labour compensation
- :math:`l_{i,0}` is the initial labour compensation
- :math:`x^{\text{cap}}_{i,0}` is the initial production capacity (equal to initial output)




9. Rationing

Proportional rationing of output across buyers.
When output falls short of demand, industry *i* rations its output
proportionally across all customers.  Each buyer receives a share
of their order equal to the ratio of output to demand.  An optional
``firm_priority`` mode gives precedence to intermediate demand over
final consumption.

.. math::
	:label: rationing
	:nowrap:

	\begin{align}
	Z_{ji,t} &= O_{ji,t}\,\frac{x_{j,t}}{d_{j,t}}, &
	c_{i,t}  &= c^d_{i,t}\,\frac{x_{i,t}}{d_{i,t}}, &
	f_{i,t}  &= f^d_{i,t}\,\frac{x_{i,t}}{d_{i,t}}
	\end{align}


where:
- :math:`Z_{ji,t}` is the realized intermediate delivery from *j* to *i*
- :math:`O_{ji,t}` is the intermediate order placed by *i* to *j*
- :math:`c_{i,t}` is realized household consumption of good *i*
- :math:`c^d_{i,t}` is household consumption demand for good *i*
- :math:`f_{i,t}` is realized other final demand for good *i*
- :math:`f^d_{i,t}` is exogenous other final demand
- :math:`x_{i,t}` is output of industry *i*
- :math:`d_{i,t}` is total demand for the output of industry *i*
