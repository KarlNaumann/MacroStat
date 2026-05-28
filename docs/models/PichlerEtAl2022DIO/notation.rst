==========================================
Notation: Paper Symbols to Code Variables
==========================================

This page maps the mathematical symbols used in Pichler et al. (2022) to the
variable names used in the Python implementation. The **Source** column
indicates where each value lives at runtime: ``state`` (current-step
``self.state[...]``), ``prior`` (previous-step ``self.prior[...]``), ``params``
(scalar or tensor parameters in ``params[...]``), ``hyper``
(``params.hyper[...]``), ``scenario`` (per-step scenario tensor), or ``local``
(method-local variable).

Time-Varying Variables
----------------------
Stored in ``self.state[...]`` and ``self.prior[...]``. Addressable via
:py:attr:`macrostat.models.PichlerEtAl2022DIO.VariablesPichlerEtAl2022DIO`.

.. list-table::
   :header-rows: 1
   :widths: 12 28 12 48

   * - Paper symbol
     - Code variable
     - Source
     - Meaning
   * - :math:`x_{i,t}`
     - ``GrossOutput``
     - state
     - Realised gross output of industry *i*.
   * - :math:`d_{i,t}`
     - ``AggregateDemand``
     - state
     - Total demand for output of industry *i*.
   * - :math:`c^d_{i,t}`
     - ``ConsumptionDemand``
     - state
     - Household consumption demand for good *i*.
   * - :math:`c_{i,t}`
     - ``RealizedConsumption``
     - state
     - Realised household consumption of good *i*.
   * - :math:`l_{i,t}`
     - ``LabourCompensation``
     - state
     - Labour compensation paid by industry *i*.
   * - :math:`\pi_{i,t}`
     - ``Profits``
     - state
     - Profit of industry *i*.
   * - :math:`x^{\text{cap}}_{i,t}`
     - ``ProductiveCapacity``
     - state
     - Labour-based productive capacity.
   * - :math:`x^{\text{inp}}_{i,t}`
     - ``InputCapacity``
     - state
     - Input-based productive capacity.
   * - :math:`S_{ji,t}`
     - ``Inventories``
     - state
     - Inventory of input *j* held by industry *i*.
   * - :math:`Z_{ji,t}`
     - ``IntermediateConsumption``
     - state
     - Intermediate delivery of *j* received by *i*.
   * - :math:`O_{ji,t}`
     - ``IntermediateOrders``
     - state
     - Intermediate order from *i* to *j*.
   * - :math:`\tilde{c}^d_t`
     - ``TotalConsumptionDemand``
     - state
     - Household-aggregate consumption demand.
   * - :math:`s_t`
     - ``Savings``
     - state
     - Household savings stock.

Data Parameters
---------------
Tensor-valued parameters initialised from external CSV files. Addressable via
``model.parameters.data["..."]["value"]``.

.. list-table::
   :header-rows: 1
   :widths: 12 28 12 48

   * - Paper symbol
     - Code variable
     - Source
     - Meaning
   * - :math:`A_{ji}`
     - ``TechnicalCoefficients``
     - params
     - Technical coefficient: input *j* per unit output of *i*.
   * - :math:`A^{\text{ess}}_{ji}`
     - ``CriticalInputMatrix``
     - params
     - Mask × technical coefficient for critical inputs only.
   * - :math:`Z^{(0)}_{ji}`
     - ``IntermediateConsumptionMatrix``
     - params
     - Initial intermediate-consumption matrix.
   * - :math:`x_{i,0}`
     - ``InitialGrossOutput``
     - params
     - Initial gross output by industry.
   * - :math:`l_{i,0}`
     - ``InitialLabourCompensation``
     - params
     - Initial labour compensation by industry.
   * - :math:`c_{i,0}`
     - ``InitialHouseholdConsumption``
     - params
     - Initial household consumption by good.
   * - :math:`\pi_{i,0}`
     - ``InitialProfits``
     - params
     - Initial industry profits.
   * - :math:`f_{i,0}`
     - ``InitialOtherFinalDemand``
     - params
     - Initial other final demand by good.
   * - :math:`n_i`
     - ``InventoryTargetDays``
     - params
     - Days-of-cover inventory target.
   * - :math:`e_i / x_i`
     - ``OtherCostCoefficients``
     - params
     - Other-cost share (taxes, imports) of output.
   * - :math:`c^{\text{other}}`
     - ``HouseholdOtherCostCoefficient``
     - params
     - Household share spent on non-modelled goods.

Scalar Parameters
-----------------
Real-valued scalar parameters in ``model.parameters.values["..."]["value"]``.

.. list-table::
   :header-rows: 1
   :widths: 12 28 12 48

   * - Paper symbol
     - Code variable
     - Source
     - Meaning
   * - :math:`\tau`
     - ``InventoryAdjustmentSpeed``
     - params
     - Speed of inventory adjustment toward target.
   * - :math:`\gamma_H`
     - ``HiringRate``
     - params
     - Daily hiring rate.
   * - :math:`\gamma_F`
     - ``FiringRate``
     - params
     - Daily firing rate.
   * - :math:`\rho`
     - ``ConsumptionPersistence``
     - params
     - AR(1) persistence in aggregate consumption demand.
   * - :math:`m`
     - ``PropensityToConsume``
     - params
     - Marginal propensity to consume out of labour income.
   * - :math:`b`
     - ``BenefitRate``
     - params
     - Unemployment benefit replacement rate.
   * - :math:`\Delta s`
     - ``SavingsRedirection``
     - params
     - Fraction of savings redirected to consumption.

Hyperparameters
---------------
Configuration values in ``model.parameters.hyper["..."]``.

.. list-table::
   :header-rows: 1
   :widths: 12 28 12 48

   * - Paper symbol
     - Code variable
     - Source
     - Meaning
   * - :math:`N`
     - ``n_sectors``
     - hyper
     - Number of input-output sectors.
   * - :math:`T`
     - ``timesteps``
     - hyper
     - Simulation horizon (days).
   * - —
     - ``production_function``
     - hyper
     - One of ``leontief``, ``strongly_critical``, ``half_critical``, ``weakly_critical``, ``linear``.
   * - —
     - ``hiring_firing``
     - hyper
     - Whether labour adjusts via hire/fire rates.
   * - —
     - ``firm_priority``
     - hyper
     - Whether firms are rationed before households on tight supply.

Scenario Tensors
----------------
Per-step exogenous paths in ``scenario["..."]``.

.. list-table::
   :header-rows: 1
   :widths: 12 28 12 48

   * - Paper symbol
     - Code variable
     - Source
     - Meaning
   * - :math:`f^d_{i,t}`
     - ``other_final_demand``
     - scenario
     - Exogenous other final demand (govt + exports + investment).
   * - :math:`\epsilon^c_{i,t}`
     - ``consumption_shock``
     - scenario
     - Multiplicative shock to consumption demand.
   * - :math:`\epsilon^x_{i,t}`
     - ``productivity_shock``
     - scenario
     - Multiplicative shock to productive capacity.

Auxiliary Sets
--------------
Index sets derived from data parameters.

.. list-table::
   :header-rows: 1
   :widths: 12 28 12 48

   * - Paper symbol
     - Code variable
     - Source
     - Meaning
   * - :math:`\mathcal{V}_i`
     - ``critical_inputs[i]``
     - local
     - Indices *j* with :math:`A^{\text{ess}}_{ji} > 0`.
   * - :math:`\mathcal{U}_i`
     - ``important_inputs[i]``
     - local
     - Indices *j* with :math:`A_{ji} > 0` and :math:`A^{\text{ess}}_{ji} = 0`.

Selected Local Variables
------------------------
Names used inside ``BehaviorPichlerEtAl2022DIO`` methods.

.. list-table::
   :header-rows: 1
   :widths: 12 28 12 48

   * - Paper symbol
     - Code variable
     - Source
     - Meaning
   * - :math:`\delta l_{i,t}`
     - ``delta_labour``
     - local
     - Net hire/fire flow this step.
   * - :math:`\tilde{c}^d_t`
     - ``total_consumption_demand``
     - local
     - Aggregate of ``ConsumptionDemand`` across goods at time *t*.
   * - :math:`\log \tilde{c}^d_{t-1}`
     - ``log_total_consumption_demand``
     - local
     - Lagged log of aggregate consumption demand (AR(1) regressor).
   * - :math:`\sum_j Z_{ji,t}`
     - ``intermediate_consumption_total``
     - local
     - Column sum of intermediate deliveries by industry.
   * - :math:`\sum_j A_{ji}`
     - ``col_sum_technical_coefficients``
     - local
     - Column sum of the technical-coefficient matrix.
   * - :math:`\sum_j S_{ji,t}`
     - ``col_sum_inventory_matrix``
     - local
     - Column sum of inventories.
