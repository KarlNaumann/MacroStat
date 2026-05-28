=====================================
Scenarios PichlerEtAl2022DIO
=====================================

The ``PichlerEtAl2022DIO`` model uses time-varying exogenous shocks to
simulate the economic impact of pandemic-induced supply and demand
disruptions.  Scenarios are managed by
``macrostat.models.PichlerEtAl2022DIO.scenarios.ScenariosPichlerEtAl2022DIO``.

Scenario variables
==================

The following scenario variables are defined with default (no-shock)
values in the steady state:

- **SupplyShock** -- :math:`\epsilon^S_{i,t} \in [0, 1]`, vector ``(T, N)``.
  Fraction of industry *i*'s labour supply that is unavailable.  Default: 0
  (no supply disruption).

- **DemandPreferences** -- :math:`\theta_{i,t}`, vector ``(T, N)``.
  Time-varying household consumption preference shares across industries.
  Default: initial steady-state consumption shares.

- **FearOfInfection** -- :math:`\tilde{\epsilon}^D_t \in [0, 1]`, scalar ``(T, 1)``.
  Aggregate demand reduction due to fear of infection.  Default: 0.

- **PermanentIncomeExpectation** -- :math:`\xi_t`, scalar ``(T, 1)``.
  Scaling factor on perceived permanent income.  Default: 1 (no pessimism).

- **OtherFinalDemand** -- :math:`f^d_{i,t}`, vector ``(T, N)``.
  Exogenous non-household final demand (government, investment, exports).
  Default: initial steady-state value.

Building scenarios
==================

Scenarios can be constructed in two ways:

1. **From the shock CSV** (recommended for replication of the paper):

   .. code-block:: python

      from macrostat.models.PichlerEtAl2022DIO import (
          ParametersPichlerEtAl2022DIO,
          ScenariosPichlerEtAl2022DIO,
      )

      params = ParametersPichlerEtAl2022DIO.from_wiod_uk(
          data_dir="path/to/data",
          ihs_dir="path/to/ihs",
          inv_file="path/to/inventories.csv",
      )
      scenarios = ScenariosPichlerEtAl2022DIO.from_shocks_csv(
          parameters=params,
          shock_csv="path/to/shock_scenarios.csv",
      )

   The ``from_shocks_csv`` class method reads the same
   ``shock_scenarios.csv`` used by the original R replication code and
   constructs the full ``(T, N)`` shock tensors (supply shocks, demand
   preference reshuffling, fear-of-infection, and other final demand
   reductions) with the correct timing relative to the lockdown start
   date.

2. **Manually via** ``add_vector_scenario``:

   .. code-block:: python

      scenarios.add_vector_scenario(
          timeseries={
              "SupplyShock": my_supply_tensor,       # shape (T, N)
              "FearOfInfection": my_fear_tensor,      # shape (T, 1)
          },
          name="Custom scenario",
      )
