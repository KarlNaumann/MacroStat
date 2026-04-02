====================
Scenarios GL06INSOUT
====================

The following scenarios are available by default for the GL06INSOUT model,
corresponding to the experiments in Chapter 10 (2nd ed.) / Chapter 7 (1st ed.)
of Godley & Lavoie (2006). The R ``sfcr`` reference names are given in
parentheses.

Baseline
========

The baseline scenario has the following exogenous values:

- Real government spending :math:`g = 25`
- Bill rate :math:`r_b = 0.023`
- Bond yield :math:`r_{bl} = 0.027`

All shocks fire at the scenario trigger period (default: period 50).


Scenario 1: Higher Target Inventory Ratio (INSOUT1)
=====================================================

An increase in the target inventory-to-sales ratio baseline from
:math:`\sigma_0 = 0.3612` to :math:`\sigma_0 = 0.4`. Firms desire to hold
a larger buffer of inventories relative to expected sales, which raises
inventory investment and loan demand.


Scenario 2: Higher Government Spending (INSOUT2)
==================================================

An increase in real government spending from :math:`g = 25` to :math:`g = 30`.
This is the standard Keynesian fiscal expansion experiment, raising output,
income, and government debt.


Scenario 3: Higher Reserve Requirements (INSOUT3)
===================================================

An increase in compulsory reserve ratios from :math:`\rho_1 = \rho_2 = 0.1`
to :math:`\rho_1 = \rho_2 = 0.2`. Higher reserve requirements tie up more
bank funds in non-interest-bearing reserves, reducing the bank's bill
holdings and potentially triggering advances from the central bank.


Scenario 4: Wider Bank Liquidity Corridor (INSOUT4)
=====================================================

An increase in the acceptable bank liquidity ratio bounds from
:math:`bot = 0.02, top = 0.06` to :math:`bot = 0.20, top = 0.24`. The
wider corridor raises the bank's target bill-to-deposit ratio, leading
to higher bill holdings and reduced need for central bank advances.


Scenario 5: Lower Consumption Propensity (INSOUT5)
====================================================

A decrease in the propensity to consume out of income from
:math:`\alpha_1 = 0.95` to :math:`\alpha_1 = 0.80`. This paradox-of-thrift
experiment reduces consumption on impact but leads to wealth accumulation
that partially offsets the initial decline.


Scenario 6: Exogenous Increase in Inflation via Wage Target (INSOUT6)
======================================================================

An increase in the real wage target constant from
:math:`\Omega_0 = -0.32549` to :math:`\Omega_0 = -0.2`. Workers demand a
higher real wage, triggering a wage-price spiral that generates persistent
inflation until the economy reaches a new steady state.


Scenario 7: Combined Wage Push and Interest Rate Increase (INSOUT7)
====================================================================

A simultaneous increase in the real wage target (same as Scenario 6) and
an increase in interest rates (:math:`r_b = 0.023 \to 0.03`,
:math:`r_{bl} = 0.027 \to 0.039`). This stagflation experiment shows the
interaction between cost-push inflation and contractionary monetary policy.
