===============
Behavior PCEX
===============

This is the documentation of the `behavior` module of the **PCEX** model from Godley & Lavoie (2006, Chapter 4), detailing the equations of the model.


Initialization
==============

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.initialize
    :noindex:



Main Simulation Loop
====================

Scenario-determined variables
-----------------------------

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.consumption_government
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.set_interest_rate
    :noindex:

Variables based on prior steps
------------------------------

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.interest_earned_on_bills_household
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.expected_disposable_income
    :noindex:

Solution of the step
--------------------

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.consumption
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.national_income
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.taxes
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.disposable_income
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.wealth
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.expected_wealth
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.household_bill_demand
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.household_bill_holdings
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.household_money_stock
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.central_bank_profits
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.government_bill_issuance
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.central_bank_bill_holdings
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.central_bank_money_stock
    :noindex:

Theoretical Steady State Solution
=================================

.. autofunction:: macrostat.models.GL06PCEX.behavior.BehaviorGL06PCEX.compute_theoretical_steady_state_per_step
    :noindex:
