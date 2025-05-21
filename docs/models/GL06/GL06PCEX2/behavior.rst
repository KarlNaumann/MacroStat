===============
Behavior PCEX2
===============

This is the documentation of the `behavior` module of the **PCEX2** model from Godley & Lavoie (2006, Chapter 4), detailing the equations of the model.


Initialization
==============

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.initialize
    :noindex:



Main Simulation Loop
====================

Scenario-determined variables
-----------------------------

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.consumption_government
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.set_interest_rate
    :noindex:

Variables based on prior steps
------------------------------

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.interest_earned_on_bills_household
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.expected_disposable_income
    :noindex:

Solution of the step
--------------------

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.consumption
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.national_income
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.taxes
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.disposable_income
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.wealth
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.expected_wealth
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.household_bill_demand
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.household_bill_holdings
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.household_money_stock
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.central_bank_profits
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.government_bill_issuance
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.central_bank_bill_holdings
    :noindex:

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.central_bank_money_stock
    :noindex:

Theoretical Steady State Solution
=================================

.. autofunction:: macrostat.models.GL06PCEX2.behavior.BehaviorGL06PCEX2.compute_theoretical_steady_state_per_step
    :noindex:
