===============
Behavior PC
===============

This is the documentation of the `behavior` module of the **PC** model from Godley & Lavoie (2006, Chapter 4), detailing the equations of the model.


Initialization
==============

.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.initialize
    :no-index:


Main Simulation Loop
====================

Scenario-determined variables
-----------------------------
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.consumption_government
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.set_interest_rate
    :no-index:

Variables based on prior steps
------------------------------
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.interest_earned_on_bills_household
    :no-index:

Solution of the step
--------------------
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.national_income
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.taxes
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.disposable_income
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.consumption
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.wealth
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.household_bill_holdings
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.household_money_stock
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.central_bank_profits
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.government_bill_issuance
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.central_bank_bill_holdings
    :no-index:
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.central_bank_money_stock
    :no-index:


Computation of the theoretical steady state
===========================================
.. automethod:: macrostat.models.GL06PC.behavior.BehaviorGL06PC.compute_theoretical_steady_state_per_step
    :no-index:
