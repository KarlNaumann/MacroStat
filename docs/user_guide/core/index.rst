========
Core
========

.. _core:

Introduction
============

The `core` module is the foundation of MacroStat models. It provides the basic classes and functions that are used to build the models, taking a separation of concerns approach.
Specifically: the :doc:`model` class is the main class that provides an interface to the macroeconomic model in general. It allows for an easy `simulate` method, and can be saved and loaded. Each model is a container that holds the four main components of a model:

1. :doc:`parameters`: This is the main class that holds the parameters of the model. There are two types of parameters that are stored in this class:
   a. exogenous constants, for example the discount factor or the population growth rate, that are set once and used throughout the model. These constants can also be used in calibration activities.
   b. hyper-parameters, such as the number of timesteps to simulate, or any specific flags that the user might consider
2. :doc:`variables`: This is the main class that holds the variables of the model. Variables are endogenous quantities that are computed by the model. In principle, the variables class manages these values and can then return, amongst other things, a dataframe with the realizations of all the variables across the simulation periods.
3. :doc:`scenarios`: This is the main class that holds the scenarios of the model. Unlike parameters, scenarios are exogenous shocks, such as total factor productivity or government spending, that can vary across timesteps (e.g. they could be thought of as the residuals in matching data). They can also be used, for instance, to compute impulse response functions of the model.
4. :doc:`behavior`: This is the main class that holds the behavior of the model. Behavior is the set of equations that are used to compute the realizations of the endogenous variables of the model given the parameters and scenarios.

Module Components
=================
The following pages contain an overview of the components of the core module and their functionalities.

.. toctree::
   :maxdepth: 2

   Model <model>
   Parameters <parameters>
   Scenarios <scenarios>
   Variables <variables>
   Behavior <behavior>
