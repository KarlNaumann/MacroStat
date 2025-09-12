"""New Keynesian 3-Equation (NK3E) Model - https://macrosimulation.org/a_new_keynesian_3_equation_model

The macrostat.models.NK3E module consists of the following classes

.. autosummary::
    :toctree: models/NK3E

    NK3E
    behavior
    parameters
    scenarios
    variables
"""

from .behavior import BehaviorNK3E
from .nk3e import NK3E
from .parameters import ParametersNK3E
from .scenarios import ScenariosNK3E
from .variables import VariablesNK3E

__all__ = [
    "NK3E",
    "BehaviorNK3E",
    "ParametersNK3E",
    "VariablesNK3E",
    "ScenariosNK3E",
]
