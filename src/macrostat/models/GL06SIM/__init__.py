"""Godley & Lavoie (2006, Chapter 3) Model SIM

The macrostat.models.GL06SIM module consists of the following classes

.. autosummary::
    :toctree: models/GL06SIM

    GL06SIM
    behavior
    parameters
    scenarios
    variables

"""

from .behavior import BehaviorGL06SIM
from .gl06sim import GL06SIM
from .parameters import ParametersGL06SIM
from .scenarios import ScenariosGL06SIM
from .variables import VariablesGL06SIM

__all__ = [
    "GL06SIM",
    "BehaviorGL06SIM",
    "ParametersGL06SIM",
    "VariablesGL06SIM",
    "ScenariosGL06SIM",
]
