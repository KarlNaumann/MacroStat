"""Godley & Lavoie (2006, Chapter 4) Model PC

The macrostat.models.GL06PC module consists of the following classes

.. autosummary::
    :toctree: models/GL06PC

    GL06PC
    behavior
    parameters
    scenarios
    variables
"""

from .behavior import BehaviorGL06PC
from .gl06pc import GL06PC
from .parameters import ParametersGL06PC
from .scenarios import ScenariosGL06PC
from .variables import VariablesGL06PC

__all__ = [
    "GL06PC",
    "BehaviorGL06PC",
    "ParametersGL06PC",
    "VariablesGL06PC",
    "ScenariosGL06PC",
]
