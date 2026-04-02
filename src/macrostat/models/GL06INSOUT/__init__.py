"""Godley & Lavoie (2006, Chapter 7) Model INSOUT

The macrostat.models.GL06INSOUT module consists of the following classes

.. autosummary::
    :toctree: models/GL06INSOUT

    GL06INSOUT
    BehaviorGL06INSOUT
    ParametersGL06INSOUT
    ScenariosGL06INSOUT
    VariablesGL06INSOUT
"""

from .behavior import BehaviorGL06INSOUT
from .gl06insout import GL06INSOUT
from .parameters import ParametersGL06INSOUT
from .scenarios import ScenariosGL06INSOUT
from .variables import VariablesGL06INSOUT

__all__ = [
    "GL06INSOUT",
    "BehaviorGL06INSOUT",
    "ParametersGL06INSOUT",
    "VariablesGL06INSOUT",
    "ScenariosGL06INSOUT",
]
