"""Godley & Lavoie (2006, Chapter 4) Model PCEX2

The macrostat.models.GL06PCEX2 module consists of the following classes

.. autosummary::
    :toctree: models/GL06PCEX2

    GL06PCEX2
    BehaviorGL06PCEX2
    ParametersGL06PCEX2
    ScenariosGL06PCEX2
    VariablesGL06PCEX2
"""

from .behavior import BehaviorGL06PCEX2
from .gl06pcex2 import GL06PCEX2
from .parameters import ParametersGL06PCEX2
from .scenarios import ScenariosGL06PCEX2
from .variables import VariablesGL06PCEX2

__all__ = [
    "GL06PCEX2",
    "BehaviorGL06PCEX2",
    "ParametersGL06PCEX2",
    "VariablesGL06PCEX2",
    "ScenariosGL06PCEX2",
]
