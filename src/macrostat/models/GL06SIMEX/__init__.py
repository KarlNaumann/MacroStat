"""Godley & Lavoie (2006, Chapter 3) Model SIMEX

The macrostat.models.GL06SIMEX module consists of the following classes

.. autosummary::
    :toctree: models/GL06SIMEX

    GL06SIMEX
    behavior
    parameters
    scenarios
    variables

"""

from .behavior import BehaviorGL06SIMEX
from .gl06simex import GL06SIMEX
from .parameters import ParametersGL06SIMEX
from .scenarios import ScenariosGL06SIMEX

__all__ = [
    "GL06SIMEX",
    "BehaviorGL06SIMEX",
    "ParametersGL06SIMEX",
    "VariablesGL06SIMEX",
    "ScenariosGL06SIMEX",
]
