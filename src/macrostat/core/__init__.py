"""
Core components of the MacroStat model.

The macrostat.core module consists of the following classes

.. autosummary::
    :toctree: core

    BoundaryError
    Behavior
    Model
    Parameters
    Scenarios
    Variables
"""

from .behavior import Behavior
from .constraints import LinearConstraint
from .model import Model
from .parameters import BoundaryError, Parameters
from .scenarios import Scenarios
from .variables import Variables

__all__ = [
    "Behavior",
    "BoundaryError",
    "LinearConstraint",
    "Model",
    "Parameters",
    "Scenarios",
    "Variables",
]
