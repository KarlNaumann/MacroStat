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
from .model import Model
from .parameters import BoundaryError, Parameters
from .scenarios import Scenarios
from .variables import Variables

__all__ = ["Behavior", "Parameters", "Model", "Scenarios", "Variables", "BoundaryError"]
