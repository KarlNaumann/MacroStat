"""
Core components of the MacroStat model.

The macrostat.core module consists of the following classes

.. autosummary::
    :toctree: core

    Behavior
    BoundaryError
    ConstraintError
    ConstraintResolver
    LinearConstraint
    Model
    ParameterLocation
    Parameters
    Scenarios
    Variables
"""

from .behavior import Behavior
from .constraints import (
    ConstraintError,
    ConstraintResolver,
    LinearConstraint,
    ParameterLocation,
)
from .model import Model
from .parameters import BoundaryError, Parameters
from .scenarios import Scenarios
from .variables import Variables

__all__ = [
    "Behavior",
    "BoundaryError",
    "ConstraintError",
    "ConstraintResolver",
    "LinearConstraint",
    "Model",
    "ParameterLocation",
    "Parameters",
    "Scenarios",
    "Variables",
]
