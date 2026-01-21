"""
Simple 2D linear model for Jacobian testing.

The macrostat.models.LINEAR2D module consists of the following classes

.. autosummary::
    :toctree: models/LINEAR2D

    LINEAR2D
    BehaviorLINEAR2D
    ParametersLINEAR2D
    ScenariosLINEAR2D
    VariablesLINEAR2D
"""

from .behavior import BehaviorLINEAR2D
from .linear2d import LINEAR2D
from .parameters import ParametersLINEAR2D
from .scenarios import ScenariosLINEAR2D
from .variables import VariablesLINEAR2D

__all__ = [
    "LINEAR2D",
    "BehaviorLINEAR2D",
    "ParametersLINEAR2D",
    "VariablesLINEAR2D",
    "ScenariosLINEAR2D",
]
