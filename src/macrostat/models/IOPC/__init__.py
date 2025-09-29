"""Macro Veronese Passarella's 3IO-PC model

The macrostat.models.IOPC module consists of the following classes

.. autosummary::
    :toctree: models/IOPC

    IOPC
    BehaviorIOPC
    ParametersIOPC
    ScenariosIOPC
    VariablesIOPC
"""

from .behavior import BehaviorIOPC
from .iopc import IOPC
from .parameters import ParametersIOPC
from .scenarios import ScenariosIOPC
from .variables import VariablesIOPC

__all__ = [
    "IOPC",
    "BehaviorIOPC",
    "ParametersIOPC",
    "VariablesIOPC",
    "ScenariosIOPC",
]
