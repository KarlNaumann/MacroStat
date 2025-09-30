"""Macro Veronese Passarella's ECO-3IO-PC model

The macrostat.models.ECO3IOPC module consists of the following classes

.. autosummary::
    :toctree: models/ECO3IOPC

    ECO3IOPC
    BehaviorECO3IOPC
    ParametersECO3IOPC
    ScenariosECO3IOPC
    VariablesECO3IOPC
"""

from .behavior import BehaviorECO3IOPC
from .eco3iopc import ECO3IOPC
from .parameters import ParametersECO3IOPC
from .scenarios import ScenariosECO3IOPC
from .variables import VariablesECO3IOPC

__all__ = [
    "ECO3IOPC",
    "BehaviorECO3IOPC",
    "ParametersECO3IOPC",
    "VariablesECO3IOPC",
    "ScenariosECO3IOPC",
]
