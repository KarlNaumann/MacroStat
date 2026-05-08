"""The macrostat.models module

The macrostat.models module consists of the following classes

.. autosummary::
    :toctree: models

    ECO3IOPC
    GL06INSOUT
    GL06LP
    GL06LP2
    GL06LP3
    GL06PC
    GL06PCEX
    GL06PCEX2
    GL06SIM
    GL06SIMEX
    IOPC
    KirmansAnts
    NK3E
    model_manager

"""

from .model_manager import get_available_models, get_model, get_model_classes

__all__ = [
    "get_available_models",
    "get_model",
    "get_model_classes",
]
