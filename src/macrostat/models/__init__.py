"""The macrostat.models module

The macrostat.models module consists of the following classes

.. autosummary::
    :toctree: models

    GL06PC
    GL06SIM
    GL06SIMEX
    model_manager

"""

from .model_manager import get_available_models, get_model, get_model_classes

__all__ = [
    "get_available_models",
    "get_model",
    "get_model_classes",
]
