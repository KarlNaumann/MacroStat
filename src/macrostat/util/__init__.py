"""
Utility functions for the MacroStat model.

The macrostat.util module consists of the following classes

.. autosummary::
    :toctree: util

    latex_model_documentation
    batchprocessing
"""

from .batchprocessing import parallel_processor
from .latex_model_documentation import generate_latex_documentation

__all__ = [
    "generate_latex_documentation",
    "parallel_processor",
]
