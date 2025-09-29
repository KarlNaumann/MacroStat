"""
Utility functions for the MacroStat model.

The macrostat.util module consists of the following classes

.. autosummary::
    :toctree: util

    autodocs
    batchprocessing
"""

from .autodocs import generate_docs
from .batchprocessing import parallel_processor

__all__ = [
    "generate_docs",
    "parallel_processor",
]
