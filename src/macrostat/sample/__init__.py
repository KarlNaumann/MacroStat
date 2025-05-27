"""
Parameter space sampling components of the MacroStat model.

The macrostat.sample module consists of the following classes

.. autosummary::
    :toctree: sample

    <Base Sampler>sampler
    <Sobol Sampler>sobol
"""

from .sampler import BaseSampler
from .sobol import SobolSampler

__all__ = ["BaseSampler", "SobolSampler"]
