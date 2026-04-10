"""
Differentiation module for MacroStat

The macrostat.diff module consists of a series of classes that implement both
a numerical and autodiff approach to differentiation of a model's output.

The module consists of the following classes

.. autosummary::
    :toctree: diff

    JacobianBase
    JacobianNumerical
    JacobianAutograd
"""

from .checker import (
    DifferentiabilityReport,
    JacobianComparisonReport,
    ParameterComparison,
    check_model_differentiability,
    compare_jacobian_dicts,
)
from .jacobian_autograd import JacobianAutograd
from .jacobian_base import JacobianBase
from .jacobian_numerical import JacobianNumerical

__all__ = [
    "JacobianBase",
    "JacobianNumerical",
    "JacobianAutograd",
    "DifferentiabilityReport",
    "JacobianComparisonReport",
    "ParameterComparison",
    "check_model_differentiability",
    "compare_jacobian_dicts",
]
