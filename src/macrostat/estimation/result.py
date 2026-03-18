# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Karl Naumann-Woleske
"""
Result dataclass for parameter estimation.

This module provides the EstimationResult dataclass, which is analogous to
scipy.optimize.OptimizeResult but designed for MacroStat models with PyTorch
tensors.
"""

from __future__ import annotations

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

from dataclasses import dataclass

import torch


@dataclass
class EstimationResult:
    """Result from parameter estimation.

    This dataclass is analogous to scipy.optimize.OptimizeResult, designed for
    MacroStat model calibration with PyTorch tensors.

    Attributes
    ----------
    success : bool
        Whether the optimization converged successfully.
    status : int
        Termination status code:
        - 0: ftol criterion satisfied (cost change below tolerance)
        - 1: xtol criterion satisfied (parameter change below tolerance)
        - 2: gtol criterion satisfied (gradient norm below tolerance)
        - -1: maximum iterations reached
        - -2: optimization failed (e.g., singular Jacobian)
    message : str
        Human-readable description of termination reason.
    params : dict[str, torch.Tensor]
        Final parameter values.
    cost : float
        Final cost value (0.5 * ||residuals||²).
    residuals : torch.Tensor
        Final residual vector.
    jacobian : dict[str, torch.Tensor] | None
        Final Jacobian dictionary mapping parameter names to gradient tensors.
        None if Jacobian was not computed at final iteration.
    nfev : int
        Number of function evaluations (residual computations).
    njev : int
        Number of Jacobian evaluations.
    nit : int
        Number of iterations performed.
    optimality : float
        Infinity norm of the gradient (||J^T r||_∞).

    Examples
    --------
    >>> result = EstimationResult(
    ...     success=True,
    ...     status=0,
    ...     message="ftol termination condition satisfied",
    ...     params={"alpha": torch.tensor(0.8)},
    ...     cost=0.0012,
    ...     residuals=torch.zeros(100),
    ...     jacobian={"alpha": torch.randn(100)},
    ...     nfev=45,
    ...     njev=12,
    ...     nit=12,
    ...     optimality=1e-9,
    ... )
    >>> print(f"Converged: {result.success}, Iterations: {result.nit}")
    Converged: True, Iterations: 12

    Notes
    -----
    The cost is defined as 0.5 * ||residuals||² to match the least-squares
    convention used by scipy.optimize.least_squares.
    """

    success: bool
    status: int
    message: str
    params: dict[str, torch.Tensor]
    cost: float
    residuals: torch.Tensor
    jacobian: dict[str, torch.Tensor] | None
    nfev: int
    njev: int
    nit: int
    optimality: float

    def __str__(self):
        """Return human-readable string representation."""
        lines = [
            "EstimationResult:",
            f"  success: {self.success}",
            f"  status: {self.status}",
            f"  message: {self.message}",
            f"  cost: {self.cost:.6e}",
            f"  optimality: {self.optimality:.6e}",
            f"  iterations: {self.nit}",
            f"  function evals: {self.nfev}",
            f"  jacobian evals: {self.njev}",
            f"  parameters: {list(self.params.keys())}",
        ]
        return "\n".join(lines)
