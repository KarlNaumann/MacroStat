# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Karl Naumann-Woleske
"""
Parameter estimation module for MacroStat models.

This module provides tools for calibrating MacroStat model parameters to data,
including loss functions and optimization result structures. It is designed to
work with both Levenberg-Marquardt optimization (requiring residual vectors)
and standard PyTorch optimizers (requiring scalar losses).

Classes
-------
EstimationResult
    Dataclass containing optimization results (analogous to scipy.optimize.OptimizeResult).

Functions
---------
mse_loss
    Mean squared error loss between model output and target data.
weighted_residuals
    Weighted residuals for multi-variable calibration.
composite_loss
    Combine multiple loss components (e.g., data fit + regularization).

Examples
--------
Using loss functions with torch.optim.Adam:

>>> from macrostat.models import get_model
>>> from macrostat.estimation import mse_loss
>>> import torch
>>>
>>> model = get_model("GL06SIM")()
>>> target_data = {...}  # Your target data
>>>
>>> def loss_fn(output):
...     return mse_loss(output, target_data, reduction="mean")
>>>
>>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
>>> for epoch in range(100):
...     optimizer.zero_grad()
...     output = model.simulate()
...     loss = loss_fn(output)
...     loss.backward()
...     optimizer.step()

Using loss functions for Jacobian computation (for future LM optimizer):

>>> from macrostat.diff import JacobianAutograd
>>> from macrostat.estimation import mse_loss
>>>
>>> def loss_fn(output):
...     return mse_loss(output, target_data, reduction="none")
>>>
>>> jac = JacobianAutograd(model, scenario=0)
>>> jacobian = jac.compute(loss_fn)
"""

from __future__ import annotations

from macrostat.estimation.lm import LevenbergMarquardt
from macrostat.estimation.losses import (
    composite_loss,
    mse_loss,
    weighted_residuals,
)
from macrostat.estimation.result import EstimationResult

__all__ = [
    "EstimationResult",
    "LevenbergMarquardt",
    "mse_loss",
    "weighted_residuals",
    "composite_loss",
]
