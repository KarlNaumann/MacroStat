# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Karl Naumann-Woleske
"""Loss functions for parameter estimation."""

from __future__ import annotations

from macrostat.estimation.losses.residuals import (
    composite_loss,
    mse_loss,
    weighted_residuals,
)

__all__ = ["mse_loss", "weighted_residuals", "composite_loss"]
