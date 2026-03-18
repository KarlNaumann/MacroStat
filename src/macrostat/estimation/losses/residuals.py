# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Karl Naumann-Woleske
"""
Loss functions for MacroStat parameter estimation.

This module provides loss functions compatible with both Levenberg-Marquardt
(requiring residual vectors) and torch.optim optimizers (requiring scalars).
"""

from __future__ import annotations

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

from typing import Callable, Literal

import torch


def mse_loss(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    variables: list[str] | None = None,
    timesteps: slice | None = None,
    reduction: Literal["mean", "sum", "none"] = "none",
) -> torch.Tensor:
    """Mean squared error loss between model output and target data.

    Parameters
    ----------
    output : dict[str, torch.Tensor]
        Model output from Behavior.forward(), mapping variable names to tensors.
    target : dict[str, torch.Tensor]
        Target values with the same structure as output.
    variables : list[str], optional
        Variables to include in the loss. If None, uses all common keys between
        output and target.
    timesteps : slice, optional
        Time selection for computing loss (e.g., slice(-20, None) for last 20
        timesteps). If None, uses all timesteps.
    reduction : {"mean", "sum", "none"}
        Reduction mode:
        - "none": return 1D residual vector (for Levenberg-Marquardt)
        - "mean": return scalar mean squared error (for torch.optim)
        - "sum": return scalar sum of squared errors

    Returns
    -------
    torch.Tensor
        If reduction="none": 1D residual vector of shape (n_elements,)
        Otherwise: scalar tensor

    Examples
    --------
    >>> # For Levenberg-Marquardt (returns residual vector)
    >>> def loss_fn(output):
    ...     return mse_loss(output, target_data, reduction="none")
    >>> from macrostat.diff import JacobianAutograd
    >>> jac = JacobianAutograd(model, scenario=0)
    >>> jacobian = jac.compute(loss_fn)

    >>> # For torch.optim (returns scalar)
    >>> def loss_fn(output):
    ...     return mse_loss(output, target_data, reduction="mean")
    >>> optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    >>> loss = loss_fn(model.simulate())
    >>> loss.backward()

    Notes
    -----
    The residual vector (reduction="none") is computed as the flattened
    concatenation of (output[var] - target[var]) for all selected variables
    and timesteps.
    """
    # Determine which variables to use
    if variables is None:
        # Use all common keys
        variables = sorted(set(output.keys()) & set(target.keys()))
    else:
        # Validate that all requested variables are present
        missing_output = set(variables) - set(output.keys())
        missing_target = set(variables) - set(target.keys())
        if missing_output:
            raise ValueError(f"Variables not in output: {missing_output}")
        if missing_target:
            raise ValueError(f"Variables not in target: {missing_target}")

    # Compute residuals for each variable
    residuals = []
    for var in variables:
        out = output[var]
        tgt = target[var]

        # Apply timestep selection
        if timesteps is not None:
            out = out[timesteps]
            tgt = tgt[timesteps]

        # Compute residual (output - target)
        resid = out - tgt
        residuals.append(resid.flatten())

    # Concatenate all residuals into a single vector
    residual_vector = torch.cat(residuals)

    # Apply reduction
    if reduction == "none":
        return residual_vector
    elif reduction == "sum":
        return (residual_vector**2).sum()
    elif reduction == "mean":
        return (residual_vector**2).mean()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def weighted_residuals(
    output: dict[str, torch.Tensor],
    target: dict[str, torch.Tensor],
    weights: dict[str, float],
    variables: list[str] | None = None,
    timesteps: slice | None = None,
    reduction: Literal["mean", "sum", "none"] = "none",
) -> torch.Tensor:
    """Weighted residuals for multi-variable calibration with different scales.

    This function is useful when calibrating to multiple variables with different
    units or magnitudes. Weights allow balancing the contribution of each variable
    to the loss.

    Parameters
    ----------
    output : dict[str, torch.Tensor]
        Model output from Behavior.forward().
    target : dict[str, torch.Tensor]
        Target values with the same structure as output.
    weights : dict[str, float]
        Weights for each variable. Variables not in weights dict default to 1.0.
        Weights are normalized so they sum to the number of variables.
    variables : list[str], optional
        Variables to include. If None, uses all common keys.
    timesteps : slice, optional
        Time selection (e.g., slice(-20, None) for last 20 timesteps).
    reduction : {"mean", "sum", "none"}
        Reduction mode (see mse_loss for details).

    Returns
    -------
    torch.Tensor
        If reduction="none": 1D weighted residual vector
        Otherwise: scalar

    Examples
    --------
    >>> # Weight GDP heavily, consumption moderately
    >>> weights = {"GDP": 10.0, "Consumption": 1.0}
    >>> def loss_fn(output):
    ...     return weighted_residuals(
    ...         output, target_data, weights, reduction="none"
    ...     )

    Notes
    -----
    Weights are normalized to sum to the number of variables, so the average
    weight is 1.0. This ensures that the loss magnitude is comparable to
    unweighted losses.

    The weighted residual for each variable is computed as:
    weighted_resid[var] = sqrt(normalized_weight[var]) * (output[var] - target[var])
    """
    # Determine which variables to use
    if variables is None:
        variables = sorted(set(output.keys()) & set(target.keys()))
    else:
        missing_output = set(variables) - set(output.keys())
        missing_target = set(variables) - set(target.keys())
        if missing_output:
            raise ValueError(f"Variables not in output: {missing_output}")
        if missing_target:
            raise ValueError(f"Variables not in target: {missing_target}")

    # Normalize weights so they sum to the number of variables
    raw_weights = {var: weights.get(var, 1.0) for var in variables}
    total_weight = sum(raw_weights.values())
    normalized_weights = {
        var: w * len(variables) / total_weight for var, w in raw_weights.items()
    }

    # Compute weighted residuals for each variable
    residuals = []
    for var in variables:
        out = output[var]
        tgt = target[var]

        # Apply timestep selection
        if timesteps is not None:
            out = out[timesteps]
            tgt = tgt[timesteps]

        # Compute weighted residual: sqrt(weight) * (output - target)
        # We use sqrt(weight) so that the squared residuals have the desired weight
        weight = normalized_weights[var]
        resid = torch.sqrt(torch.tensor(weight)) * (out - tgt)
        residuals.append(resid.flatten())

    # Concatenate all residuals into a single vector
    residual_vector = torch.cat(residuals)

    # Apply reduction
    if reduction == "none":
        return residual_vector
    elif reduction == "sum":
        return (residual_vector**2).sum()
    elif reduction == "mean":
        return (residual_vector**2).mean()
    else:
        raise ValueError(f"Invalid reduction mode: {reduction}")


def composite_loss(
    output: dict[str, torch.Tensor],
    targets: list[dict[str, torch.Tensor]],
    loss_fns: list[Callable[[dict[str, torch.Tensor]], torch.Tensor]],
    loss_weights: list[float] | None = None,
    reduction: Literal["mean", "sum", "none"] = "none",
) -> torch.Tensor:
    """Combine multiple loss functions (e.g., data fit + regularization).

    This function allows combining different loss components, such as:
    - Data fitting losses for different time periods
    - Data fitting + parameter regularization
    - Multiple target datasets with different variables

    Parameters
    ----------
    output : dict[str, torch.Tensor]
        Model output from Behavior.forward().
    targets : list[dict[str, torch.Tensor]]
        List of target dictionaries, one for each loss function.
        If a loss function doesn't use the target, pass None for that element.
    loss_fns : list[Callable]
        List of loss functions. Each should accept (output, target) and return
        a tensor (residual vector if reduction="none", scalar otherwise).
    loss_weights : list[float], optional
        Weights for each loss component. If None, all losses weighted equally.
        Weights are normalized to sum to len(loss_fns).
    reduction : {"mean", "sum", "none"}
        Reduction mode for the combined loss.

    Returns
    -------
    torch.Tensor
        If reduction="none": 1D concatenated residual vector
        Otherwise: scalar

    Examples
    --------
    >>> # Combine data fit loss + regularization
    >>> def data_loss(output, target):
    ...     return mse_loss(output, target, reduction="none")
    >>> def reg_loss(output, _):
    ...     # Penalize large deviations from prior
    ...     return (output["alpha"] - 0.5) * 10.0  # scalar penalty
    >>> loss_fn = lambda out: composite_loss(
    ...     out,
    ...     targets=[target_data, None],
    ...     loss_fns=[data_loss, reg_loss],
    ...     loss_weights=[1.0, 0.1],
    ...     reduction="none",
    ... )

    Notes
    -----
    When reduction="none", all loss functions must return 1D tensors (residual
    vectors). The composite loss is the concatenation of these vectors, weighted
    by sqrt(normalized_weight) to preserve the weighting when squared.

    When reduction="mean" or "sum", loss functions can return scalars or tensors.
    The composite loss is the weighted sum of the individual losses.
    """
    if loss_weights is None:
        loss_weights = [1.0] * len(loss_fns)

    if len(loss_fns) != len(loss_weights):
        raise ValueError("loss_fns and loss_weights must have the same length")

    if len(targets) != len(loss_fns):
        raise ValueError("targets and loss_fns must have the same length")

    # Normalize weights
    total_weight = sum(loss_weights)
    normalized_weights = [w * len(loss_fns) / total_weight for w in loss_weights]

    if reduction == "none":
        # Concatenate weighted residual vectors
        residuals = []
        for target, loss_fn, weight in zip(targets, loss_fns, normalized_weights):
            loss = loss_fn(output, target)
            if loss.ndim != 1:
                raise ValueError(
                    f"With reduction='none', loss functions must return 1D tensors, "
                    f"got shape {loss.shape}"
                )
            # Weight by sqrt(weight) so that squared residuals have the desired weight
            weighted_loss = torch.sqrt(torch.tensor(weight)) * loss
            residuals.append(weighted_loss)
        return torch.cat(residuals)
    else:
        # Compute weighted sum of losses
        total_loss = 0.0
        for target, loss_fn, weight in zip(targets, loss_fns, normalized_weights):
            loss = loss_fn(output, target)
            # For scalar reduction, we just add weighted losses
            if reduction == "sum":
                total_loss = total_loss + weight * loss.sum()
            elif reduction == "mean":
                total_loss = total_loss + weight * loss.mean()
        return total_loss
