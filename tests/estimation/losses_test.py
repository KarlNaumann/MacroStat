# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Karl Naumann-Woleske
"""Tests for loss functions."""

from __future__ import annotations

import pytest
import torch

from macrostat.estimation import composite_loss, mse_loss, weighted_residuals


def test_mse_loss_reduction_none():
    """Test mse_loss with reduction='none' returns residual vector."""
    output = {
        "var1": torch.tensor([1.0, 2.0, 3.0]),
        "var2": torch.tensor([4.0, 5.0, 6.0]),
    }
    target = {
        "var1": torch.tensor([1.1, 2.1, 3.1]),
        "var2": torch.tensor([3.9, 5.1, 6.2]),
    }

    loss = mse_loss(output, target, reduction="none")

    # Should return flattened residual vector
    assert loss.ndim == 1
    assert loss.shape[0] == 6  # 3 timesteps * 2 variables

    # Check values: residuals are (output - target)
    expected = torch.tensor([-0.1, -0.1, -0.1, 0.1, -0.1, -0.2])
    assert torch.allclose(loss, expected, atol=1e-6)


def test_mse_loss_reduction_mean():
    """Test mse_loss with reduction='mean' returns scalar."""
    output = {
        "var1": torch.tensor([1.0, 2.0, 3.0]),
    }
    target = {
        "var1": torch.tensor([1.1, 2.1, 3.1]),
    }

    loss = mse_loss(output, target, reduction="mean")

    # Should return scalar
    assert loss.ndim == 0

    # Mean of squared residuals: mean([0.01, 0.01, 0.01]) = 0.01
    expected = 0.01
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)


def test_mse_loss_reduction_sum():
    """Test mse_loss with reduction='sum' returns scalar sum."""
    output = {
        "var1": torch.tensor([1.0, 2.0]),
    }
    target = {
        "var1": torch.tensor([1.1, 2.1]),
    }

    loss = mse_loss(output, target, reduction="sum")

    # Should return scalar
    assert loss.ndim == 0

    # Sum of squared residuals: 0.01 + 0.01 = 0.02
    expected = 0.02
    assert torch.isclose(loss, torch.tensor(expected), atol=1e-6)


def test_mse_loss_variable_selection():
    """Test mse_loss with specific variable selection."""
    output = {
        "var1": torch.tensor([1.0, 2.0]),
        "var2": torch.tensor([3.0, 4.0]),
        "var3": torch.tensor([5.0, 6.0]),
    }
    target = {
        "var1": torch.tensor([1.0, 2.0]),
        "var2": torch.tensor([3.0, 4.0]),
        "var3": torch.tensor([5.0, 6.0]),
    }

    # Select only var1 and var3
    loss = mse_loss(output, target, variables=["var1", "var3"], reduction="none")

    # Should have 4 elements (2 timesteps * 2 variables)
    assert loss.shape[0] == 4

    # All residuals should be zero
    assert torch.allclose(loss, torch.zeros(4))


def test_mse_loss_timestep_selection():
    """Test mse_loss with timestep slice."""
    output = {
        "var1": torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0]),
    }
    target = {
        "var1": torch.tensor([1.1, 2.1, 3.0, 4.0, 5.0]),
    }

    # Use only last 3 timesteps
    loss = mse_loss(output, target, timesteps=slice(-3, None), reduction="none")

    # Should have 3 elements
    assert loss.shape[0] == 3

    # Residuals for timesteps 2, 3, 4: [0.0, 0.0, 0.0]
    assert torch.allclose(loss, torch.zeros(3))


def test_mse_loss_missing_variable_error():
    """Test mse_loss raises error for missing variables."""
    output = {"var1": torch.tensor([1.0])}
    target = {"var2": torch.tensor([2.0])}

    with pytest.raises(ValueError, match="Variables not in output"):
        mse_loss(output, target, variables=["var2"])

    with pytest.raises(ValueError, match="Variables not in target"):
        mse_loss(output, target, variables=["var1"])


def test_mse_loss_invalid_reduction():
    """Test mse_loss raises error for invalid reduction mode."""
    output = {"var1": torch.tensor([1.0])}
    target = {"var1": torch.tensor([1.0])}

    with pytest.raises(ValueError, match="Invalid reduction mode"):
        mse_loss(output, target, reduction="invalid")


def test_weighted_residuals_weight_normalization():
    """Test weighted_residuals normalizes weights correctly."""
    output = {
        "var1": torch.tensor([1.0, 2.0]),
        "var2": torch.tensor([3.0, 4.0]),
    }
    target = {
        "var1": torch.tensor([1.1, 2.1]),
        "var2": torch.tensor([3.2, 4.2]),
    }

    # Weights: var1=10.0, var2=1.0
    # Normalized: var1=10*2/11=1.818, var2=1*2/11=0.182
    weights = {"var1": 10.0, "var2": 1.0}
    loss = weighted_residuals(output, target, weights, reduction="none")

    # Should return 4 elements
    assert loss.shape[0] == 4

    # First two elements (var1) should be weighted more heavily
    # sqrt(1.818) * (-0.1) ≈ -0.135
    # Last two elements (var2) should be weighted less
    # sqrt(0.182) * (-0.2) ≈ -0.085
    assert abs(loss[0]) > abs(loss[2])


def test_weighted_residuals_default_weights():
    """Test weighted_residuals with default weights (all 1.0)."""
    output = {
        "var1": torch.tensor([1.0, 2.0]),
        "var2": torch.tensor([3.0, 4.0]),
    }
    target = {
        "var1": torch.tensor([1.1, 2.1]),
        "var2": torch.tensor([3.1, 4.1]),
    }

    # No weights specified - should default to 1.0 for all
    weights = {}
    loss = weighted_residuals(output, target, weights, reduction="none")

    # With equal weights, should be similar to mse_loss
    loss_unweighted = mse_loss(output, target, reduction="none")
    assert torch.allclose(loss, loss_unweighted)


def test_weighted_residuals_reduction_modes():
    """Test weighted_residuals with different reduction modes."""
    output = {"var1": torch.tensor([1.0, 2.0])}
    target = {"var1": torch.tensor([1.1, 2.1])}
    weights = {"var1": 1.0}

    loss_none = weighted_residuals(output, target, weights, reduction="none")
    loss_mean = weighted_residuals(output, target, weights, reduction="mean")
    loss_sum = weighted_residuals(output, target, weights, reduction="sum")

    assert loss_none.ndim == 1
    assert loss_mean.ndim == 0
    assert loss_sum.ndim == 0

    # Verify relationship: sum = mean * n_elements
    assert torch.isclose(loss_sum, loss_mean * loss_none.numel())


def test_composite_loss_simple():
    """Test composite_loss with two simple losses."""
    output = {"var1": torch.tensor([1.0, 2.0])}
    target1 = {"var1": torch.tensor([1.1, 2.1])}
    target2 = {"var1": torch.tensor([0.9, 1.9])}

    def loss_fn1(out, tgt):
        return mse_loss(out, tgt, reduction="none")

    def loss_fn2(out, tgt):
        return mse_loss(out, tgt, reduction="none")

    loss = composite_loss(
        output,
        targets=[target1, target2],
        loss_fns=[loss_fn1, loss_fn2],
        reduction="none",
    )

    # Should concatenate both residual vectors (2 + 2 = 4 elements)
    assert loss.shape[0] == 4


def test_composite_loss_with_weights():
    """Test composite_loss with custom weights."""
    output = {"var1": torch.tensor([1.0])}
    target1 = {"var1": torch.tensor([1.1])}
    target2 = {"var1": torch.tensor([0.9])}

    def loss_fn1(out, tgt):
        return mse_loss(out, tgt, reduction="none")

    def loss_fn2(out, tgt):
        return mse_loss(out, tgt, reduction="none")

    # Weight first loss 10x more than second
    loss = composite_loss(
        output,
        targets=[target1, target2],
        loss_fns=[loss_fn1, loss_fn2],
        loss_weights=[10.0, 1.0],
        reduction="none",
    )

    # Weights normalized: [10*2/11, 1*2/11] = [1.818, 0.182]
    # First residual: sqrt(1.818) * (-0.1) ≈ -0.135
    # Second residual: sqrt(0.182) * (0.1) ≈ 0.043
    assert abs(loss[0]) > abs(loss[1])


def test_composite_loss_scalar_reduction():
    """Test composite_loss with scalar reduction modes."""
    output = {"var1": torch.tensor([1.0, 2.0])}
    target1 = {"var1": torch.tensor([1.1, 2.1])}
    target2 = {"var1": torch.tensor([0.9, 1.9])}

    def loss_fn1(out, tgt):
        return mse_loss(out, tgt, reduction="none")

    def loss_fn2(out, tgt):
        return mse_loss(out, tgt, reduction="none")

    loss_mean = composite_loss(
        output,
        targets=[target1, target2],
        loss_fns=[loss_fn1, loss_fn2],
        reduction="mean",
    )

    loss_sum = composite_loss(
        output,
        targets=[target1, target2],
        loss_fns=[loss_fn1, loss_fn2],
        reduction="sum",
    )

    assert loss_mean.ndim == 0
    assert loss_sum.ndim == 0


def test_composite_loss_validation_errors():
    """Test composite_loss validation errors."""
    output = {"var1": torch.tensor([1.0])}
    target = {"var1": torch.tensor([1.0])}

    def dummy_loss(out, tgt):
        return mse_loss(out, tgt, reduction="none")

    # Mismatched lengths
    with pytest.raises(ValueError, match="loss_fns and loss_weights"):
        composite_loss(
            output,
            targets=[target],
            loss_fns=[dummy_loss],
            loss_weights=[1.0, 2.0],
            reduction="none",
        )

    with pytest.raises(ValueError, match="targets and loss_fns"):
        composite_loss(
            output,
            targets=[target, target],
            loss_fns=[dummy_loss],
            reduction="none",
        )


def test_composite_loss_non_1d_error():
    """Test composite_loss raises error for non-1D losses with reduction='none'."""
    output = {"var1": torch.tensor([1.0])}
    target = {"var1": torch.tensor([1.0])}

    # This loss function returns a scalar (0D tensor)
    def bad_loss(out, tgt):
        return torch.tensor(0.5)

    with pytest.raises(ValueError, match="must return 1D tensors"):
        composite_loss(
            output,
            targets=[target],
            loss_fns=[bad_loss],
            reduction="none",
        )
