# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Karl Naumann-Woleske
"""Tests for Levenberg-Marquardt optimizer."""

from __future__ import annotations

import torch

from macrostat.estimation import LevenbergMarquardt, mse_loss
from macrostat.models import get_model


def test_lm_initialization():
    """Test LM optimizer initialization."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)  # Dummy loss

    lm = LevenbergMarquardt(
        model,
        loss_fn,
        ftol=1e-6,
        xtol=1e-6,
        gtol=1e-6,
        max_nfev=500,
        verbose=0,
    )

    assert lm.ftol == 1e-6
    assert lm.xtol == 1e-6
    assert lm.gtol == 1e-6
    assert lm.max_nfev == 500
    assert lm.nfev == 0
    assert lm.njev == 0
    assert lm.nit == 0


def test_lm_get_parameters():
    """Test parameter extraction from behavior."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn)

    params = lm._get_parameters()

    # GL06SIM has 3 parameters
    assert len(params) == 3
    assert "PropensityToConsumeIncome" in params
    assert "PropensityToConsumeSavings" in params
    assert "TaxRate" in params

    # Check values are tensors
    for value in params.values():
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0  # Scalar


def test_lm_dict_to_matrix():
    """Test Jacobian dict to matrix conversion."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn)

    # Create dummy Jacobian dict
    params = {"alpha": torch.tensor(0.6), "beta": torch.tensor(0.4)}

    jac_dict = {
        "alpha": torch.randn(10),  # 10 residuals
        "beta": torch.randn(10),
    }

    # Convert to matrix
    J = lm._dict_to_matrix(jac_dict, params)

    # Check shape
    assert J.shape == (10, 2)  # (n_residuals, n_params)

    # Check ordering matches params
    assert torch.allclose(J[:, 0], jac_dict["alpha"])
    assert torch.allclose(J[:, 1], jac_dict["beta"])


def test_lm_dict_vector_conversion():
    """Test dict <-> vector conversion."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn)

    params = {
        "alpha": torch.tensor(0.6),
        "beta": torch.tensor(0.4),
        "gamma": torch.tensor(0.2),
    }

    # Dict to vector
    vec = lm._dict_to_vector(params, params)
    assert vec.shape == (3,)
    assert abs(vec[0].item() - 0.6) < 1e-6
    assert abs(vec[1].item() - 0.4) < 1e-6
    assert abs(vec[2].item() - 0.2) < 1e-6

    # Vector to dict
    vec_new = torch.tensor([0.7, 0.3, 0.1])
    params_new = lm._vector_to_dict(vec_new, params)
    assert len(params_new) == 3
    assert abs(params_new["alpha"].item() - 0.7) < 1e-6
    assert abs(params_new["beta"].item() - 0.3) < 1e-6
    assert abs(params_new["gamma"].item() - 0.1) < 1e-6


def test_lm_solve_step_marquardt():
    """Test LM step solution with Marquardt scaling."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn, scaling="marquardt")

    # Create simple test case
    params = {"alpha": torch.tensor(0.6), "beta": torch.tensor(0.4)}

    J = torch.randn(10, 2)
    residuals = torch.randn(10)
    damping = 0.1

    # Solve step
    step = lm._solve_step(J, residuals, damping, params)

    # Check step is a dict with correct keys
    assert set(step.keys()) == {"alpha", "beta"}

    # Check step values are tensors
    for value in step.values():
        assert isinstance(value, torch.Tensor)


def test_lm_solve_step_identity():
    """Test LM step solution with identity scaling."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn, scaling="identity")

    params = {"alpha": torch.tensor(0.6)}
    J = torch.randn(10, 1)
    residuals = torch.randn(10)
    damping = 0.1

    step = lm._solve_step(J, residuals, damping, params)

    assert "alpha" in step


def test_lm_add_step():
    """Test adding step to parameters."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn)

    params = {
        "alpha": torch.tensor(0.6),
        "beta": torch.tensor(0.4),
    }

    step = {
        "alpha": torch.tensor(0.1),
        "beta": torch.tensor(-0.05),
    }

    params_new = lm._add_step(params, step)

    assert abs(params_new["alpha"].item() - 0.7) < 1e-6
    assert abs(params_new["beta"].item() - 0.35) < 1e-6

    # Original params unchanged
    assert abs(params["alpha"].item() - 0.6) < 1e-6
    assert abs(params["beta"].item() - 0.4) < 1e-6


def test_lm_check_termination_ftol():
    """Test ftol termination criterion."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn, ftol=1e-6)

    params = {"alpha": torch.tensor(0.6)}
    step = {"alpha": torch.tensor(0.1)}
    J = torch.randn(10, 1)
    residuals = torch.randn(10)

    # Small cost change (below ftol)
    cost = 1.0
    cost_new = 1.0 - 1e-7

    converged, status, message = lm._check_termination(
        cost, cost_new, step, params, J, residuals, accepted=True
    )

    assert converged
    assert status == 0
    assert "ftol" in message


def test_lm_check_termination_xtol():
    """Test xtol termination criterion."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn, xtol=1e-6)

    params = {"alpha": torch.tensor(0.6)}
    step = {"alpha": torch.tensor(1e-8)}  # Very small step
    J = torch.randn(10, 1)
    residuals = torch.randn(10)

    cost = 1.0
    cost_new = 0.99  # Large cost change (ftol won't trigger)

    converged, status, message = lm._check_termination(
        cost, cost_new, step, params, J, residuals, accepted=True
    )

    assert converged
    assert status == 1
    assert "xtol" in message


def test_lm_check_termination_gtol():
    """Test gtol termination criterion."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn, gtol=1e-6)

    params = {"alpha": torch.tensor(0.6)}
    step = {"alpha": torch.tensor(0.1)}
    residuals = torch.randn(10) * 1e-8  # Very small residuals
    J = torch.randn(10, 1)

    cost = 1.0
    cost_new = 0.99

    converged, status, message = lm._check_termination(
        cost, cost_new, step, params, J, residuals, accepted=True
    )

    assert converged
    assert status == 2
    assert "gtol" in message


def test_lm_check_termination_not_converged():
    """Test when no termination criterion is met."""
    model = get_model("GL06SIM")()

    def loss_fn(output):
        return torch.zeros(10)

    lm = LevenbergMarquardt(model, loss_fn, ftol=1e-8, xtol=1e-8, gtol=1e-8)

    params = {"alpha": torch.tensor(0.6)}
    step = {"alpha": torch.tensor(0.1)}
    J = torch.randn(10, 1)
    residuals = torch.randn(10)

    cost = 1.0
    cost_new = 0.5  # Large change

    converged, status, message = lm._check_termination(
        cost, cost_new, step, params, J, residuals, accepted=True
    )

    assert not converged


def test_lm_optimization_simple():
    """Test LM optimization on simple problem with GL06SIM."""
    model = get_model("GL06SIM")()

    # Generate synthetic target data
    model.parameters["PropensityToConsumeIncome"] = 0.6
    model.parameters["PropensityToConsumeSavings"] = 0.4

    with torch.no_grad():
        target_output = model.simulate(scenario=0)

    # Perturb parameters
    model.parameters["PropensityToConsumeIncome"] = 0.7
    model.parameters["PropensityToConsumeSavings"] = 0.3

    # Define loss function
    def loss_fn(output):
        return mse_loss(
            output,
            target_output,
            variables=["ConsumptionDemand", "DisposableIncome"],
            timesteps=slice(5, 40),
            reduction="none",
        )

    # Optimize
    lm = LevenbergMarquardt(
        model, loss_fn, max_nfev=100, ftol=1e-6, xtol=1e-6, gtol=1e-6, verbose=0
    )

    result = lm.optimize()

    # Check basic properties
    assert isinstance(result.success, bool)
    assert result.nfev > 0
    assert result.njev > 0
    assert result.nit > 0
    assert result.cost >= 0
    assert result.residuals.ndim == 1

    # If converged, cost should be small
    if result.success:
        assert result.cost < 1.0  # Should be much smaller for perfect data


def test_lm_zero_residuals():
    """Test LM with zero residuals (already at optimum)."""
    model = get_model("GL06SIM")()

    # Generate target data with current parameters
    with torch.no_grad():
        target_output = model.simulate(scenario=0)

    # Don't perturb parameters - already at optimum

    def loss_fn(output):
        return mse_loss(
            output,
            target_output,
            variables=["ConsumptionDemand"],
            timesteps=slice(10, 30),
            reduction="none",
        )

    lm = LevenbergMarquardt(model, loss_fn, max_nfev=50, verbose=0)

    result = lm.optimize()

    # Should have essentially zero cost (may not converge via normal criteria
    # when starting exactly at optimum due to numerical issues)
    assert result.cost < 1e-10  # Essentially zero
    assert result.optimality < 1e-6  # Gradient essentially zero


def test_lm_result_structure():
    """Test that LM returns proper EstimationResult."""
    model = get_model("GL06SIM")()

    with torch.no_grad():
        target_output = model.simulate(scenario=0)

    def loss_fn(output):
        return mse_loss(
            output, target_output, variables=["ConsumptionDemand"], reduction="none"
        )

    lm = LevenbergMarquardt(model, loss_fn, max_nfev=10, verbose=0)

    result = lm.optimize()

    # Check all required fields
    assert hasattr(result, "success")
    assert hasattr(result, "status")
    assert hasattr(result, "message")
    assert hasattr(result, "params")
    assert hasattr(result, "cost")
    assert hasattr(result, "residuals")
    assert hasattr(result, "jacobian")
    assert hasattr(result, "nfev")
    assert hasattr(result, "njev")
    assert hasattr(result, "nit")
    assert hasattr(result, "optimality")

    # Check types
    assert isinstance(result.params, dict)
    assert isinstance(result.jacobian, dict)
    assert isinstance(result.residuals, torch.Tensor)
