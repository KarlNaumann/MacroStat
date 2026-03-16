# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Karl Naumann-Woleske
"""Tests for EstimationResult dataclass."""

from __future__ import annotations

import torch

from macrostat.estimation import EstimationResult


def test_estimation_result_creation():
    """Test creating an EstimationResult."""
    result = EstimationResult(
        success=True,
        status=0,
        message="ftol termination condition satisfied",
        params={"alpha": torch.tensor(0.8), "beta": torch.tensor(0.3)},
        cost=0.0012,
        residuals=torch.zeros(100),
        jacobian={"alpha": torch.randn(100), "beta": torch.randn(100)},
        nfev=45,
        njev=12,
        nit=12,
        optimality=1e-9,
    )

    assert result.success is True
    assert result.status == 0
    assert result.message == "ftol termination condition satisfied"
    assert "alpha" in result.params
    assert "beta" in result.params
    assert result.cost == 0.0012
    assert result.residuals.shape == (100,)
    assert result.jacobian is not None
    assert "alpha" in result.jacobian
    assert result.nfev == 45
    assert result.njev == 12
    assert result.nit == 12
    assert result.optimality == 1e-9


def test_estimation_result_str():
    """Test string representation of EstimationResult."""
    result = EstimationResult(
        success=True,
        status=1,
        message="xtol termination condition satisfied",
        params={"alpha": torch.tensor(0.8)},
        cost=5.2e-8,
        residuals=torch.zeros(50),
        jacobian=None,
        nfev=30,
        njev=8,
        nit=8,
        optimality=2.3e-7,
    )

    result_str = str(result)
    assert "EstimationResult:" in result_str
    assert "success: True" in result_str
    assert "status: 1" in result_str
    assert "xtol termination condition satisfied" in result_str
    assert "cost: 5.200000e-08" in result_str
    assert "optimality: 2.300000e-07" in result_str
    assert "iterations: 8" in result_str
    assert "function evals: 30" in result_str
    assert "jacobian evals: 8" in result_str
    assert "['alpha']" in result_str


def test_estimation_result_failure():
    """Test EstimationResult for failed optimization."""
    result = EstimationResult(
        success=False,
        status=-1,
        message="maximum iterations reached",
        params={"alpha": torch.tensor(0.5)},
        cost=10.5,
        residuals=torch.ones(100),
        jacobian=None,
        nfev=1000,
        njev=250,
        nit=250,
        optimality=0.05,
    )

    assert result.success is False
    assert result.status == -1
    assert "maximum iterations" in result.message
    assert result.nfev == 1000


def test_estimation_result_none_jacobian():
    """Test EstimationResult with None jacobian."""
    result = EstimationResult(
        success=True,
        status=2,
        message="gtol termination condition satisfied",
        params={"gamma": torch.tensor(0.1)},
        cost=0.0,
        residuals=torch.zeros(10),
        jacobian=None,
        nfev=5,
        njev=2,
        nit=2,
        optimality=1e-12,
    )

    assert result.jacobian is None
    assert result.success is True
