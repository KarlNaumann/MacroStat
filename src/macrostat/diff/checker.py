"""
High-level differentiability checks for MacroStat models.

This module provides a small API to compare:

- Reverse-mode vs forward-mode autograd Jacobians, and
- Autograd vs numerical finite-difference Jacobians

for a user-specified scalar loss function of model outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Literal, Optional

import torch

from macrostat.diff.jacobian_autograd import JacobianAutograd
from macrostat.diff.jacobian_numerical import JacobianNumerical

LossFn = Callable[[Dict[str, torch.Tensor]], torch.Tensor]


@dataclass
class DifferentiabilityReport:
    """Summary of differentiability checks."""

    passed: bool
    nan_or_inf: bool
    fwd_vs_rev_ok: Optional[bool]
    autodiff_vs_numerical_ok: Optional[bool]
    max_abs_diff_fwd_rev: Optional[float]
    max_abs_diff_autodiff_num: Optional[float]
    rel_err_fwd_rev: Optional[float]
    rel_err_autodiff_num: Optional[float]
    details: Dict[str, dict]

    def summary(self) -> str:
        """Return a short human-readable summary."""
        lines = []
        lines.append(f"Passed: {self.passed}")
        lines.append(f"NaN/Inf gradients: {self.nan_or_inf}")
        if self.fwd_vs_rev_ok is not None:
            lines.append(
                f"Forward vs reverse: rel≈{self.rel_err_fwd_rev:.3e} "
                f"(max abs diff={self.max_abs_diff_fwd_rev})"
            )
        if self.autodiff_vs_numerical_ok is not None:
            lines.append(
                f"Autograd vs numerical: rel≈{self.rel_err_autodiff_num:.3e} "
                f"(max abs diff={self.max_abs_diff_autodiff_num})"
            )
        return "\n".join(lines)


def _max_abs_diff_dict(
    a: Dict[str, torch.Tensor],
    b: Dict[str, torch.Tensor],
) -> float:
    """Compute the maximum absolute difference between two grad dicts."""
    max_diff = 0.0
    for name in a.keys() & b.keys():
        diff = (a[name] - b[name]).abs().max().item()
        max_diff = max(max_diff, float(diff))
    return max_diff


def _max_abs_val_dict(
    grads: Dict[str, torch.Tensor],
) -> float:
    """Maximum absolute value across all gradients in a dict."""
    max_val = 0.0
    for g in grads.values():
        if g.numel() == 0:
            continue
        val = g.abs().max().item()
        max_val = max(max_val, float(val))
    return max_val


def _has_nan_or_inf(grads: Dict[str, torch.Tensor]) -> bool:
    for g in grads.values():
        if not torch.isfinite(g).all():
            return True
    return False


def check_model_differentiability(
    model,
    loss_fn: LossFn,
    scenario: int | str = 0,
    target: Literal["parameters"] = "parameters",
    rtol: float = 1e-5,
    atol: float = 1e-8,
    compare_forward_reverse: bool = True,
    compare_numerical: bool = True,
    numerical_mode: Literal["central", "forward", "backward"] = "central",
    epsilon: float = 1e-5,
    raise_on_failure: Optional[bool] = None,
) -> DifferentiabilityReport:
    """
    Run a suite of differentiability checks on a MacroStat model.

    Parameters
    ----------
    model :
        MacroStat model instance.
    loss_fn :
        Scalar loss function of the model outputs (dict[str, torch.Tensor]).
    scenario :
        Scenario index or name to use.
    target :
        Currently only ``\"parameters\"`` is supported (placeholder for future).
    rtol, atol :
        Relative and absolute tolerances for comparisons.
    compare_forward_reverse :
        If True, compare reverse-mode and forward-mode autograd gradients.
    compare_numerical :
        If True, compare autograd gradients to numerical finite-difference gradients.
    numerical_mode :
        Finite-difference scheme to use when comparing against numerical.
    epsilon :
        Step size for finite differences.
    raise_on_failure :
        If True and checks fail, raise a RuntimeError instead of just returning
        the report. If None, do not raise.
    """
    if target != "parameters":
        raise NotImplementedError(
            "Only target='parameters' is supported at the moment."
        )

    details: Dict[str, dict] = {}

    # ------------------------------------------------------------------
    # Autograd (reverse-mode) baseline
    # ------------------------------------------------------------------
    auto = JacobianAutograd(model=model, scenario=scenario)
    grads_rev = auto.compute(loss_fn=loss_fn, mode="rev")
    nan_or_inf = _has_nan_or_inf(grads_rev)
    details["autograd_rev"] = {"nan_or_inf": nan_or_inf}

    # ------------------------------------------------------------------
    # Forward vs reverse comparison
    # ------------------------------------------------------------------
    fwd_vs_rev_ok: Optional[bool] = None
    max_abs_diff_fwd_rev: Optional[float] = None
    rel_err_fwd_rev: Optional[float] = None

    if compare_forward_reverse:
        grads_fwd = auto.compute(loss_fn=loss_fn, mode="fwd")
        max_abs_diff_fwd_rev = _max_abs_diff_dict(grads_rev, grads_fwd)
        # Scale tolerance by the typical gradient magnitude
        scale = max(
            _max_abs_val_dict(grads_rev),
            _max_abs_val_dict(grads_fwd),
            1.0,
        )
        rel_err_fwd_rev = max_abs_diff_fwd_rev / scale
        fwd_vs_rev_ok = max_abs_diff_fwd_rev <= (atol + rtol * scale)
        details["autograd_fwd"] = {
            "max_abs_diff_fwd_rev": max_abs_diff_fwd_rev,
            "ok": fwd_vs_rev_ok,
            "rel_err": rel_err_fwd_rev,
        }

    # ------------------------------------------------------------------
    # Numerical comparison
    # ------------------------------------------------------------------
    autodiff_vs_numerical_ok: Optional[bool] = None
    max_abs_diff_autodiff_num: Optional[float] = None
    rel_err_autodiff_num: Optional[float] = None

    if compare_numerical:
        num = JacobianNumerical(model=model, scenario=scenario, epsilon=epsilon)
        grads_num = num.compute(loss_fn=loss_fn, mode=numerical_mode)
        max_abs_diff_autodiff_num = _max_abs_diff_dict(grads_rev, grads_num)
        scale = max(
            _max_abs_val_dict(grads_rev),
            _max_abs_val_dict(grads_num),
            1.0,
        )
        rel_err_autodiff_num = max_abs_diff_autodiff_num / scale
        autodiff_vs_numerical_ok = max_abs_diff_autodiff_num <= (atol + rtol * scale)
        details["numerical"] = {
            "max_abs_diff_autodiff_num": max_abs_diff_autodiff_num,
            "ok": autodiff_vs_numerical_ok,
            "rel_err": rel_err_autodiff_num,
        }

    # ------------------------------------------------------------------
    # Overall status
    # ------------------------------------------------------------------
    passed_checks = [not nan_or_inf]
    if fwd_vs_rev_ok is not None:
        passed_checks.append(fwd_vs_rev_ok)
    if autodiff_vs_numerical_ok is not None:
        passed_checks.append(autodiff_vs_numerical_ok)

    passed = all(passed_checks)

    report = DifferentiabilityReport(
        passed=passed,
        nan_or_inf=nan_or_inf,
        fwd_vs_rev_ok=fwd_vs_rev_ok,
        autodiff_vs_numerical_ok=autodiff_vs_numerical_ok,
        max_abs_diff_fwd_rev=max_abs_diff_fwd_rev,
        max_abs_diff_autodiff_num=max_abs_diff_autodiff_num,
        rel_err_fwd_rev=rel_err_fwd_rev,
        rel_err_autodiff_num=rel_err_autodiff_num,
        details=details,
    )

    if raise_on_failure and not passed:
        raise RuntimeError(f"Differentiability check failed:\n{report.summary()}")

    return report
