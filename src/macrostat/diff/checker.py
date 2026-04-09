"""
High-level differentiability checks for MacroStat models.

This module provides a small API to compare:

- Reverse-mode vs forward-mode autograd Jacobians, and
- Autograd vs numerical finite-difference Jacobians

for a user-specified scalar loss function of model outputs.

It also provides :func:`compare_jacobian_dicts` for element-wise,
per-parameter comparison of two Jacobian dictionaries.
"""

from __future__ import annotations

from dataclasses import dataclass, field
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
    parameter_space: Literal["direct", "log"] = "direct",
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
    parameter_space :
        Space in which to apply perturbations for numerical Jacobian.
        Default ``"direct"`` is safe for all parameters including zeros.
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
        num = JacobianNumerical(
            model=model,
            scenario=scenario,
            epsilon=epsilon,
            parameter_space=parameter_space,
        )
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


# ---------------------------------------------------------------------------
# Per-parameter Jacobian comparison
# ---------------------------------------------------------------------------


@dataclass
class ParameterComparison:
    """Element-wise comparison statistics for a single parameter's Jacobian."""

    name: str
    max_abs_diff: float
    mean_abs_diff: float
    max_rel_diff: float
    mean_rel_diff: float
    num_elements: int
    num_close: int
    has_nan_inf: bool


@dataclass
class JacobianComparisonReport:
    """Per-parameter comparison of two Jacobian dictionaries.

    Attributes
    ----------
    method_a, method_b :
        Human-readable labels for the two methods being compared.
    per_parameter :
        Mapping from parameter name to its :class:`ParameterComparison`.
    """

    method_a: str
    method_b: str
    per_parameter: dict[str, ParameterComparison] = field(default_factory=dict)

    # -- derived properties --------------------------------------------------

    @property
    def overall_max_abs_diff(self) -> float:
        if not self.per_parameter:
            return 0.0
        return max(p.max_abs_diff for p in self.per_parameter.values())

    @property
    def overall_max_rel_diff(self) -> float:
        if not self.per_parameter:
            return 0.0
        return max(p.max_rel_diff for p in self.per_parameter.values())

    # -- utilities -----------------------------------------------------------

    def worst_parameters(self, n: int = 5) -> list[ParameterComparison]:
        """Return the *n* parameters with the largest ``max_rel_diff``."""
        return sorted(
            self.per_parameter.values(),
            key=lambda p: p.max_rel_diff,
            reverse=True,
        )[:n]

    def summary(self) -> str:
        """Return a human-readable table of per-parameter comparison stats."""
        lines: list[str] = []
        lines.append(f"Jacobian comparison: {self.method_a} vs {self.method_b}")
        lines.append(f"Parameters compared: {len(self.per_parameter)}")
        lines.append(
            f"Overall max abs diff: {self.overall_max_abs_diff:.3e}  "
            f"max rel diff: {self.overall_max_rel_diff:.3e}"
        )
        lines.append("")

        header = (
            f"{'Parameter':<40s} {'max_abs':>10s} {'mean_abs':>10s} "
            f"{'max_rel':>10s} {'mean_rel':>10s} {'close':>12s} {'nan/inf':>7s}"
        )
        lines.append(header)
        lines.append("-" * len(header))

        for pc in sorted(self.per_parameter.values(), key=lambda p: -p.max_rel_diff):
            close_str = f"{pc.num_close}/{pc.num_elements}"
            lines.append(
                f"{pc.name:<40s} {pc.max_abs_diff:>10.3e} {pc.mean_abs_diff:>10.3e} "
                f"{pc.max_rel_diff:>10.3e} {pc.mean_rel_diff:>10.3e} "
                f"{close_str:>12s} {'YES' if pc.has_nan_inf else 'no':>7s}"
            )

        return "\n".join(lines)


def compare_jacobian_dicts(
    jac_a: Dict[str, torch.Tensor],
    jac_b: Dict[str, torch.Tensor],
    method_a: str = "A",
    method_b: str = "B",
    atol: float = 1e-8,
    rtol: float = 1e-5,
) -> JacobianComparisonReport:
    """Element-wise, per-parameter comparison of two Jacobian dictionaries.

    Only parameters present in **both** dictionaries are compared.

    Parameters
    ----------
    jac_a, jac_b :
        Jacobian dictionaries as returned by
        :meth:`JacobianAutograd.compute` or :meth:`JacobianNumerical.compute`.
    method_a, method_b :
        Human-readable labels (used in the report).
    atol, rtol :
        Absolute and relative tolerances.  An element is "close" when
        ``|a - b| <= atol + rtol * max(|a|, |b|)``.

    Returns
    -------
    JacobianComparisonReport
    """
    shared_keys = sorted(jac_a.keys() & jac_b.keys())
    per_parameter: dict[str, ParameterComparison] = {}

    for name in shared_keys:
        ga = jac_a[name]
        gb = jac_b[name]

        abs_diff = (ga - gb).abs()
        scale = torch.maximum(ga.abs(), gb.abs()).clamp_min(1.0)
        rel_diff = abs_diff / scale

        has_nan_inf = not (torch.isfinite(ga).all() and torch.isfinite(gb).all())
        num_elements = int(ga.numel())
        num_close = int((abs_diff <= atol + rtol * scale).sum().item())

        per_parameter[name] = ParameterComparison(
            name=name,
            max_abs_diff=float(abs_diff.max().item()),
            mean_abs_diff=float(abs_diff.mean().item()),
            max_rel_diff=float(rel_diff.max().item()),
            mean_rel_diff=float(rel_diff.mean().item()),
            num_elements=num_elements,
            num_close=num_close,
            has_nan_inf=has_nan_inf,
        )

    return JacobianComparisonReport(
        method_a=method_a,
        method_b=method_b,
        per_parameter=per_parameter,
    )
