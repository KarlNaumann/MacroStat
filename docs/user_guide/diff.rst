Differentiability and Jacobian Tools
====================================

This page describes the :mod:`macrostat.diff` module, which provides utilities
for computing and validating Jacobians of MacroStat models using PyTorch.

Overview
--------

The differentiability tools are designed for two use cases:

* **Development-time checks** that a model is differentiable and that
  gradients are sensible (no NaNs/Infs, forward vs reverse AD match, etc.).
* **Calibration and analysis** that require Jacobians or gradient information
  with respect to model parameters.

The core pieces live in :mod:`macrostat.diff`:

* :class:`macrostat.diff.JacobianBase`
* :class:`macrostat.diff.JacobianAutograd`
* :class:`macrostat.diff.JacobianNumerical`
* :func:`macrostat.diff.check_model_differentiability`
* :func:`macrostat.diff.compare_jacobian_dicts`


Quick start
-----------

Given a MacroStat model instance (for example :class:`macrostat.models.GL06SIMEX`
or the small linear test model :class:`macrostat.models.LINEAR2D`), you can
define a scalar loss and compute gradients as follows:

.. code-block:: python

    import torch
    from macrostat.models import get_model
    from macrostat.diff import JacobianAutograd, JacobianNumerical

    # Build a model
    ModelClass = get_model("LINEAR2D")
    model = ModelClass()

    # Define a loss on the model outputs
    target = torch.tensor([1.0, 0.0])

    def loss_fn(output: dict[str, torch.Tensor]) -> torch.Tensor:
        y = output["State"][-1]  # final 2D state
        return torch.nn.functional.mse_loss(y, target)

    # Autograd-based gradients
    jac_auto = JacobianAutograd(model)
    grads_rev = jac_auto.compute(loss_fn=loss_fn, mode="rev")

    # Numerical finite-difference gradients (default: relative-step, eps=1e-3)
    jac_num_rel = JacobianNumerical(model)
    grads_num_rel = jac_num_rel.compute(loss_fn=loss_fn, mode="central")

    # Or in direct space, useful when any parameter equals zero
    jac_num_direct = JacobianNumerical(
        model, epsilon=1e-5, parameter_space="direct"
    )
    grads_num_direct = jac_num_direct.compute(loss_fn=loss_fn, mode="central")


Perturbation schemes: direct, relative, and log
------------------------------------------------

:class:`~macrostat.diff.JacobianNumerical` supports three schemes,
controlled by the ``parameter_space`` argument. Two independent choices
are folded into that one word: *how the step is sized* and *which
derivative is returned*. ``"direct"`` sizes the step additively;
``"relative"`` and ``"log"`` size it as a fraction of the parameter.
``"direct"`` and ``"relative"`` return the ordinary derivative
:math:`\partial f/\partial p`; ``"log"`` returns the logarithmic
derivative :math:`\partial f/\partial \log|p|`. So ``"relative"`` and
``"log"`` fire the identical perturbation and differ only in the
quantity they hand back.

.. versionchanged:: 0.7.0
   ``parameter_space="log"`` previously sized a fractional step but
   returned :math:`\partial f/\partial p`. That behaviour is now named
   ``"relative"`` and is the default. ``"log"`` now returns the
   logarithmic derivative :math:`\partial f/\partial \log|p|`. Replace
   ``parameter_space="log"`` with ``parameter_space="relative"`` to keep
   the old numbers.

**Direct** ``parameter_space="direct"`` perturbs each parameter
additively and returns the ordinary derivative:

.. math::

    \frac{\partial f}{\partial p}
    \approx \frac{f(p + \varepsilon) - f(p - \varepsilon)}{2\varepsilon}.

This is the textbook central difference. It works for any value including
zero, but the absolute step has no relation to the parameter scale, so a
single ``epsilon`` is too coarse for parameters near unity and too fine
for parameters near zero.

**Relative** ``parameter_space="relative"`` sizes the step as a fixed
fraction of the parameter, :math:`p \mapsto p\,e^{\pm\varepsilon}`, and
returns the same ordinary derivative :math:`\partial f/\partial p`:

.. math::

    \frac{\partial f}{\partial p}
    \approx \frac{f(p\,e^{\varepsilon}) - f(p\,e^{-\varepsilon})}
                 {p\,(e^{\varepsilon} - e^{-\varepsilon})}.

The step is now scale-invariant, so the same ``epsilon`` (default
``1e-3``, roughly a 0.1% relative step) is appropriate across parameters
that differ by many orders of magnitude. This is the recommended default
for SFC and behavioral models. Note the :math:`p` in the denominator: the
fractional step is undone on the way out, so the returned number is still
:math:`\partial f/\partial p`, in the same units as the ``"direct"``
result.

**Log** ``parameter_space="log"`` fires the identical perturbation as
``"relative"`` but returns the logarithmic derivative
:math:`\partial f/\partial \log|p|`, obtained by dropping the :math:`p`
from the denominator:

.. math::

    \frac{\partial f}{\partial \log|p|}
    \approx \frac{f(p\,e^{\varepsilon}) - f(p\,e^{-\varepsilon})}
                 {e^{\varepsilon} - e^{-\varepsilon}}
    \;=\; p\,\frac{\partial f}{\partial p}.

The result equals the ``"relative"`` result multiplied by the signed
parameter value. A log-derivative reads as a semi-elasticity: the change
in :math:`f` per unit change in :math:`\log|p|`, i.e. per fractional
change in :math:`|p|`. Use it to compare sensitivities across parameters
of very different magnitude on a common footing, where the raw
:math:`\partial f/\partial p` would be dominated by units.

.. warning::

   The log scheme differentiates with respect to :math:`\log|p|`, not
   :math:`\log p`, because :math:`\log p` is complex for :math:`p<0`. The
   sign of :math:`p` is held fixed across the perturbation, so the
   returned value is :math:`p\,\partial f/\partial p` carrying the sign of
   :math:`p`. For a negative parameter the reported log-sensitivity
   therefore has the opposite sign to :math:`\partial f/\partial |p|`.
   When you rank or compare log-sensitivities across parameters of mixed
   sign, compare magnitudes, not signed values, or the sign of :math:`p`
   will masquerade as a difference in sensitivity. :math:`p=0` has no
   logarithmic derivative and raises.

Choice of ``epsilon`` is a tradeoff: large values amplify truncation
error from the finite-difference formula, small values amplify
floating-point roundoff. For float32 in the relative/log scheme, values
between ``1e-4`` and ``1e-2`` typically work well; ``1e-3`` is the
package default.

**Zero-parameter caveat.** The ``"relative"`` and ``"log"`` schemes are
undefined when a parameter equals zero (you cannot take the log of zero,
and the relative step vanishes). :class:`~macrostat.diff.JacobianNumerical`
detects this during ``compute`` and raises
``ValueError: Cannot use parameter_space in {'relative', 'log'} with zero
parameters: [...]``. For example, the :class:`~macrostat.models.LINEAR2D`
test model has a parameter ``x0_2`` that defaults to zero, so its tests
must set it non-zero or use ``parameter_space="direct"``. The autograd
backend instead reports a zero-valued parameter's log-derivative column as
an exact zero. Negative parameters are handled by the sign-preserving
transformation ``sign(p) * exp(log(abs(p)) +/- eps)``.


High-level differentiability check
----------------------------------

For a more complete diagnostic, use
:func:`macrostat.diff.check_model_differentiability`:

.. code-block:: python

    from macrostat.diff import check_model_differentiability

    report = check_model_differentiability(
        model=model,
        loss_fn=loss_fn,
        scenario=0,
        rtol=1e-5,
        atol=1e-8,
        compare_forward_reverse=True,
        compare_numerical=True,
        numerical_mode="central",
        epsilon=1e-3,
        parameter_space="relative",
    )

    print(report.summary())

The resulting :class:`macrostat.diff.DifferentiabilityReport` contains:

* ``nan_or_inf`` -- whether any gradient contains NaNs or Infs.
* ``rel_err_fwd_rev`` -- relative discrepancy between forward- and reverse-mode
  autograd gradients.
* ``rel_err_autodiff_num`` -- relative discrepancy between autograd and
  numerical finite-difference gradients.

Rather than only returning a pass/fail flag, the report is intended to help
you interpret *how accurate* the gradients are (e.g. "accurate to about
``1e-3`` relative error" on a given model and loss).

Set ``parameter_space="direct"`` if your model has any parameter
equal to zero. The default is ``"relative"``.


Per-parameter Jacobian comparison
---------------------------------

When the high-level check reports a disagreement, the next question
is *which parameter* is responsible.
:func:`~macrostat.diff.compare_jacobian_dicts` performs an
element-wise, per-parameter comparison of two Jacobian dictionaries
and returns a :class:`~macrostat.diff.JacobianComparisonReport` with
the following fields per parameter:

* ``max_abs_diff``, ``mean_abs_diff`` -- absolute differences across
  all elements of the parameter's Jacobian slice.
* ``max_rel_diff``, ``mean_rel_diff`` -- the same, normalised by
  ``max(|a|, |b|)`` clipped to one.
* ``num_close`` / ``num_elements`` -- how many entries are within
  the ``atol``/``rtol`` tolerance.
* ``has_nan_inf`` -- whether either Jacobian contains NaNs or Infs.

A typical workflow:

.. code-block:: python

    from macrostat.diff import (
        JacobianAutograd, JacobianNumerical, compare_jacobian_dicts,
    )

    jac_auto = JacobianAutograd(model).compute(loss_fn=loss_fn, mode="rev")
    jac_num = JacobianNumerical(model).compute(loss_fn=loss_fn, mode="central")

    report = compare_jacobian_dicts(
        jac_auto, jac_num, method_a="autograd", method_b="numerical"
    )
    print(report.summary())

The summary prints one row per parameter, sorted by descending
relative disagreement, so the offending parameters are at the top.
See :doc:`jacobian_diagnostics` for a runnable walkthrough that
applies this workflow to LINEAR2D and several GL06 variants.


Expected disagreements
----------------------

Two classes of "disagreement" between autograd and numerical
Jacobians are expected and not bugs.

**Constraint-derived parameters.** When a model declares a
:class:`~macrostat.core.constraints.LinearConstraint`, the derived
(residual) parameter is no longer free: its value is fixed by the
adding-up identity. Autograd correctly reports a zero gradient for
the derived parameter and absorbs the sensitivity into the free
parameters of the same group. Numerical Jacobians computed by
perturbing the *free* parameters agree with autograd; numerical
Jacobians computed by perturbing the *derived* parameter directly
will disagree, because such a perturbation would violate the
constraint. See :doc:`core/constraints` for the residual
parameterization and :doc:`constraints_example` for a worked
example.

**Step-function operations.** Operations like
``torch.where(x > 0, a, b)``, ``torch.clamp``, and any other
discontinuous switch are non-differentiable through the condition
argument. Autograd reports zero gradient at the discontinuity, while
finite differences see a finite jump whenever the perturbation
straddles it. The disagreement is real but local, and it disappears
when the step function is replaced by a smooth approximation. The
:class:`~macrostat.core.behavior.Behavior` base class provides
:meth:`~macrostat.core.behavior.Behavior.diffwhere`,
:meth:`~macrostat.core.behavior.Behavior.tanhmask`,
:meth:`~macrostat.core.behavior.Behavior.diffmin`, and
:meth:`~macrostat.core.behavior.Behavior.diffmax` for this purpose.
:doc:`jacobian_diagnostics` shows how to localize this kind of
disagreement to a specific parameter, timestep, and output.


Test model: LINEAR2D
--------------------

To validate the implementation itself, MacroStat ships with a tiny linear
test model :class:`macrostat.models.LINEAR2D.LINEAR2D`. Its dynamics are:

.. math::

    x_{t+1} = A x_t,

where :math:`x_t` is a 2D state vector and :math:`A` is a fully parameterised
2x2 matrix. This model has an analytical Jacobian, so it is used
in the test suite to confirm that:

* Forward and reverse autograd agree to high precision, and
* Autograd gradients match finite-difference gradients up to a small relative
  error.

Because LINEAR2D has a parameter (``x0_2``) that defaults to zero, the
``"relative"`` and ``"log"`` schemes are undefined for it, so its
differentiability tests set that parameter non-zero or use
``parameter_space="direct"`` explicitly.
