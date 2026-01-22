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

    # Numerical finite-difference gradients
    jac_num = JacobianNumerical(model, epsilon=1e-5)
    grads_num = jac_num.compute(loss_fn=loss_fn, mode="central")


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
        epsilon=1e-5,
    )

    print(report.summary())

The resulting :class:`macrostat.diff.DifferentiabilityReport` contains:

* ``nan_or_inf`` – whether any gradient contains NaNs or Infs.
* ``rel_err_fwd_rev`` – relative discrepancy between forward- and reverse-mode
  autograd gradients.
* ``rel_err_autodiff_num`` – relative discrepancy between autograd and
  numerical finite-difference gradients.

Rather than only returning a pass/fail flag, the report is intended to help
you interpret *how accurate* the gradients are (e.g. "accurate to about
``1e-3`` relative error" on a given model and loss).


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
