Parameter Estimation and Calibration
=====================================

This page describes the :mod:`macrostat.estimation` module, which provides tools
for calibrating MacroStat model parameters to empirical data using PyTorch
optimizers and loss functions.

Overview
--------

Parameter estimation in macroeconomic models is essential for matching model
dynamics to real-world data. The :mod:`macrostat.estimation` module provides
flexible loss functions designed to work with both:

* **Standard PyTorch optimizers** (Adam, SGD, etc.) for general-purpose
  calibration with gradient descent methods.
* **Levenberg-Marquardt optimization** (future implementation) for fast,
  accurate nonlinear least squares estimation.

The core components are:

* :class:`macrostat.estimation.EstimationResult` – Dataclass for storing
  optimization results
* :func:`macrostat.estimation.mse_loss` – Mean squared error loss with
  variable/timestep selection
* :func:`macrostat.estimation.weighted_residuals` – Weighted MSE for
  multi-variable calibration
* :func:`macrostat.estimation.composite_loss` – Combine multiple loss
  components (data fit + regularization)


Quick start
-----------

The following example demonstrates calibrating GL06SIM model parameters to
synthetic target data using :func:`torch.optim.Adam`:

.. code-block:: python

    import torch
    from macrostat.models import get_model
    from macrostat.estimation import mse_loss

    # Load model and generate synthetic target data
    model = get_model("GL06SIM")()
    model.parameters["PropensityToConsumeIncome"] = 0.6  # True value

    with torch.no_grad():
        target_output = model.simulate(scenario=0)

    # Perturb parameter for calibration
    model.parameters["PropensityToConsumeIncome"] = 0.7

    # Define loss function
    def loss_fn(output):
        return mse_loss(
            output,
            target_output,
            variables=["ConsumptionDemand", "DisposableIncome"],
            timesteps=slice(5, 40),  # Transition period
            reduction="mean",
        )

    # Optimize with Adam
    behavior = model.get_model_training_instance(scenario=0)
    params = [p for name, p in behavior.named_parameters() if name.startswith("params.")]
    optimizer = torch.optim.Adam(params, lr=5e-3)

    for epoch in range(200):
        optimizer.zero_grad()
        output = behavior.forward()
        loss = loss_fn(output)
        loss.backward()
        optimizer.step()


Loss functions
--------------

Mean squared error loss
^^^^^^^^^^^^^^^^^^^^^^^

The :func:`macrostat.estimation.mse_loss` function computes the mean squared
error between model output and target data. It supports variable selection and
timestep slicing to focus calibration on specific model behaviors.

**Variable selection**: By default, the loss includes all variables present in
both output and target dictionaries. Use the ``variables`` parameter to restrict
calibration to specific endogenous variables:

.. code-block:: python

    # Focus on consumption and income dynamics
    loss = mse_loss(
        output,
        target,
        variables=["ConsumptionDemand", "DisposableIncome"],
        reduction="mean",
    )

**Timestep selection**: Use the ``timesteps`` parameter to select which periods
contribute to the loss. This is crucial for effective calibration:

.. code-block:: python

    # Use transition period (most informative for parameter sensitivity)
    loss = mse_loss(output, target, timesteps=slice(5, 40), reduction="mean")

    # Use steady state (final 20 periods)
    loss = mse_loss(output, target, timesteps=slice(-20, None), reduction="mean")

**Reduction modes**: The ``reduction`` parameter controls the output format:

* ``"none"`` – Returns a 1D residual vector (for Levenberg-Marquardt)
* ``"mean"`` – Returns scalar mean squared error (for torch.optim)
* ``"sum"`` – Returns scalar sum of squared errors

When using standard PyTorch optimizers, always use ``reduction="mean"`` or
``reduction="sum"``. The ``reduction="none"`` mode is reserved for future
Levenberg-Marquardt integration, which requires the full residual vector for
computing the Jacobian.


Weighted residuals
^^^^^^^^^^^^^^^^^^

When calibrating to multiple variables with different scales or importance,
:func:`macrostat.estimation.weighted_residuals` allows balancing their
contributions to the loss:

.. code-block:: python

    from macrostat.estimation import weighted_residuals

    # Prioritize GDP fit over consumption
    weights = {"GDP": 10.0, "Consumption": 1.0, "Investment": 1.0}

    def loss_fn(output):
        return weighted_residuals(
            output,
            target,
            weights=weights,
            reduction="mean",
        )

Weights are automatically normalized to sum to the number of variables, ensuring
that the loss magnitude remains comparable to unweighted losses. This prevents
the loss scale from changing dramatically when adding or removing variables.


Composite loss
^^^^^^^^^^^^^^

The :func:`macrostat.estimation.composite_loss` function combines multiple loss
components, enabling calibration with regularization or multiple datasets:

.. code-block:: python

    from macrostat.estimation import mse_loss, composite_loss

    # Data fitting loss
    def data_loss(output, target):
        return mse_loss(output, target, reduction="mean")

    # Regularization loss (prevent overfitting)
    def regularization_loss(output, _):
        params = model.get_model_training_instance(scenario=0).named_parameters()
        penalty = sum((p - prior[name])**2 for name, p in params)
        return 0.01 * penalty

    # Combine losses
    def loss_fn(output):
        return composite_loss(
            output,
            targets=[target_data, None],  # regularization doesn't use target
            loss_fns=[data_loss, regularization_loss],
            loss_weights=[1.0, 0.1],  # data fit weighted 10x more
            reduction="mean",
        )

The weights are normalized so that the average weight is 1.0, maintaining
consistent loss magnitudes across different numbers of components.


Calibration best practices
---------------------------

Use transition dynamics, not steady state
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The steady state of a model contains limited information about parameter values
because all dynamics have converged. The transition period (typically timesteps
5-40 for a 50-period simulation) contains rich dynamics that are highly
sensitive to parameter values.

**Bad practice** (steady state only):

.. code-block:: python

    # Limited information – all variables have converged
    loss = mse_loss(output, target, timesteps=slice(-10, None), reduction="mean")

**Good practice** (transition period):

.. code-block:: python

    # Rich dynamics – captures parameter sensitivity
    loss = mse_loss(output, target, timesteps=slice(5, 40), reduction="mean")


Select informative variables
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Not all variables are equally informative for calibration. Focus on variables
that are directly influenced by the parameters you are estimating and avoid
redundant accounting identities.

For example, when calibrating consumption parameters in GL06SIM:

.. code-block:: python

    # Focus on consumption and income (directly related to parameters)
    variables = ["ConsumptionDemand", "DisposableIncome"]

    # Avoid including GDP (redundant with consumption + investment)
    # Avoid including Savings (accounting identity: S = Y - C)


Monitor parameter bounds
^^^^^^^^^^^^^^^^^^^^^^^^^

Macroeconomic parameters often have natural bounds (e.g., propensities between 0
and 1, depreciation rates below 1). Standard PyTorch optimizers like Adam can
violate these bounds during optimization.

Consider using constrained optimizers or parameter transformations:

.. code-block:: python

    # Transform parameters to enforce bounds
    class BoundedParameter(torch.nn.Module):
        def __init__(self, value, lower=0.0, upper=1.0):
            super().__init__()
            # Use logit transform to map (0,1) to (-inf, inf)
            self.param_unconstrained = torch.nn.Parameter(
                torch.logit(torch.tensor((value - lower) / (upper - lower)))
            )
            self.lower = lower
            self.upper = upper

        def forward(self):
            # Map back to (lower, upper)
            return self.lower + (self.upper - self.lower) * torch.sigmoid(
                self.param_unconstrained
            )


Choosing an optimizer
^^^^^^^^^^^^^^^^^^^^^

Different optimizers have different strengths for parameter calibration:

* **Adam** – Good general-purpose choice. Fast convergence, handles different
  parameter scales well. May overshoot optimal values.
* **L-BFGS** – Quasi-Newton method. Often converges faster than Adam for smooth
  losses. Requires strong_wolfe line search for MacroStat models.
* **SGD with momentum** – Simple and robust, but slower convergence. Good for
  final refinement after Adam.
* **Levenberg-Marquardt** (future) – Specialized for nonlinear least squares.
  Expected to provide fastest, most accurate convergence for typical calibration
  problems.

Example using L-BFGS:

.. code-block:: python

    optimizer = torch.optim.LBFGS(
        params,
        lr=1.0,
        max_iter=20,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        output = behavior.forward()
        loss = loss_fn(output)
        loss.backward()
        return loss

    for epoch in range(50):
        loss = optimizer.step(closure)


Working with EstimationResult
------------------------------

The :class:`macrostat.estimation.EstimationResult` dataclass provides a
standardized container for optimization results, analogous to
:class:`scipy.optimize.OptimizeResult`:

.. code-block:: python

    from macrostat.estimation import EstimationResult

    result = EstimationResult(
        success=True,
        status=0,
        message="ftol termination condition satisfied",
        params={"alpha": torch.tensor(0.8)},
        cost=0.0012,
        residuals=torch.zeros(100),
        jacobian={"alpha": torch.randn(100)},
        nfev=45,
        njev=12,
        nit=12,
        optimality=1e-9,
    )

    print(result)  # Human-readable summary

This class is designed for future Levenberg-Marquardt integration, which will
return populated EstimationResult instances with convergence diagnostics. The
current torch.optim workflow does not use this class, but you may manually
construct EstimationResult objects to maintain consistent result structures
across different optimization methods.


Integration with differentiability tools
-----------------------------------------

The estimation module is designed to work seamlessly with
:mod:`macrostat.diff`. For future Levenberg-Marquardt optimization, use
``reduction="none"`` to return residual vectors compatible with Jacobian
computation:

.. code-block:: python

    from macrostat.diff import JacobianAutograd
    from macrostat.estimation import mse_loss

    # Loss function returning residual vector
    def loss_fn(output):
        return mse_loss(
            output,
            target_data,
            variables=["ConsumptionDemand"],
            timesteps=slice(5, 40),
            reduction="none",
        )

    # Compute Jacobian
    jac = JacobianAutograd(model, scenario=0)
    jacobian = jac.complete(loss_fn=loss_fn, mode="rev")

This pattern will be used internally by the Levenberg-Marquardt optimizer to
compute the Jacobian of residuals with respect to parameters, enabling efficient
second-order optimization.


Complete calibration example
-----------------------------

The following complete example demonstrates the full calibration workflow,
including parameter perturbation, optimization, and result validation:

.. literalinclude:: ../../examples/calibration_basic.py
   :language: python
   :linenos:
   :lines: 29-

This example can be run from the MacroStat repository:

.. code-block:: bash

    uv run python examples/calibration_basic.py


See also
--------

* :doc:`diff` – Jacobian computation and differentiability checking
* :doc:`sample` – Parameter space sampling for sensitivity analysis
* :ref:`api` – Complete API reference
