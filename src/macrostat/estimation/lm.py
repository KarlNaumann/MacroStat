# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Karl Naumann-Woleske
"""
Levenberg-Marquardt optimizer for MacroStat models.

This module implements trust-region nonlinear least-squares optimization
using the Nielsen damping strategy for parameter calibration.
"""

from __future__ import annotations

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

from typing import Callable, Literal

import torch

from macrostat.core.model import Model
from macrostat.diff import JacobianAutograd
from macrostat.estimation.result import EstimationResult

# Type alias for loss functions
LossFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]


class LevenbergMarquardt:
    """
    Levenberg-Marquardt optimizer for MacroStat models.

    Implements trust-region nonlinear least-squares with Nielsen damping
    strategy for robust parameter calibration.

    Parameters
    ----------
    model : Model
        MacroStat model instance to optimize.
    loss_fn : Callable
        Loss function that returns residual vector (reduction="none").
        Must accept model output dict and return 1D tensor.
    scenario : int | str, optional
        Scenario index or name for simulation (default: 0).
    ftol : float, optional
        Cost change tolerance for convergence (relative). Default: 1e-8.
    xtol : float, optional
        Parameter change tolerance for convergence (relative). Default: 1e-8.
    gtol : float, optional
        Gradient norm tolerance for convergence. Default: 1e-8.
    max_nfev : int, optional
        Maximum number of function evaluations. Default: 1000.
    damping_init : float, optional
        Initial damping parameter (Nielsen: 1e-3). Default: 1e-3.
    damping_update_factor : float, optional
        Damping increase factor (Nielsen: 2.0). Default: 2.0.
    scaling : {"marquardt", "identity"}, optional
        Damping scaling strategy. Default: "marquardt".
        - "marquardt": J^T J + λ diag(J^T J) (scale-invariant)
        - "identity": J^T J + λI (Levenberg's original)
    jacobian_mode : {"rev", "fwd"}, optional
        Jacobian computation mode. Default: "rev".
        - "rev": Reverse-mode autodiff (efficient for many residuals)
        - "fwd": Forward-mode autodiff (efficient for few parameters)
    verbose : int, optional
        Verbosity level. Default: 0.
        - 0: Silent
        - 1: Print iteration summary
        - 2: Print detailed progress

    Attributes
    ----------
    nfev : int
        Number of function evaluations performed.
    njev : int
        Number of Jacobian evaluations performed.
    nit : int
        Number of iterations performed.

    Examples
    --------
    >>> from macrostat.models import get_model
    >>> from macrostat.estimation import LevenbergMarquardt, mse_loss
    >>>
    >>> model = get_model("GL06SIM")()
    >>> target_data = {...}  # Your target timeseries
    >>>
    >>> def loss_fn(output):
    ...     return mse_loss(output, target_data, reduction="none")
    >>>
    >>> lm = LevenbergMarquardt(model, loss_fn, verbose=1)
    >>> result = lm.optimize()
    >>> print(f"Converged: {result.success}, Iterations: {result.nit}")

    Notes
    -----
    The algorithm follows Madsen, Nielsen, Tingleff (2004):

    1. Initialize λ = τ * max(diag(J^T J))
    2. Solve (J^T J + λD) δ = -J^T r
    3. Compute gain ratio ρ = (F - F_new) / (predicted reduction)
    4. If ρ > 0: accept step, decrease λ
    5. Else: reject step, increase λ
    6. Check termination criteria

    Nielsen damping provides smooth, scale-invariant updates compared to
    classic multiply/divide by 10 strategy.

    References
    ----------
    Madsen, K., Nielsen, H. B., & Tingleff, O. (2004). Methods for
    Non-Linear Least Squares Problems (2nd ed.). Technical University
    of Denmark.
    """

    def __init__(
        self,
        model: Model,
        loss_fn: LossFn,
        scenario: int | str = 0,
        ftol: float = 1e-8,
        xtol: float = 1e-8,
        gtol: float = 1e-8,
        max_nfev: int = 1000,
        damping_init: float = 1e-3,
        damping_update_factor: float = 2.0,
        scaling: Literal["marquardt", "identity"] = "marquardt",
        jacobian_mode: Literal["rev", "fwd"] = "rev",
        verbose: int = 0,
    ):
        """Initialize Levenberg-Marquardt optimizer."""
        self.model = model
        self.loss_fn = loss_fn
        self.scenario = scenario
        self.ftol = ftol
        self.xtol = xtol
        self.gtol = gtol
        self.max_nfev = max_nfev
        self.damping_init = damping_init
        self.damping_update_factor = damping_update_factor
        self.scaling = scaling
        self.jacobian_mode = jacobian_mode
        self.verbose = verbose

        # Counters
        self.nfev = 0
        self.njev = 0
        self.nit = 0

        # Initialize Jacobian computer
        self.jac_computer = JacobianAutograd(model, scenario=scenario)

    def optimize(self) -> EstimationResult:
        """
        Run Levenberg-Marquardt optimization.

        Returns
        -------
        EstimationResult
            Optimization result containing final parameters, cost, convergence
            status, and iteration statistics.

        Raises
        ------
        RuntimeError
            If optimization encounters numerical issues (e.g., singular Jacobian).
        """
        # Get initial parameters
        params = self._get_parameters()

        # Compute initial residuals and cost
        residuals = self._compute_residuals(params)
        cost = 0.5 * (residuals**2).sum().item()

        # Compute initial Jacobian
        jac_dict = self._compute_jacobian(params)
        J = self._dict_to_matrix(jac_dict, params).detach()

        # Initialize damping parameter (Nielsen)
        JtJ = J.T @ J
        diag_JtJ = torch.diag(JtJ)
        damping = self.damping_init * diag_JtJ.max().item()

        # Initialize damping update factor
        nu = self.damping_update_factor

        # Print header if verbose
        if self.verbose >= 1:
            self._print_header()
            self._print_iteration(0, cost, damping, accepted=True)

        # Main optimization loop
        while True:
            self.nit += 1

            # Solve for step: (J^T J + λD) δ = -J^T r
            step_dict = self._solve_step(J, residuals, damping, params)

            # Compute candidate parameters
            params_new = self._add_step(params, step_dict)

            # Evaluate candidate
            residuals_new = self._compute_residuals(params_new)
            cost_new = 0.5 * (residuals_new**2).sum().item()

            # Compute predicted reduction
            step_vec = self._dict_to_vector(step_dict, params)
            predicted = -step_vec @ (J.T @ residuals) - 0.5 * step_vec @ (
                JtJ @ step_vec
            )
            predicted = predicted.item()

            # Compute gain ratio
            actual = cost - cost_new
            rho = actual / predicted if abs(predicted) > 1e-20 else 0.0

            # Decide whether to accept step
            if rho > 0:
                # Accept step
                cost_prev = cost  # Save for termination check
                params = params_new
                residuals = residuals_new
                cost = cost_new

                # Update Jacobian
                jac_dict = self._compute_jacobian(params)
                J = self._dict_to_matrix(jac_dict, params).detach()
                JtJ = J.T @ J
                diag_JtJ = torch.diag(JtJ)

                # Decrease damping (Nielsen strategy)
                damping *= max(1.0 / 3.0, 1.0 - (2.0 * rho - 1.0) ** 3)
                nu = self.damping_update_factor

                accepted = True
            else:
                # Reject step
                cost_prev = cost  # For rejected steps, use current cost
                damping *= nu
                nu *= 2.0
                accepted = False

            # Print progress
            if self.verbose >= 1:
                self._print_iteration(self.nit, cost, damping, accepted)

            # Check termination criteria
            converged, status, message = self._check_termination(
                cost_prev, cost, step_dict, params, J, residuals, accepted
            )

            if converged:
                break

            # Check max iterations
            if self.nfev >= self.max_nfev:
                status = -1
                message = "maximum function evaluations reached"
                break

        # Compute final gradient norm
        g = J.T @ residuals
        optimality = torch.abs(g).max().item()

        # Build result
        result = EstimationResult(
            success=(status >= 0),
            status=status,
            message=message,
            params={k: v.detach().clone() for k, v in params.items()},
            cost=cost,
            residuals=residuals,
            jacobian=jac_dict,
            nfev=self.nfev,
            njev=self.njev,
            nit=self.nit,
            optimality=optimality,
        )

        if self.verbose >= 1:
            print("\nOptimization terminated:")
            print(f"  Status: {message}")
            print(f"  Final cost: {cost:.6e}")
            print(f"  Optimality: {optimality:.6e}")
            print(f"  Iterations: {self.nit}")
            print(f"  Function evaluations: {self.nfev}")
            print(f"  Jacobian evaluations: {self.njev}")

        return result

    def _get_parameters(self) -> dict[str, torch.Tensor]:
        """Get current free parameter values from model.

        Excludes derived (constrained) parameters, which are computed
        from free parameters via ``enforce_constraints()``.
        """
        params = {}
        for name in self.model.parameters.get_free_param_names():
            params[name] = torch.tensor(
                self.model.parameters.values[name]["value"],
                dtype=torch.float64,
            )
        return params

    def _compute_residuals(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute residuals for given parameters."""
        # Update model parameters
        for name, value in params.items():
            self.model.parameters.values[name]["value"] = value.item()

        # Get fresh behavior instance with updated parameters
        behavior = self.model.get_model_training_instance(scenario=self.scenario)

        # Run model
        with torch.no_grad():
            output = behavior()

        # Compute residuals
        residuals = self.loss_fn(output)

        # Increment counter
        self.nfev += 1

        return residuals

    def _compute_jacobian(
        self, params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Compute Jacobian for given parameters."""
        # Create parameter dict for functional_call (needed by JacobianAutograd)
        # JacobianAutograd will use functional_call internally, so we just need
        # to ensure the model parameters are set correctly
        for name, value in params.items():
            # Update the actual model parameters (not behavior)
            self.model.parameters.values[name]["value"] = value.item()

        # Compute Jacobian
        jac_dict = self.jac_computer.compute(self.loss_fn, mode=self.jacobian_mode)

        # Increment counter
        self.njev += 1

        return jac_dict

    def _dict_to_matrix(
        self, jac_dict: dict[str, torch.Tensor], params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Convert Jacobian dict to matrix."""
        # Use parameter ordering from params dict
        param_names = list(params.keys())

        # Stack Jacobian columns
        jac_cols = [jac_dict[name] for name in param_names]
        J = torch.stack(jac_cols, dim=1)

        return J

    def _dict_to_vector(
        self, vec_dict: dict[str, torch.Tensor], params: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Convert parameter dict to vector."""
        param_names = list(params.keys())
        vec = torch.tensor([vec_dict[name].item() for name in param_names])
        return vec

    def _vector_to_dict(
        self, vec: torch.Tensor, params: dict[str, torch.Tensor]
    ) -> dict[str, torch.Tensor]:
        """Convert vector to parameter dict."""
        param_names = list(params.keys())
        vec_dict = {name: vec[i].detach() for i, name in enumerate(param_names)}
        return vec_dict

    def _solve_step(
        self,
        J: torch.Tensor,
        residuals: torch.Tensor,
        damping: float,
        params: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Solve for LM step: (J^T J + λD) δ = -J^T r.

        Parameters
        ----------
        J : torch.Tensor
            Jacobian matrix (n_residuals, n_params).
        residuals : torch.Tensor
            Residual vector (n_residuals,).
        damping : float
            Damping parameter λ.
        params : dict[str, torch.Tensor]
            Current parameters (for ordering).

        Returns
        -------
        dict[str, torch.Tensor]
            Step vector as parameter dict.
        """
        # Compute J^T J
        JtJ = J.T @ J

        # Add damping
        if self.scaling == "marquardt":
            # Marquardt: λ * diag(J^T J)
            diag_JtJ = torch.diag(JtJ)
            A = JtJ + damping * torch.diag(diag_JtJ)
        elif self.scaling == "identity":
            # Levenberg: λ * I
            n_params = JtJ.shape[0]
            A = JtJ + damping * torch.eye(n_params)
        else:
            raise ValueError(f"Unknown scaling: {self.scaling}")

        # Compute right-hand side: -J^T r
        rhs = -J.T @ residuals

        # Solve linear system
        try:
            step = torch.linalg.solve(A, rhs)
        except torch.linalg.LinAlgError:
            # Singular matrix - add more damping
            A = A + 1e-6 * torch.eye(A.shape[0])
            step = torch.linalg.solve(A, rhs)

        # Convert to dict
        step_dict = self._vector_to_dict(step, params)

        return step_dict

    def _add_step(
        self,
        params: dict[str, torch.Tensor],
        step: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Add step to parameters."""
        params_new = {}
        for name in params.keys():
            params_new[name] = params[name] + step[name]
        return params_new

    def _check_termination(
        self,
        cost: float,
        cost_new: float,
        step: dict[str, torch.Tensor],
        params: dict[str, torch.Tensor],
        J: torch.Tensor,
        residuals: torch.Tensor,
        accepted: bool,
    ) -> tuple[bool, int, str]:
        """
        Check termination criteria.

        Returns
        -------
        tuple[bool, int, str]
            (converged, status, message)
            status: 0=ftol, 1=xtol, 2=gtol, -1=maxiter, -2=failure
        """
        # Only check convergence if step was accepted
        if not accepted:
            return False, -1, ""

        # ftol: relative cost change
        if abs(cost - cost_new) < self.ftol * (cost + self.ftol):
            return True, 0, "ftol termination condition satisfied"

        # xtol: relative parameter change
        step_vec = self._dict_to_vector(step, params)
        params_vec = self._dict_to_vector(params, params)
        step_norm = torch.norm(step_vec).item()
        params_norm = torch.norm(params_vec).item()

        if step_norm < self.xtol * (params_norm + self.xtol):
            return True, 1, "xtol termination condition satisfied"

        # gtol: gradient norm
        g = J.T @ residuals
        g_inf_norm = torch.abs(g).max().item()

        if g_inf_norm < self.gtol:
            return True, 2, "gtol termination condition satisfied"

        return False, -1, ""

    def _print_header(self):
        """Print optimization header."""
        print("=" * 70)
        print("Levenberg-Marquardt Optimization")
        print("=" * 70)
        print(
            f"{'Iter':>5} {'Cost':>12} {'Damping':>12} {'Status':>10} "
            f"{'nfev':>5} {'njev':>5}"
        )
        print("-" * 70)

    def _print_iteration(
        self, iteration: int, cost: float, damping: float, accepted: bool
    ):
        """Print iteration summary."""
        status = "accept" if accepted else "reject"
        print(
            f"{iteration:5d} {cost:12.6e} {damping:12.6e} {status:>10} "
            f"{self.nfev:5d} {self.njev:5d}"
        )
