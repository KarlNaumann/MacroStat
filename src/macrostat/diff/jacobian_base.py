"""
Base class and utilities for Jacobian computation in MacroStat.

This module provides shared functionality for working with MacroStat models
in a way that is compatible with PyTorch's autograd and torch.func APIs.
"""

from __future__ import annotations

import warnings
from typing import Dict, Tuple

import pandas as pd
import torch

from macrostat.core.model import Model


class JacobianBase:
    """
    Base class for Jacobian computation.

    This class is responsible for:

    - Constructing a training ``Behavior`` instance from a MacroStat ``Model``.
    - Extracting the parameters of the behavior as a dictionary suitable for
      use with :mod:`torch.func` utilities.
    """

    def __init__(self, model: Model, scenario: int | str = 0):
        """
        Parameters
        ----------
        model :
            A MacroStat model instance (e.g. ``GL06SIMEX()``).
        scenario :
            Scenario index or name to use when constructing the behavior.
        """
        self.model = model
        self.scenario = scenario

    # ------------------------------------------------------------------
    # Core utilities
    # ------------------------------------------------------------------
    def _get_behavior_and_params(
        self,
    ) -> Tuple[torch.nn.Module, Dict[str, torch.Tensor]]:
        """
        Construct a training Behavior instance and extract its parameters.

        Returns
        -------
        behavior :
            The behavior module obtained from
            ``model.get_model_training_instance(scenario=...)``.
        params :
            A dictionary mapping parameter names to tensors, as returned by
            ``behavior.named_parameters()``.
        """
        behavior = self.model.get_model_training_instance(scenario=self.scenario)
        # Ensure we're in training mode for gradient computations
        behavior.train()

        # Convert named_parameters() iterator to a plain dict[str, Tensor]
        # Only keep true model parameters (exclude scenario-related tensors)
        params = {
            name: p
            for name, p in behavior.named_parameters()
            if name.startswith("params.")
        }
        return behavior, params

    def _validate_relative_space_params(
        self, param_names: list[str], *, raise_on_zero: bool
    ) -> list[str]:
        """Validate parameters for the ``relative`` / ``log`` schemes.

        The relative step :math:`p \\mapsto p\\,e^{\\pm\\varepsilon}` and the
        logarithmic derivative :math:`\\partial f/\\partial\\log|p|` are both
        defined through :math:`\\log|p|`, so a zero-valued parameter has no
        valid perturbation (numerical) and no finite log-derivative (autograd).
        Negative parameters are handled by the sign-preserving transform
        :math:`\\mathrm{sign}(p)\\,e^{\\log|p|\\pm\\varepsilon}` and only warn.

        Parameters
        ----------
        param_names : list[str]
            Parameter names to validate.
        raise_on_zero : bool
            If True, raise on any zero-valued parameter (numerical, which
            cannot take a log-space step of zero). If False, return the
            zero-valued names so the caller can handle them (autograd, which
            reports the column as an exact zero).

        Returns
        -------
        list[str]
            The zero-valued parameter names.

        Raises
        ------
        ValueError
            If ``raise_on_zero`` and any parameter equals zero.

        Warns
        -----
        UserWarning
            If any parameter is negative.
        """
        zero_params = [n for n in param_names if self.model.parameters[n] == 0]
        negative_params = [n for n in param_names if self.model.parameters[n] < 0]

        if negative_params:
            warnings.warn(
                f"Found negative parameters in relative/log mode: {negative_params}. "
                "Using sign-preserving transformation: "
                "sign(p) * exp(log(abs(p)) +/- epsilon)."
            )

        if raise_on_zero and zero_params:
            raise ValueError(
                "Cannot use parameter_space in {'relative', 'log'} with zero "
                f"parameters: {zero_params}. Use 'direct', or exclude them via "
                "param_names."
            )

        return zero_params

    # ------------------------------------------------------------------
    # Abstract API
    # ------------------------------------------------------------------
    def compute(self, *args, **kwargs):
        """
        Placeholder for subclasses to implement Jacobian computation.

        Subclasses should implement this method with a consistent interface,
        for example:

        ``compute(loss_fn=..., mode=...)``.
        """
        raise NotImplementedError(
            "JacobianBase.compute() must be implemented by subclasses"
        )

    # ------------------------------------------------------------------
    # Helper methods for output format conversion
    # ------------------------------------------------------------------
    def to_tensor(
        self,
        jacobian_dict: Dict[str, torch.Tensor] | None = None,
        param_order: list[str] | None = None,
        flatten: bool = True,
    ) -> torch.Tensor:
        """Convert Jacobian dict to a single tensor with parameters on the last axis.

        Parameters
        ----------
        jacobian_dict : dict[str, torch.Tensor], optional
            Jacobian dictionary mapping parameter names to gradient tensors.
            If None, uses self.jacobian if available.
        param_order : list[str], optional
            Order of parameters for columns. If None, uses dict iteration order.
        flatten : bool, default True
            If True, collapse the loss shape into a single axis. If False, keep
            the native loss shape, e.g. a ``(timesteps, variables)`` loss yields
            a ``(timesteps, variables, num_params)`` tensor.

        Returns
        -------
        torch.Tensor
            If ``flatten`` is True, a 2D tensor of shape (loss_elements, num_params)
            where loss_elements is the number of elements in the loss output. If
            ``flatten`` is False, a tensor of shape (*loss_shape, num_params). The
            two are related by ``to_tensor(flatten=False)`` equals
            ``to_tensor(flatten=True).reshape(*loss_shape, num_params)``.

        Raises
        ------
        ValueError
            If loss structure is incompatible (e.g., different shapes per parameter)
            or if jacobian_dict is None and self.jacobian is not set.
        """
        if jacobian_dict is None:
            if not hasattr(self, "jacobian") or self.jacobian is None:
                raise ValueError(
                    "jacobian_dict must be provided or self.jacobian must be set"
                )
            jacobian_dict = self.jacobian

        if not jacobian_dict:
            return torch.tensor([])

        if param_order is None:
            param_order = list(jacobian_dict.keys())

        # Check that all gradients have the same shape
        first_grad = jacobian_dict[param_order[0]]
        loss_shape = first_grad.shape

        for pname in param_order:
            if pname not in jacobian_dict:
                raise ValueError(f"Parameter {pname} not found in jacobian dict")
            grad = jacobian_dict[pname]
            if grad.shape != loss_shape:
                raise ValueError(
                    f"Incompatible loss structure: parameter {pname} has shape {grad.shape}, "
                    f"expected {loss_shape}"
                )

        # Stack gradients into a single tensor, parameters on the last axis.
        if flatten:
            gradients = [jacobian_dict[pname].flatten() for pname in param_order]
            return torch.stack(gradients, dim=1)  # (loss_elements, num_params)
        gradients = [jacobian_dict[pname] for pname in param_order]
        return torch.stack(gradients, dim=-1)  # (*loss_shape, num_params)

    def to_pandas(
        self,
        jacobian_dict: Dict[str, torch.Tensor] | None = None,
        timesteps: int | None = None,
        variable_names: list[str] | None = None,
        param_order: list[str] | None = None,
    ) -> "pd.DataFrame":
        """Convert Jacobian dict to pandas DataFrame with MultiIndex (single-index
        if losses are scalar or 1D).

        This function attempts to unflatten the loss structure into a DataFrame.
        It requires knowledge of the loss structure (timesteps, variable names).

        Parameters
        ----------
        jacobian_dict : dict[str, torch.Tensor], optional
            Jacobian dictionary mapping parameter names to gradient tensors.
            If None, uses self.jacobian if available.
        timesteps : int, optional
            Number of timesteps (if loss has time dimension).
        variable_names : list[str], optional
            Names of variables (if loss has variable dimension).
        param_order : tuple[str], optional
            Order of parameters for columns. If None, uses dict iteration order.

        Notes
        -----
        This function prefers to use variable_names over timesteps if both are provided
        and the shape of the gradient is a 1D vector

        Returns
        -------
        pd.DataFrame
            DataFrame with MultiIndex rows and parameter columns.

        Raises
        ------
        ValueError
            If jacobian_dict is None and self.jacobian is not set.
        """
        if jacobian_dict is None:
            if not hasattr(self, "jacobian") or self.jacobian is None:
                raise ValueError(
                    "jacobian_dict must be provided or self.jacobian must be set"
                )
            jacobian_dict = self.jacobian

        if not jacobian_dict:
            return pd.DataFrame()

        if param_order is None:
            param_order = tuple(jacobian_dict.keys())

        # Get first gradient to determine structure
        first_grad = jacobian_dict[param_order[0]]
        loss_shape = first_grad.shape

        # Validate shape of given items

        if len(loss_shape) == 0:
            # If scalar, use flat index
            index = pd.Index(["loss"], name="element")
        elif len(loss_shape) == 1:
            if timesteps is None and variable_names is None:
                # 1D loss: infer if no structure is provided
                index = pd.RangeIndex(first_grad.numel(), name="element")
            elif variable_names is None:
                # Without variable names, assume timesteps
                index = pd.RangeIndex(timesteps, name="timestep")
                if len(index) != loss_shape[0]:
                    raise ValueError(
                        f"Incompatible loss structure: timesteps {timesteps} does not match shape {loss_shape}"
                    )
            else:
                # With variable names, use provided names
                index = pd.Index(variable_names, name="variable")
                if len(index) != loss_shape[0]:
                    raise ValueError(
                        f"Incompatible loss structure: variable names {variable_names} does not match shape {loss_shape}"
                    )
        elif len(loss_shape) == 2:
            # 2D loss: assume (timesteps, variables) structure
            if timesteps is None:
                timesteps = loss_shape[0]
            if variable_names is None:
                variable_names = [f"var_{i}" for i in range(loss_shape[1])]

            index = pd.MultiIndex.from_product(
                [range(timesteps), variable_names], names=["timestep", "variable"]
            )
            if len(index) != loss_shape[0] * loss_shape[1]:
                raise ValueError(
                    f"Incompatible loss structure: timesteps {timesteps} and variable names {variable_names} do not match shape {loss_shape}"
                )
        else:
            raise ValueError(f"Unsupported loss shape: {loss_shape}")

        # Convert each parameter's gradient to numpy array
        data = {}
        for pname in param_order:
            grad = jacobian_dict[pname]
            if grad.shape != loss_shape:
                raise ValueError(
                    f"Incompatible loss structure: parameter {pname} has shape {grad.shape}, "
                    f"expected {loss_shape}"
                )
            # Flatten gradient to match index length
            data[pname] = grad.flatten().detach().cpu().numpy()

        data = pd.DataFrame(data, index=index)
        return data
