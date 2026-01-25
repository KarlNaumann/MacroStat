"""
Base class and utilities for Jacobian computation in MacroStat.

This module provides shared functionality for working with MacroStat models
in a way that is compatible with PyTorch's autograd and torch.func APIs.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, Tuple

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
        self, jacobian_dict: Dict[str, torch.Tensor] | None = None, param_order: list[str] | None = None
    ) -> torch.Tensor:
        """Convert Jacobian dict to single 2D tensor.

        Parameters
        ----------
        jacobian_dict : dict[str, torch.Tensor], optional
            Jacobian dictionary mapping parameter names to gradient tensors.
            If None, uses self.jacobian if available.
        param_order : list[str], optional
            Order of parameters for columns. If None, uses dict iteration order.

        Returns
        -------
        torch.Tensor
            2D tensor of shape (loss_elements, num_params) where loss_elements
            is the number of elements in the loss output.

        Raises
        ------
        ValueError
            If loss structure is incompatible (e.g., different shapes per parameter)
            or if jacobian_dict is None and self.jacobian is not set.
        """
        if jacobian_dict is None:
            if not hasattr(self, "jacobian") or self.jacobian is None:
                raise ValueError("jacobian_dict must be provided or self.jacobian must be set")
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

        # Stack gradients into a single tensor
        gradients = [jacobian_dict[pname].flatten() for pname in param_order]
        return torch.stack(gradients, dim=1)  # Shape: (loss_elements, num_params)

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
                raise ValueError("jacobian_dict must be provided or self.jacobian must be set")
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
                    raise ValueError(f"Incompatible loss structure: timesteps {timesteps} does not match shape {loss_shape}")
            else:
                # With variable names, use provided names
                index = pd.Index(variable_names, name="variable")
                if len(index) != loss_shape[0]:
                    raise ValueError(f"Incompatible loss structure: variable names {variable_names} does not match shape {loss_shape}")
        elif len(loss_shape) == 2:
            # 2D loss: assume (timesteps, variables) structure
            if timesteps is None:
                timesteps = loss_shape[0]
            if variable_names is None:
                variable_names = [f"var_{i}" for i in range(loss_shape[1])]
            
            index = pd.MultiIndex.from_product(
                [range(timesteps), variable_names],
                names=["timestep", "variable"]
            )
            if len(index) != loss_shape[0] * loss_shape[1]:
                raise ValueError(f"Incompatible loss structure: timesteps {timesteps} and variable names {variable_names} do not match shape {loss_shape}")
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
