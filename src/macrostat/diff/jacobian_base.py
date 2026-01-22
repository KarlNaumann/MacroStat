"""
Base class and utilities for Jacobian computation in MacroStat.

This module provides shared functionality for working with MacroStat models
in a way that is compatible with PyTorch's autograd and torch.func APIs.
"""

from __future__ import annotations

from typing import Dict, Tuple

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
