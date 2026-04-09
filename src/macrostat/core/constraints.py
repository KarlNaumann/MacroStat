# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

"""Linear adding-up constraints for parameter groups."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class LinearConstraint:
    """A linear adding-up constraint on a group of parameters.

    Enforces that a group of parameters sums to a target value via
    residual parameterization: the last parameter in ``param_names``
    is derived (computed as ``target - sum(free_params)``), while
    all others remain free.

    Parameters
    ----------
    param_names : tuple[str, ...]
        Names of all parameters in this constraint group.
        The **last** element is the derived (residual) parameter.
    target : float
        The value that all parameters in the group must sum to.

    Examples
    --------
    Portfolio constants must sum to 1:

    >>> c = LinearConstraint(
    ...     param_names=("Bills_Const", "Bonds_Const", "M1_Const"),
    ...     target=1.0,
    ... )
    >>> c.free_params
    ('Bills_Const', 'Bonds_Const')
    >>> c.derived_param
    'M1_Const'
    """

    param_names: tuple[str, ...]
    target: float

    @property
    def free_params(self) -> tuple[str, ...]:
        """Parameter names that are free (all except the last)."""
        return self.param_names[:-1]

    @property
    def derived_param(self) -> str:
        """The parameter name that is derived from the others."""
        return self.param_names[-1]

    def enforce(self, params: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Enforce the constraint by recomputing the derived parameter.

        This operation is differentiable: the derived parameter's value
        is a function of the free parameters, so autograd tracks the
        dependency (``d(derived)/d(free_i) = -1``).

        Parameters
        ----------
        params : dict[str, torch.Tensor]
            Mutable parameter dictionary. Modified in-place and returned.

        Returns
        -------
        dict[str, torch.Tensor]
            The same dictionary with the derived parameter updated.
        """
        free_sum = sum(params[name] for name in self.free_params)
        params[self.derived_param] = free_sum.new_tensor(self.target) - free_sum
        return params
