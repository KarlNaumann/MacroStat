# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

"""Linear adding-up constraints for parameter groups.

This module implements residual parameterization for linear adding-up
constraints of the form ``sum(params) == target``. One parameter in
each group is designated as the *derived* (residual) parameter and is
recomputed as ``target - sum(free_params)``. This makes the derived
parameter a non-leaf tensor with zero autograd gradient; sensitivity
flows into the free parameters only.

Two abstraction layers are provided:

* :class:`LinearConstraint` declares the constraint in terms of
  canonical (dotted) parameter names.
* :class:`ParameterLocation` and :class:`ConstraintResolver` map those
  names to positions inside the vectorized tensor dictionary that
  :meth:`macrostat.core.parameters.Parameters.vectorize_parameters`
  produces. This is what lets the same constraint apply to scalar,
  sector-indexed vector, and sector-indexed matrix parameters
  uniformly at step time.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch


class ConstraintError(ValueError):
    """Raised for malformed or unresolvable parameter constraints.

    Subclasses :class:`ValueError` so existing code that catches
    ``ValueError`` continues to work.
    """


@dataclass(frozen=True)
class ParameterLocation:
    """Where a named parameter lives in a vectorized tensor dict.

    A :class:`ParameterLocation` is a small value object produced by
    :class:`ConstraintResolver`. It records the key under which a
    parameter's tensor appears in the step-time parameter dictionary,
    together with an index if the parameter is one element of a larger
    tensor (for example one sector of a 1-D sector-indexed tensor, or
    one cell of a 2-D sector-by-sector matrix).

    Parameters
    ----------
    tensor_key : str
        Key into the step-time tensor dict (the dict produced by
        :meth:`macrostat.core.parameters.Parameters.vectorize_parameters`).
    index : tuple[int, ...] | None
        ``None`` for scalar slots (the whole dict entry *is* the
        parameter). ``(i,)`` for 1-D sector-indexed parameters.
        ``(i, j)`` for 2-D sector-by-sector parameters.
    """

    tensor_key: str
    index: tuple[int, ...] | None

    def read(self, params: dict[str, torch.Tensor]) -> torch.Tensor:
        """Return the tensor slice corresponding to this location.

        Parameters
        ----------
        params : dict[str, torch.Tensor]
            Step-time parameter dictionary.

        Returns
        -------
        torch.Tensor
            The 0-D tensor at this location. For a scalar slot this is
            the whole dict entry; for indexed slots it is the result of
            indexing the full tensor, which autograd tracks.
        """
        t = params[self.tensor_key]
        return t if self.index is None else t[self.index]

    def write(self, params: dict[str, torch.Tensor], value: torch.Tensor) -> None:
        """Write ``value`` into this location, preserving differentiability.

        For scalar slots this is a plain dict rebinding. For indexed
        slots it uses :func:`torch.Tensor.index_put` with
        ``accumulate=False``, which is out-of-place and differentiable:
        gradients flow into ``value`` and into the untouched entries of
        the base tensor.

        The caller must use the dict after this call and not retain
        references to the old tensor stored under ``tensor_key``,
        because indexed writes rebind the dict entry to a new tensor.

        Parameters
        ----------
        params : dict[str, torch.Tensor]
            Step-time parameter dictionary. Modified in place.
        value : torch.Tensor
            Scalar tensor to store at this location. For indexed
            writes the value is reshaped to match the single-element
            slice expected by :func:`torch.Tensor.index_put`.
        """
        if self.index is None:
            params[self.tensor_key] = value
            return

        base = params[self.tensor_key]
        idx_tuple = tuple(
            torch.tensor([k], dtype=torch.long, device=base.device) for k in self.index
        )
        params[self.tensor_key] = base.index_put(
            idx_tuple, value.reshape(1), accumulate=False
        )


class ConstraintResolver:
    """Map canonical parameter names to :class:`ParameterLocation` objects.

    A resolver is built once by
    :meth:`macrostat.core.parameters.Parameters.get_constraint_resolver`
    and reused across step-time constraint applications. It is the
    seam that isolates :class:`LinearConstraint` from the details of
    how ``Parameters.vectorize_parameters`` lays parameters out in the
    step-time tensor dict.

    Parameters
    ----------
    mapping : dict[str, ParameterLocation]
        Canonical (dotted) parameter name → location in the vectorized
        tensor dict. Ownership of the mapping passes to the resolver;
        do not mutate it after construction.
    """

    def __init__(self, mapping: dict[str, ParameterLocation]):
        self._map = dict(mapping)

    def locate(self, name: str) -> ParameterLocation:
        """Return the :class:`ParameterLocation` for ``name``.

        Parameters
        ----------
        name : str
            Canonical (dotted) parameter name as it appears in the
            constraint declaration.

        Returns
        -------
        ParameterLocation
            The recorded location for this name.

        Raises
        ------
        ConstraintError
            If ``name`` is not known to this resolver.
        """
        if name not in self._map:
            raise ConstraintError(
                f"Unknown constraint parameter: {name!r}. "
                f"Known names: {sorted(self._map)}"
            )
        return self._map[name]

    def __contains__(self, name: str) -> bool:
        return name in self._map

    def __len__(self) -> int:
        return len(self._map)


@dataclass(frozen=True)
class LinearConstraint:
    """A linear adding-up constraint on a group of parameters.

    Enforces that a group of parameters sums to a target value via
    residual parameterization: the last parameter in ``param_names``
    is derived (computed as ``target - sum(free_params)``), while
    all others remain free.

    The constraint is declared in terms of canonical (dotted)
    parameter names and is evaluated against a
    :class:`ConstraintResolver` at step time, which lets the same
    constraint apply to scalar, vector, or matrix parameter layouts
    uniformly.

    Parameters
    ----------
    param_names : tuple[str, ...]
        Names of all parameters in this constraint group.
        The **last** element is the derived (residual) parameter.
        Lists are coerced to tuples.
    target : float
        The value that all parameters in the group must sum to.
        Must be finite and not a bool.

    Raises
    ------
    ConstraintError
        If ``param_names`` has fewer than two entries, contains
        duplicates, contains non-string names, or if ``target`` is
        non-finite or a bool.

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

    def __post_init__(self):
        # Coerce lists (and other sequences) to tuples. Frozen
        # dataclass requires object.__setattr__ for mutation.
        if not isinstance(self.param_names, tuple):
            try:
                coerced = tuple(self.param_names)
            except TypeError as exc:
                raise ConstraintError(
                    f"param_names must be a sequence of strings, "
                    f"got {type(self.param_names).__name__}"
                ) from exc
            object.__setattr__(self, "param_names", coerced)

        if len(self.param_names) < 2:
            raise ConstraintError(
                f"LinearConstraint needs at least two parameter names "
                f"(one free and one derived); got {self.param_names!r}"
            )

        for name in self.param_names:
            if not isinstance(name, str):
                raise ConstraintError(
                    f"LinearConstraint param_names must all be strings; "
                    f"got {name!r} of type {type(name).__name__}"
                )

        if len(set(self.param_names)) != len(self.param_names):
            raise ConstraintError(
                f"LinearConstraint param_names must be unique; "
                f"got {self.param_names!r}"
            )

        # Reject bool explicitly because bool is a subclass of int.
        if isinstance(self.target, bool):
            raise ConstraintError(
                f"LinearConstraint target must be a float, not a bool; "
                f"got {self.target!r}"
            )
        if not isinstance(self.target, (int, float)):
            raise ConstraintError(
                f"LinearConstraint target must be a number; "
                f"got {self.target!r} of type {type(self.target).__name__}"
            )
        if not math.isfinite(float(self.target)):
            raise ConstraintError(
                f"LinearConstraint target must be finite; " f"got {self.target!r}"
            )

    @property
    def free_params(self) -> tuple[str, ...]:
        """Parameter names that are free (all except the last)."""
        return self.param_names[:-1]

    @property
    def derived_param(self) -> str:
        """The parameter name that is derived from the others."""
        return self.param_names[-1]

    def apply(
        self,
        params: dict[str, torch.Tensor],
        resolver: ConstraintResolver,
    ) -> dict[str, torch.Tensor]:
        """Enforce the constraint on a vectorized tensor dict.

        Computes ``derived = target - sum(free_params)`` and writes
        the result into the location of the derived parameter. The
        operation is differentiable end-to-end: autograd flows
        ``-1`` gradients back to each free parameter and leaves the
        derived slot with zero gradient.

        The method mutates ``params`` in place and returns it.
        Callers must use the return value and must not retain
        references to individual tensors across the call, because
        indexed writes rebind the dict entry via
        :func:`torch.Tensor.index_put`.

        Parameters
        ----------
        params : dict[str, torch.Tensor]
            Step-time parameter dictionary, as produced by
            :meth:`macrostat.core.parameters.Parameters.vectorize_parameters`
            and possibly mutated by
            :meth:`macrostat.core.behavior.Behavior.apply_parameter_shocks`.
        resolver : ConstraintResolver
            Resolver mapping the constraint's canonical parameter
            names to locations in ``params``.

        Returns
        -------
        dict[str, torch.Tensor]
            The same dictionary with the derived parameter updated.

        Raises
        ------
        ConstraintError
            If any of the constraint's names are not known to the
            resolver.
        """
        free_sum = sum(resolver.locate(name).read(params) for name in self.free_params)
        new_val = free_sum.new_tensor(self.target) - free_sum
        resolver.locate(self.derived_param).write(params, new_val)
        return params
