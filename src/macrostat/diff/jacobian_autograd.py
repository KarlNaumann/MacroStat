"""
Autograd-based Jacobian computation for MacroStat models.

This module uses PyTorch's torch.func API (functional_call, jacrev, jacfwd)
to compute Jacobians of a user-specified loss function with respect to the
parameters of a model's Behavior module.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch.func import functional_call, jacfwd, jacrev

from macrostat.diff.jacobian_base import JacobianBase

LossFn = Callable[[dict[str, torch.Tensor]], torch.Tensor]


class JacobianAutograd(JacobianBase):
    """
    Compute Jacobians using PyTorch's autograd function transforms.

    Notes
    -----
    - Currently supports differentiation with respect to **parameters only**.
    - The loss function can return a scalar tensor or tensor of any shape.
    - Output structure matches the numerical Jacobian exactly for
      plug-and-play interchangeability.
    """

    def __init__(
        self,
        model,
        scenario: int | str = 0,
        parameter_space: Literal["direct", "log"] = "direct",
    ):
        """
        Parameters
        ----------
        model :
            A MacroStat model instance.
        scenario :
            Scenario index or name to use when constructing the behavior.
        parameter_space : {"direct", "log"}, optional
            Derivative to return, by default "direct". "direct" returns
            df/dp (autograd has no finite step, so "relative" would be
            identical and is not accepted here). "log" returns the
            logarithmic derivative df/dlog|p| = p * df/dp; a zero-valued
            parameter has no log-derivative and its column is reported as an
            exact zero (the numerical backend raises instead).
        """
        super().__init__(model, scenario)
        if parameter_space not in {"direct", "log"}:
            raise ValueError(
                f"Unsupported parameter_space '{parameter_space}'. "
                "JacobianAutograd supports 'direct' or 'log'."
            )
        self.parameter_space = parameter_space
        self._log_output = parameter_space == "log"

    def compute(
        self,
        loss_fn: LossFn,
        mode: Literal["rev", "fwd"] = "rev",
        chunk_size: int | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute the Jacobian of the loss with respect to the model parameters.

        Parameters
        ----------
        loss_fn :
            A callable that takes the output dictionary from the model
            (as returned by ``Behavior.forward``) and returns a tensor
            (scalar or any shape).
        mode :
            Autograd mode to use:

            - ``\"rev\"``: reverse-mode via ``torch.func.jacrev`` (default).
            - ``\"fwd\"``: forward-mode via ``torch.func.jacfwd``.
        chunk_size : int, optional
            Chunk size for processing large non-scalar outputs. If None,
            processes all elements at once. Only used for non-scalar losses.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary mapping parameter names to gradient tensors.
            The shape of each gradient tensor matches the shape of the
            loss function output.
        """
        if mode not in {"rev", "fwd"}:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'rev' or 'fwd'.")

        behavior, base_params = self._get_behavior_and_params()

        def compute_loss(params: dict[str, torch.Tensor]) -> torch.Tensor:
            return loss_fn(functional_call(behavior, params, ()))

        if mode == "rev":
            grads = jacrev(compute_loss)(base_params)
        else:  # mode == "fwd"
            grads = jacfwd(compute_loss)(base_params)

        # jacrev/jacfwd return a dict keyed by the ParameterDict leaves, each
        # prefixed with "params.". A single leaf may be an assembled sector-
        # indexed vector or matrix (e.g. an input-output coefficient matrix),
        # so it can carry several scalar model parameters at once. Decompose
        # each leaf back into its scalar free parameters via the constraint
        # resolver, so the output is keyed by the same free-parameter names as
        # the numerical Jacobian and every downstream tool sees one gradient
        # per scalar parameter. For a leaf gradient of shape
        # ``(*loss_shape, *param_shape)`` the leading axes are the loss and the
        # trailing axes the parameter, so an indexed slot reads ``leaf[..., i, j]``.
        leaf_grads = {name.replace("params.", "", 1): g for name, g in grads.items()}
        resolver = self.model.parameters.get_constraint_resolver()

        jacobian = {}
        for pname in self.model.parameters.get_free_param_names():
            location = resolver.locate(pname)
            leaf = leaf_grads[location.tensor_key]
            if location.index is None:
                jacobian[pname] = leaf
            else:
                jacobian[pname] = leaf[(...,) + location.index]

        if self._log_output:
            # df/dlog|p| = p * df/dp. Multiply by the signed parameter value
            # (theta=0 collapses the column to an exact zero, matching the
            # numerical backend's raised-on-zero contract via a zero column).
            # The shared validator emits the negative-theta warning for parity.
            self._validate_relative_space_params(list(jacobian), raise_on_zero=False)
            for pname, grad in jacobian.items():
                jacobian[pname] = grad * float(self.model.parameters[pname])

        self.jacobian = jacobian
        return jacobian
