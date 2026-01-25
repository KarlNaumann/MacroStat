"""
Autograd-based Jacobian computation for MacroStat models.

This module uses PyTorch's torch.func API (functional_call, jacrev, jacfwd)
to compute Jacobians of a user-specified loss function with respect to the
parameters of a model's Behavior module.
"""

from __future__ import annotations

from typing import Callable, Dict, Literal

import torch
from torch.func import functional_call, jacfwd, jacrev

from macrostat.diff.jacobian_base import JacobianBase

LossFn = Callable[[Dict[str, torch.Tensor]], torch.Tensor]


class JacobianAutograd(JacobianBase):
    """
    Compute Jacobians using PyTorch's autograd function transforms.

    Notes
    -----
    - Currently supports differentiation with respect to **parameters only**.
    - The loss function can return a scalar tensor or tensor of any shape.
    - Output structure matches numerical Jacobian exactly for plug-and-play interchangeability.
    """

    def compute(
        self,
        loss_fn: LossFn,
        mode: Literal["rev", "fwd"] = "rev",
        chunk_size: int | None = None,
    ) -> Dict[str, torch.Tensor]:
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

        def compute_loss(params: Dict[str, torch.Tensor]) -> torch.Tensor:
            output = functional_call(behavior, params, ())
            loss = loss_fn(output)
            return loss

        if mode == "rev":
            grads = jacrev(compute_loss)(base_params)
        else:  # mode == "fwd"
            grads = jacfwd(compute_loss)(base_params)

        # jacrev/jacfwd return a structure matching the inputs (dict[name -> tensor])
        # where names have "params." prefix. Strip prefix to match model.parameters format
        jacobian = {
            name.replace("params.", "", 1): g
            for name, g in grads.items()
        }
        self.jacobian = jacobian
        return jacobian
