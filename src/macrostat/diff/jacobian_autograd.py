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
    - The loss function must return a **scalar tensor**.
    """

    def compute(
        self,
        loss_fn: LossFn,
        mode: Literal["rev", "fwd"] = "rev",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the Jacobian of the loss with respect to the model parameters.

        Parameters
        ----------
        loss_fn :
            A callable that takes the output dictionary from the model
            (as returned by ``Behavior.forward``) and returns a **scalar**
            ``torch.Tensor`` loss.
        mode :
            Autograd mode to use:

            - ``\"rev\"``: reverse-mode via ``torch.func.jacrev`` (default).
            - ``\"fwd\"``: forward-mode via ``torch.func.jacfwd``.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary mapping parameter names to gradient tensors of the
            same shape as the corresponding parameters.
        """
        if mode not in {"rev", "fwd"}:
            raise ValueError(f"Unsupported mode '{mode}'. Use 'rev' or 'fwd'.")

        behavior, base_params = self._get_behavior_and_params()

        def compute_loss(params: Dict[str, torch.Tensor]) -> torch.Tensor:
            # Run the behavior with the provided parameters
            output = functional_call(behavior, params, ())
            loss = loss_fn(output)
            if loss.ndim != 0:
                raise ValueError(
                    "JacobianAutograd currently expects loss_fn to return a scalar "
                    f"tensor, but got shape {tuple(loss.shape)}."
                )
            return loss

        if mode == "rev":
            grads = jacrev(compute_loss)(base_params)
        else:  # mode == "fwd"
            grads = jacfwd(compute_loss)(base_params)

        # jacrev/jacfwd return a structure matching the inputs (dict[name -> tensor])
        # We ensure the output is a plain dict[str, Tensor]
        return {name: g for name, g in grads.items()}
