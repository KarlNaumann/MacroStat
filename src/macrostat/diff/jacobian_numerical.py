"""
Numerical (finite-difference) Jacobian computation for MacroStat models.

This module approximates the gradient of a scalar loss with respect to the
parameters of a model's Behavior module using finite differences.
"""

from __future__ import annotations

from typing import Callable, Dict, Literal

import torch

from macrostat.diff.jacobian_base import JacobianBase

LossFn = Callable[[Dict[str, torch.Tensor]], torch.Tensor]


class JacobianNumerical(JacobianBase):
    """
    Compute Jacobians via finite differences.

    Notes
    -----
    - Currently supports differentiation with respect to **parameters only**.
    - The loss function must return a **scalar tensor**.
    """

    def __init__(self, model, scenario: int | str = 0, epsilon: float = 1e-5):
        super().__init__(model=model, scenario=scenario)
        self.epsilon = float(epsilon)

    def compute(
        self,
        loss_fn: LossFn,
        mode: Literal["central", "forward", "backward"] = "central",
    ) -> Dict[str, torch.Tensor]:
        """
        Compute the numerical Jacobian of the loss w.r.t. model parameters.

        Parameters
        ----------
        loss_fn :
            A callable that takes the output dictionary from the model
            (as returned by ``Behavior.forward``) and returns a **scalar**
            ``torch.Tensor`` loss.
        mode :
            Finite-difference scheme to use:

            - ``\"central\"``: central differences (default).
            - ``\"forward\"``: forward differences.
            - ``\"backward\"``: backward differences.

        Returns
        -------
        dict[str, torch.Tensor]
            A dictionary mapping parameter names to gradient tensors of the
            same shape as the corresponding parameters.
        """
        if mode not in {"central", "forward", "backward"}:
            raise ValueError(
                f"Unsupported mode '{mode}'. Use 'central', 'forward', or 'backward'."
            )

        behavior, base_params = self._get_behavior_and_params()

        def compute_loss(params: Dict[str, torch.Tensor]) -> torch.Tensor:
            # We re-use the same behavior instance but override parameters
            from torch.func import functional_call

            with torch.no_grad():
                output = functional_call(behavior, params, ())
                loss = loss_fn(output)
                if loss.ndim != 0:
                    raise ValueError(
                        "JacobianNumerical currently expects loss_fn to return a scalar "
                        f"tensor, but got shape {tuple(loss.shape)}."
                    )
                return loss

        epsilon = self.epsilon

        # Evaluate base loss once
        base_loss = compute_loss(base_params)

        jacobian: Dict[str, torch.Tensor] = {}

        for name, p in base_params.items():
            # Work on a flattened view for simplicity
            p_flat = p.detach().clone().reshape(-1)
            grad_flat = torch.empty_like(p_flat)

            for i in range(p_flat.numel()):
                # Prepare perturbed parameter vectors
                if mode in {"central", "forward"}:
                    p_pos = p_flat.clone()
                    p_pos[i] += epsilon
                    params_pos = dict(base_params)
                    params_pos[name] = p_pos.reshape_as(p)
                    loss_pos = compute_loss(params_pos)

                if mode in {"central", "backward"}:
                    p_neg = p_flat.clone()
                    p_neg[i] -= epsilon
                    params_neg = dict(base_params)
                    params_neg[name] = p_neg.reshape_as(p)
                    loss_neg = compute_loss(params_neg)

                if mode == "central":
                    grad_flat[i] = (loss_pos - loss_neg) / (2.0 * epsilon)
                elif mode == "forward":
                    grad_flat[i] = (loss_pos - base_loss) / epsilon
                else:  # backward
                    grad_flat[i] = (base_loss - loss_neg) / epsilon

            jacobian[name] = grad_flat.reshape_as(p)

        return jacobian
