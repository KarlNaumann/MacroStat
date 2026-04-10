# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

"""Top-level loss functions used by the user-guide example notebooks.

These functions live in a module (rather than inline in the notebooks)
because :class:`macrostat.diff.JacobianNumerical` runs its
finite-difference passes through ``multiprocessing`` with the ``spawn``
start method, which requires worker callables to be importable from a
module rather than defined in ``__main__`` (the notebook scope).
"""

import torch


def mean_nominal_output(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean of ``NominalOutput`` across timesteps.

    Used as a generic scalar loss for GL06INSOUT and later GL06 variants in
    the user-guide notebooks. Picks a single observable so that the example
    output stays interpretable.
    """
    return output["NominalOutput"].mean()


def mean_national_income(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """Mean of ``NationalIncome`` across timesteps.

    Used as the scalar loss for GL06SIMEX in the diagnostics notebook, which
    exposes national income rather than nominal output.
    """
    return output["NationalIncome"].mean()


def final_state_norm(output: dict[str, torch.Tensor]) -> torch.Tensor:
    """L2 norm of the final state vector for LINEAR2D.

    Used by the diagnostics notebook to demonstrate direct-space
    finite differences on a model with a zero parameter.
    """
    return output["State"][-1].pow(2).sum().sqrt()
