# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
"""Kirman ant recruitment SDE — Kirman (1993), QJE.

The macrostat.models.KirmansAnts module consists of the following classes

.. autosummary::
    :toctree: models/KirmansAnts

    KirmansAnts
    behavior
    parameters
    scenarios
    variables
"""

from .behavior import BehaviorKirmansAnts
from .kirmansants import KirmansAnts
from .parameters import ParametersKirmansAnts
from .scenarios import ScenariosKirmansAnts
from .variables import VariablesKirmansAnts

__all__ = [
    "KirmansAnts",
    "BehaviorKirmansAnts",
    "ParametersKirmansAnts",
    "VariablesKirmansAnts",
    "ScenariosKirmansAnts",
]
