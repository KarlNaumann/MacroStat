"""
Godley-Lavoie 2006 LP3 model (Endogenous Government Spending
via Fiscal Rule).
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

from macrostat.models.GL06LP3.behavior import BehaviorGL06LP3
from macrostat.models.GL06LP3.gl06lp3 import GL06LP3
from macrostat.models.GL06LP3.parameters import ParametersGL06LP3
from macrostat.models.GL06LP3.scenarios import ScenariosGL06LP3
from macrostat.models.GL06LP3.variables import VariablesGL06LP3

__all__ = [
    "GL06LP3",
    "BehaviorGL06LP3",
    "ParametersGL06LP3",
    "ScenariosGL06LP3",
    "VariablesGL06LP3",
]
