"""
Godley-Lavoie 2006 LP2 model (Endogenous Bond Price with
Target Proportions).
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

from macrostat.models.GL06LP2.behavior import BehaviorGL06LP2
from macrostat.models.GL06LP2.gl06lp2 import GL06LP2
from macrostat.models.GL06LP2.parameters import ParametersGL06LP2
from macrostat.models.GL06LP2.scenarios import ScenariosGL06LP2
from macrostat.models.GL06LP2.variables import VariablesGL06LP2

__all__ = [
    "GL06LP2",
    "BehaviorGL06LP2",
    "ParametersGL06LP2",
    "ScenariosGL06LP2",
    "VariablesGL06LP2",
]
