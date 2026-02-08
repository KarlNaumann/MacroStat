"""
LP3 model class for the Godley-Lavoie 2006 LP3 model.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.model import Model
from macrostat.models.GL06LP3.behavior import BehaviorGL06LP3
from macrostat.models.GL06LP3.parameters import ParametersGL06LP3
from macrostat.models.GL06LP3.scenarios import ScenariosGL06LP3
from macrostat.models.GL06LP3.variables import VariablesGL06LP3

logger = logging.getLogger(__name__)


class GL06LP3(Model):
    """LP3 model class for the Godley-Lavoie 2006 LP3 model.

    Extends Model LP2 by making government expenditures endogenous.
    When the deficit-to-GDP ratio (PSBR/Y) exceeds a threshold, the
    government enacts fiscal austerity. This introduces hysteresis —
    the steady state becomes path-dependent.
    """

    version = "GL06LP3"

    def __init__(
        self,
        parameters: ParametersGL06LP3 | None = ParametersGL06LP3(),
        variables: VariablesGL06LP3 | None = None,
        scenarios: ScenariosGL06LP3 | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the LP3 model.

        Parameters
        ----------
        parameters : ParametersGL06LP3 | None
            The parameters of the model. If None, default parameters
            will be used.
        variables : VariablesGL06LP3 | None
            The variables of the model. If None, default variables
            will be used.
        scenarios : ScenariosGL06LP3 | None
            The scenarios of the model. If None, default scenarios
            will be used.
        """
        if parameters is None:
            parameters = ParametersGL06LP3()
        if variables is None:
            variables = VariablesGL06LP3(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosGL06LP3(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorGL06LP3,
            *args,
            **kwargs,
        )
