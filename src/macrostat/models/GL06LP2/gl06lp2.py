"""
LP2 model class for the Godley-Lavoie 2006 LP2 model.
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
from macrostat.models.GL06LP2.behavior import BehaviorGL06LP2
from macrostat.models.GL06LP2.parameters import ParametersGL06LP2
from macrostat.models.GL06LP2.scenarios import ScenariosGL06LP2
from macrostat.models.GL06LP2.variables import VariablesGL06LP2

logger = logging.getLogger(__name__)


class GL06LP2(Model):
    """LP2 model class for the Godley-Lavoie 2006 LP2 model.

    Extends Model LP by making bond prices endogenous through a
    target-proportion mechanism. The Treasury adjusts bond prices when
    the share of bonds in household government debt moves outside a
    target range. Households form adaptive expectations about future
    bond prices.
    """

    version = "GL06LP2"

    def __init__(
        self,
        parameters: ParametersGL06LP2 | None = ParametersGL06LP2(),
        variables: VariablesGL06LP2 | None = None,
        scenarios: ScenariosGL06LP2 | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the LP2 model.

        Parameters
        ----------
        parameters : ParametersGL06LP2 | None
            The parameters of the model. If None, default parameters
            will be used.
        variables : VariablesGL06LP2 | None
            The variables of the model. If None, default variables
            will be used.
        scenarios : ScenariosGL06LP2 | None
            The scenarios of the model. If None, default scenarios
            will be used.
        """
        if parameters is None:
            parameters = ParametersGL06LP2()
        if variables is None:
            variables = VariablesGL06LP2(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosGL06LP2(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorGL06LP2,
            *args,
            **kwargs,
        )
