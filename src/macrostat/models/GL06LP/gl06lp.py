"""
LP model class for the Godley-Lavoie 2006 LP model.
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
from macrostat.models.GL06LP.behavior import BehaviorGL06LP
from macrostat.models.GL06LP.parameters import ParametersGL06LP
from macrostat.models.GL06LP.scenarios import ScenariosGL06LP
from macrostat.models.GL06LP.variables import VariablesGL06LP

logger = logging.getLogger(__name__)


class GL06LP(Model):
    """LP model class for the Godley-Lavoie 2006 LP model.

    This model introduces long-term bonds, capital gains, and liquidity
    preference into the SFC framework. It extends Model PC by adding a
    bond market with exogenous bond prices.
    """

    version = "GL06LP"

    def __init__(
        self,
        parameters: ParametersGL06LP | None = ParametersGL06LP(),
        variables: VariablesGL06LP | None = None,
        scenarios: ScenariosGL06LP | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the LP model.

        Parameters
        ----------
        parameters : ParametersGL06LP | None
            The parameters of the model. If None, default parameters
            will be used.
        variables : VariablesGL06LP | None
            The variables of the model. If None, default variables
            will be used.
        scenarios : ScenariosGL06LP | None
            The scenarios of the model. If None, default scenarios
            will be used.
        """
        if parameters is None:
            parameters = ParametersGL06LP()
        if variables is None:
            variables = VariablesGL06LP(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosGL06LP(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorGL06LP,
            *args,
            **kwargs,
        )
