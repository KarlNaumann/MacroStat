"""
Model class for the Godley-Lavoie 2006 INSOUT model (Chapter 7).
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
from macrostat.models.GL06INSOUT.behavior import BehaviorGL06INSOUT
from macrostat.models.GL06INSOUT.parameters import ParametersGL06INSOUT
from macrostat.models.GL06INSOUT.scenarios import ScenariosGL06INSOUT
from macrostat.models.GL06INSOUT.variables import VariablesGL06INSOUT

logger = logging.getLogger(__name__)


class GL06INSOUT(Model):
    """Model class for the Godley-Lavoie 2006 INSOUT model.

    Implements the INSOUT model from Chapter 7 of Godley & Lavoie (2006),
    introducing inventory dynamics, wage-price dynamics, endogenous bank
    interest rates, and a 5-sector balance sheet structure.
    """

    version = "GL06INSOUT"

    def __init__(
        self,
        parameters: ParametersGL06INSOUT | None = ParametersGL06INSOUT(),
        variables: VariablesGL06INSOUT | None = None,
        scenarios: ScenariosGL06INSOUT | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the INSOUT model.

        Parameters
        ----------
        parameters : ParametersGL06INSOUT | None
            The parameters of the model. If None, default parameters are used.
        variables : VariablesGL06INSOUT | None
            The variables of the model. If None, default variables are used.
        scenarios : ScenariosGL06INSOUT | None
            The scenarios of the model. If None, default scenarios are used.
        """
        if parameters is None:
            parameters = ParametersGL06INSOUT()
        if variables is None:
            variables = VariablesGL06INSOUT(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosGL06INSOUT(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorGL06INSOUT,
            *args,
            **kwargs,
        )
