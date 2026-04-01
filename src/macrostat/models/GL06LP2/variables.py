"""
Variables class for the Godley-Lavoie 2006 LP2 model.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.models.GL06LP2.parameters import ParametersGL06LP2
from macrostat.models.GL06LP.variables import VariablesGL06LP as _VariablesGL06LP

logger = logging.getLogger(__name__)


class VariablesGL06LP2(_VariablesGL06LP):
    """Variables class for the Godley-Lavoie 2006 LP2 model.

    Extends LP variables with the TargetProportion variable.
    """

    version = "GL06LP2"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersGL06LP2 | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables of the Godley-Lavoie 2006 LP2 model."""

        if parameters is None:
            parameters = ParametersGL06LP2()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_variables(self):
        """Return the default variables information dictionary.

        Inherits all GL06LP variables and adds TargetProportion.
        """
        variables = super().get_default_variables()
        variables.update(
            {
                "TargetProportion": {
                    "notation": r"TP(t)",
                    "unit": ".",
                    "history": 0,
                    "sectors": ["Government"],
                    "sfc": [("Index", "Government")],
                },
            }
        )
        return variables
