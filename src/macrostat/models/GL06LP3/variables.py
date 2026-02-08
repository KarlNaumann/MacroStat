"""
Variables class for the Godley-Lavoie 2006 LP3 model.
"""

# Copyright (c) 2025 Karl Naumann-Woleske
# Author: Karl Naumann-Woleske <karl@naumannwoleske.com>
# SPDX-License-Identifier: MIT

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.models.GL06LP2.variables import VariablesGL06LP2
from macrostat.models.GL06LP3.parameters import ParametersGL06LP3

logger = logging.getLogger(__name__)


class VariablesGL06LP3(VariablesGL06LP2):
    """Variables class for the Godley-Lavoie 2006 LP3 model.

    Extends LP2 variables with the PublicSectorBorrowingRequirement
    variable.
    """

    version = "GL06LP3"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersGL06LP3 | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables of the Godley-Lavoie 2006 LP3 model."""

        if parameters is None:
            parameters = ParametersGL06LP3()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_variables(self):
        """Return the default variables information dictionary.

        Inherits all GL06LP2 variables and adds
        PublicSectorBorrowingRequirement.
        """
        variables = super().get_default_variables()
        variables.update(
            {
                "PublicSectorBorrowingRequirement": {
                    "notation": r"PSBR(t)",
                    "unit": "USD",
                    "history": 0,
                    "sectors": ["Government"],
                    "sfc": [("Index", "Government")],
                },
            }
        )
        return variables
