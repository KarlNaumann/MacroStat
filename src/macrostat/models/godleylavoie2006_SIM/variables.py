"""
Variables class for the Godley-Lavoie 2006 SIM model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.variables import Variables
from macrostat.models.godleylavoie2006_SIM.parameters import ParametersSIM

logger = logging.getLogger(__name__)


class VariablesSIM(Variables):
    """Variables class for the Godley-Lavoie 2006 SIM model."""

    version = "SIM"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersSIM | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables of the Godley-Lavoie 2006 SIM model."""

        if parameters is None:
            parameters = ParametersSIM()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_variables(self):
        """Return the default variables information dictionary."""
        return {
            "V1": {
                "notation": "",
                "unit": "",
                "history": 0,
            },
        }
