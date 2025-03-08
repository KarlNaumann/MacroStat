"""
Scenarios class for the Godley-Lavoie 2006 SIM model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.godleylavoie2006_SIM.parameters import ParametersSIM

logger = logging.getLogger(__name__)


class ScenariosSIM(Scenarios):
    """Scenarios class for the Godley-Lavoie 2006 SIM model."""

    version = "SIM"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersSIM | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of the Godley-Lavoie 2006 SIM model."""

        if parameters is None:
            parameters = ParametersSIM()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_scenario_values(self):
        """Return the default scenario values."""
        return {
            "V1": 0,
        }
