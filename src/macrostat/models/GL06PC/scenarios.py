"""
Scenarios class for the Godley-Lavoie 2006 PC model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.GL06PC.parameters import ParametersGL06PC

logger = logging.getLogger(__name__)


class ScenariosGL06PC(Scenarios):
    """Scenarios class for the Godley-Lavoie 2006 PC model."""

    version = "GL06PC"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersGL06PC | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of the Godley-Lavoie 2006 PC model."""

        if parameters is None:
            parameters = ParametersGL06PC()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_scenario_values(self):
        """Return the default scenario values."""
        return {
            "GovernmentDemand": 20,
            "WageRate": 1,
            "InterestRate": 0.025,
            "PropensityToConsumeIncome_add": 0,
        }
