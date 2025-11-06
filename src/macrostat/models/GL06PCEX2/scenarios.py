"""
Scenarios class for the Godley-Lavoie 2006 PCEX2 model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.GL06PCEX2.parameters import ParametersGL06PCEX2

logger = logging.getLogger(__name__)


class ScenariosGL06PCEX2(Scenarios):
    """Scenarios class for the Godley-Lavoie 2006 PCEX2 model."""

    version = "GL06PCEX2"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersGL06PCEX2 | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of the Godley-Lavoie 2006 PCEX2 model."""

        if parameters is None:
            parameters = ParametersGL06PCEX2()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_scenario_values(self):
        """Return the default scenario values."""
        sc = {
            "GovernmentDemand": 20,
            "WageRate": 1,
            "InterestRate": 0.025,
            "PropensityToConsumeIncome_add": 0,
        }

        for k in self.parameters.values.keys():
            sc[f"{k.replace('.', '_')}_add"] = 0.0

        return sc
