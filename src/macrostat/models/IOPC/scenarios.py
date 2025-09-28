"""
Scenarios class for Marco Veronese Passarella's 3IO-PC model
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.IOPC.parameters import ParametersIOPC

logger = logging.getLogger(__name__)


class ScenariosIOPC(Scenarios):
    """Scenarios class for Marco Veronese Passarella's 3IO-PC model"""

    version = "IOPC"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersIOPC | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of Marco Veronese Passarella's 3IO-PC model"""

        if parameters is None:
            parameters = ParametersIOPC()

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
            "WageRate": 0.4,
            "InterestRate": 0.025,
        }
        for k in self.parameters.values.keys():
            sc[f"{k.replace('.', '_')}_add"] = 0.0

        return sc
