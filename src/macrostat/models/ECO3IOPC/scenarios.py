"""
Scenarios class for Marco Veronese Passarella's ECO-3IO-PC model
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.ECO3IOPC.parameters import ParametersECO3IOPC

logger = logging.getLogger(__name__)


class ScenariosECO3IOPC(Scenarios):
    """Scenarios class for Marco Veronese Passarella's ECO-3IO-PC model"""

    version = "ECO3IOPC"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersECO3IOPC | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of Marco Veronese Passarella's ECO-3IO-PC model"""

        if parameters is None:
            parameters = ParametersECO3IOPC()

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
