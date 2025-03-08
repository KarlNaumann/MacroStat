"""
SIM model class for the Godley-Lavoie 2006 SIM model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.model import Model
from macrostat.models.godleylavoie2006_SIM.parameters import ParametersSIM
from macrostat.models.godleylavoie2006_SIM.scenarios import ScenariosSIM
from macrostat.models.godleylavoie2006_SIM.variables import VariablesSIM

logger = logging.getLogger(__name__)


class SIM(Model):
    """SIM model class for the Godley-Lavoie 2006 SIM model."""

    version = "SIM"

    def __init__(
        self,
        parameters: ParametersSIM | None = None,
        variables: VariablesSIM | None = None,
        scenarios: ScenariosSIM | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the SIM model.

        Parameters
        ----------
        parameters: ParametersSIM | None
            The parameters of the model. If None, default parameters will be used.
        variables: VariablesSIM | None
            The variables of the model. If None, default variables will be used.
        scenarios: ScenariosSIM | None
            The scenarios of the model. If None, default scenarios will be used.
        """
        if parameters is None:
            parameters = ParametersSIM()
        if variables is None:
            variables = VariablesSIM(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosSIM(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            *args,
            **kwargs,
        )
