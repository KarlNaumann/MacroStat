"""
SIM model class for the Godley-Lavoie 2006 SIM model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.model import Model
from macrostat.models.GL06SIM.behavior import BehaviorGL06SIM
from macrostat.models.GL06SIM.parameters import ParametersGL06SIM
from macrostat.models.GL06SIM.scenarios import ScenariosGL06SIM
from macrostat.models.GL06SIM.variables import VariablesGL06SIM

logger = logging.getLogger(__name__)


class GL06SIM(Model):
    """SIM model class for the Godley-Lavoie 2006 SIM model."""

    version = "GL06SIM"

    def __init__(
        self,
        parameters: ParametersGL06SIM | None = ParametersGL06SIM(),
        variables: VariablesGL06SIM | None = None,
        scenarios: ScenariosGL06SIM | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the SIM model.

        Parameters
        ----------
        parameters: ParametersGL06SIM | None
            The parameters of the model. If None, default parameters will be used.
        variables: VariablesGL06SIM | None
            The variables of the model. If None, default variables will be used.
        scenarios: ScenariosGL06SIM | None
            The scenarios of the model. If None, default scenarios will be used.
        """
        if parameters is None:
            parameters = ParametersGL06SIM()
        if variables is None:
            variables = VariablesGL06SIM(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosGL06SIM(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorGL06SIM,
            *args,
            **kwargs,
        )
