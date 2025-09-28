"""
IOPC model for Marco Veronese Pasarella's 3IO-PC model
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.model import Model
from macrostat.models.IOPC.behavior import BehaviorIOPC
from macrostat.models.IOPC.parameters import ParametersIOPC
from macrostat.models.IOPC.scenarios import ScenariosIOPC
from macrostat.models.IOPC.variables import VariablesIOPC

logger = logging.getLogger(__name__)


class IOPC(Model):
    """IOPC model class for Marco Veronese Pasarella's 3IO-PC model"""

    version = "IOPC"

    def __init__(
        self,
        parameters: ParametersIOPC | None = ParametersIOPC(),
        variables: VariablesIOPC | None = None,
        scenarios: ScenariosIOPC | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the IOPC model.

        Parameters
        ----------
        parameters: ParametersIOPC | None
            The parameters of the model. If None, default parameters will be used.
        variables: VariablesIOPC | None
            The variables of the model. If None, default variables will be used.
        scenarios: ScenariosIOPC | None
            The scenarios of the model. If None, default scenarios will be used.
        """
        if parameters is None:
            parameters = ParametersIOPC()
        if variables is None:
            variables = VariablesIOPC(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosIOPC(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorIOPC,
            *args,
            **kwargs,
        )
