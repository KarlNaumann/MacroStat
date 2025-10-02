"""
ECO3ECO3IOPC model for Marco Veronese Pasarella's ECO-3IO-PC model
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.model import Model
from macrostat.models.ECO3IOPC.behavior import BehaviorECO3IOPC
from macrostat.models.ECO3IOPC.parameters import ParametersECO3IOPC
from macrostat.models.ECO3IOPC.scenarios import ScenariosECO3IOPC
from macrostat.models.ECO3IOPC.variables import VariablesECO3IOPC

logger = logging.getLogger(__name__)


class ECO3IOPC(Model):
    """ECO3IOPC model class for Marco Veronese Pasarella's ECO-3IO-PC model"""

    version = "ECO3IOPC"

    def __init__(
        self,
        parameters: ParametersECO3IOPC | None = ParametersECO3IOPC(),
        variables: VariablesECO3IOPC | None = None,
        scenarios: ScenariosECO3IOPC | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the ECO3IOPC model.

        Parameters
        ----------
        parameters: ParametersECO3IOPC | None
            The parameters of the model. If None, default parameters will be used.
        variables: VariablesECO3IOPC | None
            The variables of the model. If None, default variables will be used.
        scenarios: ScenariosECO3IOPC | None
            The scenarios of the model. If None, default scenarios will be used.
        """
        if parameters is None:
            parameters = ParametersECO3IOPC()
        if variables is None:
            variables = VariablesECO3IOPC(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosECO3IOPC(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorECO3IOPC,
            *args,
            **kwargs,
        )
