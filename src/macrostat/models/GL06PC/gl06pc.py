"""
PC model class for the Godley-Lavoie 2006 PC model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.model import Model
from macrostat.models.GL06PC.behavior import BehaviorGL06PC
from macrostat.models.GL06PC.parameters import ParametersGL06PC
from macrostat.models.GL06PC.scenarios import ScenariosGL06PC
from macrostat.models.GL06PC.variables import VariablesGL06PC

logger = logging.getLogger(__name__)


class GL06PC(Model):
    """PC model class for the Godley-Lavoie 2006 PC model."""

    version = "GL06PC"

    def __init__(
        self,
        parameters: ParametersGL06PC | None = ParametersGL06PC(),
        variables: VariablesGL06PC | None = None,
        scenarios: ScenariosGL06PC | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the PC model.

        Parameters
        ----------
        parameters: ParametersGL06PC | None
            The parameters of the model. If None, default parameters will be used.
        variables: VariablesGL06PC | None
            The variables of the model. If None, default variables will be used.
        scenarios: ScenariosGL06PC | None
            The scenarios of the model. If None, default scenarios will be used.
        """
        if parameters is None:
            parameters = ParametersGL06PC()
        if variables is None:
            variables = VariablesGL06PC(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosGL06PC(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorGL06PC,
            *args,
            **kwargs,
        )
