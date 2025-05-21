"""
PCEX2 model class for the Godley-Lavoie 2006 PCEX2 model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.model import Model
from macrostat.models.GL06PCEX2.behavior import BehaviorGL06PCEX2
from macrostat.models.GL06PCEX2.parameters import ParametersGL06PCEX2
from macrostat.models.GL06PCEX2.scenarios import ScenariosGL06PCEX2
from macrostat.models.GL06PCEX2.variables import VariablesGL06PCEX2

logger = logging.getLogger(__name__)


class GL06PCEX2(Model):
    """PCEX2 model class for the Godley-Lavoie 2006 PCEX2 model."""

    version = "GL06PCEX2"

    def __init__(
        self,
        parameters: ParametersGL06PCEX2 | None = ParametersGL06PCEX2(),
        variables: VariablesGL06PCEX2 | None = None,
        scenarios: ScenariosGL06PCEX2 | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the PCEX2 model.

        Parameters
        ----------
        parameters: ParametersGL06PCEX2 | None
            The parameters of the model. If None, default parameters will be used.
        variables: VariablesGL06PCEX2 | None
            The variables of the model. If None, default variables will be used.
        scenarios: ScenariosGL06PCEX2 | None
            The scenarios of the model. If None, default scenarios will be used.
        """
        if parameters is None:
            parameters = ParametersGL06PCEX2()
        if variables is None:
            variables = VariablesGL06PCEX2(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosGL06PCEX2(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorGL06PCEX2,
            *args,
            **kwargs,
        )
