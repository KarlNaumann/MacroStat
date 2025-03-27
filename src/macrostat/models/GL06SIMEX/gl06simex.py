"""
SIMEX model class for the Godley-Lavoie 2006 SIMEX model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.model import Model
from macrostat.models.GL06SIMEX.behavior import BehaviorGL06SIMEX
from macrostat.models.GL06SIMEX.parameters import ParametersGL06SIMEX
from macrostat.models.GL06SIMEX.scenarios import ScenariosGL06SIMEX
from macrostat.models.GL06SIMEX.variables import VariablesGL06SIMEX

logger = logging.getLogger(__name__)


class GL06SIMEX(Model):
    """SIMEX model class for the Godley-Lavoie 2006 SIMEX model."""

    version = "GL06SIMEX"

    def __init__(
        self,
        parameters: ParametersGL06SIMEX | None = ParametersGL06SIMEX(),
        variables: VariablesGL06SIMEX | None = None,
        scenarios: ScenariosGL06SIMEX | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the SIMEX model.

        Parameters
        ----------
        parameters: ParametersGL06SIMEX | None
            The parameters of the model. If None, default parameters will be used.
        variables: VariablesGL06SIMEX | None
            The variables of the model. If None, default variables will be used.
        scenarios: ScenariosGL06SIMEX | None
            The scenarios of the model. If None, default scenarios will be used.
        """
        if parameters is None:
            parameters = ParametersGL06SIMEX()
        if variables is None:
            variables = VariablesGL06SIMEX(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosGL06SIMEX(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorGL06SIMEX,
            *args,
            **kwargs,
        )
