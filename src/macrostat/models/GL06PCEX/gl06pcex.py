"""
PCEX model class for the Godley-Lavoie 2006 PCEX model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.model import Model
from macrostat.models.GL06PCEX.behavior import BehaviorGL06PCEX
from macrostat.models.GL06PCEX.parameters import ParametersGL06PCEX
from macrostat.models.GL06PCEX.scenarios import ScenariosGL06PCEX
from macrostat.models.GL06PCEX.variables import VariablesGL06PCEX

logger = logging.getLogger(__name__)


class GL06PCEX(Model):
    """PCEX model class for the Godley-Lavoie 2006 PCEX model."""

    version = "GL06PCEX"

    def __init__(
        self,
        parameters: ParametersGL06PCEX | None = ParametersGL06PCEX(),
        variables: VariablesGL06PCEX | None = None,
        scenarios: ScenariosGL06PCEX | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the PCEX model.

        Parameters
        ----------
        parameters: ParametersGL06PCEX | None
            The parameters of the model. If None, default parameters will be used.
        variables: VariablesGL06PCEX | None
            The variables of the model. If None, default variables will be used.
        scenarios: ScenariosGL06PCEX | None
            The scenarios of the model. If None, default scenarios will be used.
        """
        if parameters is None:
            parameters = ParametersGL06PCEX()
        if variables is None:
            variables = VariablesGL06PCEX(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosGL06PCEX(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorGL06PCEX,
            *args,
            **kwargs,
        )
