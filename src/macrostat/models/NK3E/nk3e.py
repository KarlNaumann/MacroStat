"""
NK3E model class for the New Keynesian 3-Equation model.

Source: A New Keynesian 3-Equation Model — https://macrosimulation.org/a_new_keynesian_3_equation_model
"""

__author__ = ["Mitja Devetak"]
__credits__ = ["Mitja Devetak"]
__license__ = "MIT"
__maintainer__ = ["Mitja Devetak"]

import logging

from macrostat.core.model import Model
from macrostat.models.NK3E.behavior import BehaviorNK3E
from macrostat.models.NK3E.parameters import ParametersNK3E
from macrostat.models.NK3E.scenarios import ScenariosNK3E
from macrostat.models.NK3E.variables import VariablesNK3E

logger = logging.getLogger(__name__)


class NK3E(Model):
    """NK3E model class for the New Keynesian 3-Equation model.

    Description: A compact three-equation New Keynesian framework with an IS
    curve (goods demand), a New Keynesian Phillips curve (price-setting), and a
    monetary policy rule. Together these describe the joint dynamics of output,
    inflation, and the real interest rate.

    Source: A New Keynesian 3-Equation Model —
    https://macrosimulation.org/a_new_keynesian_3_equation_model
    """

    version = "NK3E"

    def __init__(
        self,
        parameters: ParametersNK3E | None = ParametersNK3E(),
        variables: VariablesNK3E | None = None,
        scenarios: ScenariosNK3E | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the NK3E model.

        Parameters
        ----------
        parameters: ParametersNK3E | None
            The parameters of the model. If None, default parameters will be used.
        variables: VariablesNK3E | None
            The variables of the model. If None, default variables will be used.
        scenarios: ScenariosNK3E | None
            The scenarios of the model. If None, default scenarios will be used.
        """
        if parameters is None:
            parameters = ParametersNK3E()
        if variables is None:
            variables = VariablesNK3E(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosNK3E(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorNK3E,
            *args,
            **kwargs,
        )
