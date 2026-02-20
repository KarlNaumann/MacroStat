"""
PichlerEtAl2022DIO -- Dynamic Input-Output Model from Pichler et al. (2022).

Reference: Pichler, Pangallo, Del Rio-Chanona, Lafond & Farmer (2022),
"Forecasting the propagation of pandemic shocks with a dynamic
input-output model", Journal of Economic Dynamics & Control.
"""

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import logging

from macrostat.core.model import Model

from .behavior import BehaviorPichlerEtAl2022DIO
from .parameters import ParametersPichlerEtAl2022DIO
from .scenarios import ScenariosPichlerEtAl2022DIO
from .variables import VariablesPichlerEtAl2022DIO

logger = logging.getLogger(__name__)


class PichlerEtAl2022DIO(Model):
    """Dynamic Input-Output model from Pichler et al. (2022).

    A daily-timestep disequilibrium model of pandemic supply-and-demand
    shocks propagating through an input-output network.

    Parameters
    ----------
    parameters : ParametersPichlerEtAl2022DIO or None
        Model parameters.  If ``None``, defaults are created.
    variables : VariablesPichlerEtAl2022DIO or None
        Variable definitions.  If ``None``, defaults are created from
        ``parameters``.
    scenarios : ScenariosPichlerEtAl2022DIO or None
        Scenario definitions.  If ``None``, defaults are created from
        ``parameters``.
    """

    version = "PichlerEtAl2022DIO"

    def __init__(
        self,
        parameters: ParametersPichlerEtAl2022DIO | None = None,
        variables: VariablesPichlerEtAl2022DIO | None = None,
        scenarios: ScenariosPichlerEtAl2022DIO | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersPichlerEtAl2022DIO()
        if variables is None:
            variables = VariablesPichlerEtAl2022DIO(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosPichlerEtAl2022DIO(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorPichlerEtAl2022DIO,
            *args,
            **kwargs,
        )
