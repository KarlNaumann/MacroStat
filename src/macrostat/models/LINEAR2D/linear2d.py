"""
Model wrapper for the 2D linear test model.
"""

from __future__ import annotations

import logging

from macrostat.core.model import Model
from macrostat.models.LINEAR2D.behavior import BehaviorLINEAR2D
from macrostat.models.LINEAR2D.parameters import ParametersLINEAR2D
from macrostat.models.LINEAR2D.scenarios import ScenariosLINEAR2D
from macrostat.models.LINEAR2D.variables import VariablesLINEAR2D

logger = logging.getLogger(__name__)


class LINEAR2D(Model):
    """Simple 2D linear model for Jacobian testing."""

    version = "LINEAR2D"

    def __init__(
        self,
        parameters: ParametersLINEAR2D | None = ParametersLINEAR2D(),
        variables: VariablesLINEAR2D | None = None,
        scenarios: ScenariosLINEAR2D | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersLINEAR2D()
        if variables is None:
            variables = VariablesLINEAR2D(parameters=parameters)
        if scenarios is None:
            scenarios = ScenariosLINEAR2D(parameters=parameters)

        super().__init__(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            behavior=BehaviorLINEAR2D,
            *args,
            **kwargs,
        )
