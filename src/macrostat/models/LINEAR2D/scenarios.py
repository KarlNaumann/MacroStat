"""
Scenarios for the 2D linear test model.

This model does not use any exogenous shocks; we keep the scenario structure
minimal to satisfy the MacroStat interfaces.
"""

from __future__ import annotations

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.LINEAR2D.parameters import ParametersLINEAR2D

logger = logging.getLogger(__name__)


class ScenariosLINEAR2D(Scenarios):
    """Scenarios for the 2D linear test model."""

    version = "LINEAR2D"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersLINEAR2D | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersLINEAR2D()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_scenario_values(self):
        """Return an empty scenario dictionary (no exogenous shocks)."""

        return {}
