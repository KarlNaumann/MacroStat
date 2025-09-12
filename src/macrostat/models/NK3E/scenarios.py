"""Scenarios class for the New Keynesian 3-Equation (NK3E) model."""

__author__ = ["Mitja Devetak"]
__credits__ = ["Mitja Devetak"]
__license__ = "MIT"
__maintainer__ = ["Mitja Devetak"]

import logging

from macrostat.core.scenarios import Scenarios
from macrostat.models.NK3E.parameters import ParametersNK3E

logger = logging.getLogger(__name__)


class ScenariosNK3E(Scenarios):
    """Scenarios class for the New Keynesian 3-Equation (NK3E) model."""

    version = "NK3E"

    def __init__(
        self,
        scenario_info: dict | None = None,
        parameters: ParametersNK3E | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the scenarios of the New Keynesian 3-Equation (NK3E) model."""
        if parameters is None:
            parameters = ParametersNK3E()

        super().__init__(
            scenario_info=scenario_info,
            parameters=parameters,
            *args,
            **kwargs,
        )

        # Add the three named scenarios matching the table
        self.add_three_parameterizations()

    def get_default_scenario_values(self):
        """Return the default scenario values."""
        # Baseline (no shock): A=10, pi_T=2, y_e=5
        return {
            "A_add": 0.0,
            "pi_T_add": 0.0,
            "y_e_add": 0.0,
        }

    def add_three_parameterizations(self):
        """Register the three default NK3E scenarios."""
        # Scenario 1: rise in A (A: 12 vs baseline 10 -> +2)
        self.add_scenario(
            timeseries={
                "A_add": 2.0,
            },
            name="Scenario.1: Rise in A",
        )

        # Scenario 2: higher inflation target (pi_T: 3 vs 2 -> +1)
        self.add_scenario(
            timeseries={
                "pi_T_add": 1.0,
            },
            name="Scenario.2: Higher pi_T",
        )

        # Scenario 3: rise in equilibrium output (y_e: 7 vs 5 -> +2)
        self.add_scenario(
            timeseries={
                "y_e_add": 2.0,
            },
            name="Scenario.3: Rise in y_e",
        )
