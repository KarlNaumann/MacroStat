"""Scenarios class for the New Keynesian 3-Equation (NK3E) model."""

__author__ = ["Mitja Devetak"]
__credits__ = ["Mitja Devetak"]
__license__ = "MIT"
__maintainer__ = ["Mitja Devetak"]

import logging

import torch

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
        """Return the default scenario values.

        Two families coexist:

        - Parameter-step shocks (``A_add``, ``pi_T_add``, ``y_e_add``): these
          shift structural parameters from ``scenario_trigger`` onward, and
          flow through ``Behavior.apply_parameter_shocks``.
        - State-variable shocks (``InflationShock``, ``RateShock``,
          ``OutputShock``): these are additive perturbations of the
          corresponding state variables ``pi``, ``r``, ``y`` inside
          ``BehaviorNK3E.step``, and default to zero everywhere. Register
          them as one-period impulses by passing a length-1 list, e.g.
          ``{"InflationShock": [1.0]}``.
        """
        return {
            "A_add": 0.0,
            "pi_T_add": 0.0,
            "y_e_add": 0.0,
            "InflationShock": 0.0,
            "RateShock": 0.0,
            "OutputShock": 0.0,
        }

    def add_three_parameterizations(self):
        """Register the three default NK3E scenarios plus one state-shock scenario."""
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

        # Scenario 4: one-period inflation impulse (+1 on pi).
        # The impulse fires at t = max(trigger, init_t) — the first step that
        # is both within the scenario window and in the sim loop. The offset
        # into the timeseries vector is fire_offset = max(0, init_t - trigger),
        # which is 0 when trigger >= init_t (fires right at the trigger) and
        # positive when trigger < init_t (fires at the first sim step instead).
        # This is safe for all trigger/init_t combinations, including the common
        # notebook case trigger=25, init_t=1 where the earlier formula crashed
        # with a negative tensor index.
        trigger = self.parameters["scenario_trigger"]
        init_t = self.parameters.hyper.get("timesteps_initialization", 0)
        fire_offset = max(0, init_t - trigger)
        impulse = torch.zeros(fire_offset + 1)
        impulse[fire_offset] = 1.0
        self.add_scenario(
            timeseries={
                "InflationShock": impulse,
            },
            name="Scenario.4: Inflation impulse",
        )
