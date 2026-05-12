# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 Karl Naumann-Woleske
"""Variables class for the Kirman ants SDE model."""

import logging

from macrostat.core.variables import Variables
from macrostat.models.KirmansAnts.parameters import ParametersKirmansAnts

logger = logging.getLogger(__name__)


class VariablesKirmansAnts(Variables):
    """Variables class for the Kirman ants SDE model.

    The model has one scalar state variable, ``density``, the fraction of
    ants at source A. The micro-step trajectory is recorded out-of-band on
    the behavior instance (``BehaviorKirmansAnts._micro_trajectory``) when
    ``record_inner`` is True; ``record_state`` continues to record one
    end-of-macro-period value per outer step.
    """

    version = "KirmansAnts"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersKirmansAnts | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersKirmansAnts()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_variables(self):
        return {
            "density": {
                "notation": r"x_t",
                "unit": "fraction",
                "history": 0,
                "sectors": ["Ants"],
                "sfc": [("Index", "Ants")],
            },
        }
