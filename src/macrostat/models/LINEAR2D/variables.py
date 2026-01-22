"""
Variables for the 2D linear test model.

We expose a single variable ``State`` that records the 2D state over time.
"""

from __future__ import annotations

import logging

from macrostat.core.variables import Variables
from macrostat.models.LINEAR2D.parameters import ParametersLINEAR2D

logger = logging.getLogger(__name__)


class VariablesLINEAR2D(Variables):
    """Variables for the 2D linear test model."""

    version = "LINEAR2D"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersLINEAR2D | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersLINEAR2D()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_variables(self):
        """Return a minimal variable info dictionary."""

        return {
            "State": {
                "notation": r"x(t)",
                "unit": ".",
                "history": 0,
                "sectors": ["Linear2D"],
                "sfc": [("Index", "Linear2D")],
            },
        }
