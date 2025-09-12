"""
Variables class for the New Keynesian 3-Equation (NK3E) model.

Source: A New Keynesian 3-Equation Model — https://macrosimulation.org/a_new_keynesian_3_equation_model
"""

__author__ = ["Mitja Devetak"]
__credits__ = ["Mitja Devetak"]
__license__ = "MIT"
__maintainer__ = ["Mitja Devetak"]

import logging

from macrostat.core.variables import Variables
from macrostat.models.NK3E.parameters import ParametersNK3E

logger = logging.getLogger(__name__)


class VariablesNK3E(Variables):
    """Variables class for the NK3E model."""

    version = "NK3E"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersNK3E | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersNK3E()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_variables(self):
        return {
            "y": {
                "notation": r"y_t",
                "unit": ".",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "a3": {
                "notation": r"a_3",
                "unit": ".",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "pi": {
                "notation": r"\pi_t",
                "unit": "% per period",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "r": {
                "notation": r"r_t",
                "unit": "% per period",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
            "r_s": {
                "notation": r"r_s",
                "unit": "% per period",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
        }
