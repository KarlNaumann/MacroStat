"""
Parameters class for the New Keynesian 3-Equation (NK3E) model.

Source: A New Keynesian 3-Equation Model — https://macrosimulation.org/a_new_keynesian_3_equation_model
"""

__author__ = ["Mitja Devetak"]
__credits__ = ["Mitja Devetak"]
__license__ = "MIT"
__maintainer__ = ["Mitja Devetak"]

import logging

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class ParametersNK3E(Parameters):
    """Parameters for the NK3E model.

    Economic meaning:
    - a1 (>0): sensitivity of demand to the real rate (steeper => interest rate
      moves output more).
    - a2 (>0): sensitivity of inflation to the output gap.
    - b  (>0): central bank weight on inflation deviations in its loss.
    - A: autonomous demand (times multiplier).
    - pi_T: inflation target.
    - y_e: equilibrium (potential) output.

    Hyperparameters control the simulation environment, not the economics, e.g.,
    number of timesteps and progress bar usage.
    """

    version = "NK3E"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            parameters=parameters,
            hyperparameters=hyperparameters,
            *args,
            **kwargs,
        )

    def get_default_parameters(self):
        """Return the default parameter values for NK3E.

        Notes
        -----
        - a3 is not a primitive; it is derived each step as
          ``1 / (a1 * (1/(a2*b) + a2))``.
        - The three scenarios modify A, pi_T, or y_e around these baselines.
        """
        return {
            "a1": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"a_1",
                "unit": ".",
                "value": 0.3,
            },
            "a2": {
                "lower bound": 0.0,
                "upper bound": 10.0,
                "notation": r"a_2",
                "unit": ".",
                "value": 0.7,
            },
            "b": {
                "lower bound": 0.0,
                "upper bound": 100.0,
                "notation": r"b",
                "unit": ".",
                "value": 1.0,
            },
            "A": {
                "lower bound": -1000.0,
                "upper bound": 1000.0,
                "notation": r"A",
                "unit": ".",
                "value": 10.0,
            },
            "pi_T": {
                "lower bound": -100.0,
                "upper bound": 100.0,
                "notation": r"\pi^T",
                "unit": "% per period",
                "value": 2.0,
            },
            "y_e": {
                "lower bound": -1000.0,
                "upper bound": 1000.0,
                "notation": r"y_e",
                "unit": ".",
                "value": 5.0,
            },
        }

    def get_default_hyperparameters(self):
        hyper = super().get_default_hyperparameters()
        hyper["timesteps"] = 50
        hyper["timesteps_initialization"] = 1
        hyper["sectors"] = [
            "Macroeconomy",
        ]
        # tqdm usage toggle (default off)
        hyper["use_tqdm"] = False
        return hyper
