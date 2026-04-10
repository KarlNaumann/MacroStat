"""
Mock classes for the different tests
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

from macrostat.core import LinearConstraint, Model, Parameters, Scenarios, Variables


class MockScenarios(Scenarios):
    """Test class for the Scenarios class"""

    def get_default_scenario_values(self) -> dict:
        """Get the default values for the scenarios"""
        return {k: 0.0 for k in ["shock1", "shock2", "shock3"]}


class MockVariables(Variables):
    """Test class for the Variables class"""

    def get_default_variables(self) -> dict:
        """Get the default values for the variables"""
        return {
            f"variable{k}": {
                "notation": r"v",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            }
            for k in [1, 2, 3]
        }


class MockParameters(Parameters):
    """Test class for the Parameters class"""

    def get_default_parameters(self):
        return {
            "param1": {
                "value": 1.0,
                "lower bound": 0.0,
                "upper bound": 2.0,
                "unit": "units",
                "notation": "p_1",
            },
            "param2": {
                "value": 2.0,
                "lower bound": 0.0,
                "upper bound": 3.0,
                "unit": "units",
                "notation": "p_2",
            },
        }

    def get_default_hyperparameters(self):
        return {
            "timesteps": 500,
            "timesteps_initialization": 10,
            "scenario_trigger": 0,
            "seed": 42,
            "device": "cpu",
            "requires_grad": False,
        }


class MockModel(Model):
    parameters = MockParameters()
    variables = MockVariables(parameters=parameters)
    scenarios = MockScenarios(parameters=parameters)


class VectorMockParameters(Parameters):
    """Mock parameters with a 1-D sector-indexed constraint.

    Two sectors (``Household`` and ``Firm``), one sector-indexed
    parameter ``Share``, constrained to sum to 1. Exercises the
    1-D resolver path through init and step time uniformly.
    """

    def get_default_parameters(self):
        return {
            "Household.Share": {
                "value": 0.6,
                "lower bound": 0.0,
                "upper bound": 1.0,
                "unit": ".",
                "notation": r"s_H",
            },
            "Firm.Share": {
                "value": 0.4,
                "lower bound": 0.0,
                "upper bound": 1.0,
                "unit": ".",
                "notation": r"s_F",
            },
        }

    def get_default_hyperparameters(self):
        return {
            "timesteps": 100,
            "timesteps_initialization": 10,
            "scenario_trigger": 0,
            "seed": 42,
            "device": "cpu",
            "requires_grad": False,
            "vector_sectors": ["Household", "Firm"],
        }

    def get_constraints(self):
        return (
            LinearConstraint(
                param_names=("Household.Share", "Firm.Share"),
                target=1.0,
            ),
        )


class VectorMockVariables(Variables):
    """Minimal variables for a two-sector vector-capable mock model."""

    def get_default_variables(self) -> dict:
        return {
            "output": {
                "notation": r"Y",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household", "Firm"],
                "sfc": [("Index", "Household"), ("Index", "Firm")],
            }
        }


class VectorMockScenarios(Scenarios):
    """Scenarios for the two-sector vector-capable mock model."""

    def get_default_scenario_values(self) -> dict:
        return {"shock": 0.0}


class VectorMockModel(Model):
    """Two-sector mock model with a 1-D adding-up constraint on ``Share``."""

    parameters = VectorMockParameters()
    variables = VectorMockVariables(parameters=parameters)
    scenarios = VectorMockScenarios(parameters=parameters)
