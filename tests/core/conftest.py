"""
Mock classes for the different tests
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

from macrostat.core import Model, Parameters, Scenarios, Variables


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
