"""
Shared fixtures for sample tests
"""

import pandas as pd
import pytest

from macrostat.core import Model, Parameters


class MockParameters(Parameters):
    """Mock parameters class for testing"""

    def get_default_parameters(self):
        """Return default parameters in the format expected by generate_tasks"""
        return {
            "param1": {
                "value": 1.0,
                "lower bound": 0.1,
                "upper bound": 1.0,
                "unit": "",
                "notation": "p1",
            },
            "param2": {
                "value": 2.0,
                "lower bound": 1.0,
                "upper bound": 10.0,
                "unit": "",
                "notation": "p2",
            },
        }


class MockModel(Model):
    """Mock model class for testing"""

    def __init__(
        self,
        parameters=None,
        scenarios=None,
        variables=None,
        behavior=None,
        *args,
        **kwargs
    ):
        if parameters is None:
            parameters = MockParameters()
        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            behavior=behavior,
        )

    def simulate(self, *args, **kwargs):
        # Return a simple DataFrame for testing
        return pd.DataFrame({"time": [1, 2, 3], "value": [1.0, 2.0, 3.0]}).set_index(
            "time"
        )


@pytest.fixture
def mock_model():
    """Fixture providing a mock model for testing"""
    return MockModel()


@pytest.fixture
def mock_parameters():
    return MockParameters()


@pytest.fixture
def valid_bounds():
    """Fixture providing valid parameter bounds for testing"""
    return {"param1": (0.1, 1.0), "param2": (1.0, 10.0)}


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Fixture providing a temporary output directory for testing"""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    return output_dir
