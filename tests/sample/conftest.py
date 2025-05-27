"""
Shared fixtures for sample tests
"""

import pandas as pd
import pytest

from macrostat.core import Model, Parameters


class MockModel(Model):
    """Mock model class for testing"""

    def __init__(self, parameters=None):
        super().__init__(
            parameters=Parameters(
                {
                    "param1": {"value": 1.0, "lower bound": 0.1, "upper bound": 1.0},
                    "param2": {"value": 2.0, "lower bound": 1.0, "upper bound": 10.0},
                }
            )
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
    return Parameters(
        {
            "param1": {"value": 1.0, "lower bound": 0.1, "upper bound": 1.0},
            "param2": {"value": 2.0, "lower bound": 1.0, "upper bound": 10.0},
        }
    )


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
