"""
pytest code for the batchprocessing module
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

from unittest.mock import MagicMock, patch

import pytest

from macrostat.core import Model, Variables
from macrostat.util.batchprocessing import parallel_processor, timeseries_worker


class MockVariables(Variables):
    def get_default_variables(self):
        return {
            "Variable1": {
                "notation": r"v_1(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Sector1"],
                "sfc": [("Index", "Sector1")],
            },
            "Variable2": {
                "notation": r"v_2(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Sector2"],
                "sfc": [("Index", "Sector2")],
            },
        }


# Mock Model class for testing
class MockModel(Model):
    def __init__(self):
        self.variables = MockVariables()

    def simulate(self, *args):
        # Simulate some computation
        return self.variables.gather_timeseries()


# Test for timeseries_worker function
def test_timeseries_worker():
    model = MockModel()
    task = ("simulation_1", model, "scenario_1")

    # Execute the worker
    simulation_id, scenario_id, output = timeseries_worker(task)

    # Assert that the worker function correctly returns the simulation result
    assert simulation_id == "simulation_1"
    assert scenario_id == "scenario_1"
    assert output.equals(MockModel().variables.to_pandas())


# Test for parallel_processor when no tasks are provided
def test_parallel_processor_no_tasks():
    with pytest.raises(ValueError, match="No tasks to process."):
        parallel_processor(tasks=[], cpu_count=2)


# Test for parallel_processor with mocked ProcessPoolExecutor
def test_parallel_processor_with_tasks():
    # Mock models with different outputs
    mock_model_1 = MockModel()
    mock_model_2 = MockModel()

    tasks = [
        ("simulation_1", mock_model_1, "scenario_1"),
        ("simulation_2", mock_model_2, "scenario_2"),
    ]

    # Mock the ProcessPoolExecutor to simulate the parallel processing
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
        # Mock the map function to simulate parallel execution
        mock_executor.return_value.__enter__.return_value.map = MagicMock(
            return_value=[
                ("simulation_1", "scenario_1", {"result": 42}),
                ("simulation_2", "scenario_2", {"result": 24}),
            ]
        )

        # Call parallel_processor and capture the result
        result = parallel_processor(tasks=tasks, cpu_count=2)

        # Assert that the results are as expected
        assert len(result) == 2

        outtgt = MockModel().variables.to_pandas()
        assert result[0][0] == "simulation_1"
        assert result[0][1] == "scenario_1"
        assert result[0][2].equals(outtgt)
        assert result[1][0] == "simulation_2"
        assert result[1][1] == "scenario_2"
        assert result[1][2].equals(outtgt)


# Test parallel_processor with multiple CPU utilization
def test_parallel_processor_cpu_count():
    # Mock models
    mock_model = MockModel()

    tasks = [
        ("simulation_1", mock_model, "scenario_1"),
        ("simulation_2", mock_model, "scenario_2"),
        ("simulation_3", mock_model, "scenario_3"),
    ]

    # Patch the ProcessPoolExecutor to simulate parallel processing
    with patch("concurrent.futures.ProcessPoolExecutor") as mock_executor:
        mock_executor.return_value.__enter__.return_value.map = MagicMock(
            return_value=[
                ("simulation_1", "scenario_1", {"result": 42}),
                ("simulation_2", "scenario_2", {"result": 42}),
                ("simulation_3", "scenario_3", {"result": 42}),
            ]
        )

        # Call parallel_processor and capture the result
        result = parallel_processor(tasks=tasks, cpu_count=3)

        # Assert that the results are as expected (output equivalence via _with_tasks test)
        assert len(result) == 3
