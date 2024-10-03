"""
pytest code for the batchprocessing module
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import pytest
from unittest.mock import MagicMock, patch
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import time

from macrostat.util.batchprocessing import parallel_processor, timeseries_worker

# Mock Model class for testing
class MockModel:
    def __init__(self, output):
        self.output = output

    def simulate(self, *args):
        # Simulate some computation
        return self.output


# Test for timeseries_worker function
def test_timeseries_worker():
    model = MockModel(output={"result": 42})
    task = ("simulation_1", model, "scenario_1")

    # Execute the worker
    simulation_id, scenario_id, output = timeseries_worker(task)

    # Assert that the worker function correctly returns the simulation result
    assert simulation_id == "simulation_1"
    assert scenario_id == "scenario_1"
    assert output == {"result": 42}


# Test for parallel_processor when no tasks are provided
def test_parallel_processor_no_tasks():
    with pytest.raises(ValueError, match="No tasks to process."):
        parallel_processor(tasks=[], cpu_count=2, tqdm_info="Processing")


# Test for parallel_processor with mocked ProcessPoolExecutor
def test_parallel_processor_with_tasks():
    # Mock models with different outputs
    mock_model_1 = MockModel(output={"result": 42})
    mock_model_2 = MockModel(output={"result": 24})

    tasks = [
        ("simulation_1", mock_model_1, "scenario_1"),
        ("simulation_2", mock_model_2, "scenario_2"),
    ]

    # Mock the ProcessPoolExecutor to simulate the parallel processing
    with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
        # Mock the map function to simulate parallel execution
        mock_executor.return_value.__enter__.return_value.map = MagicMock(
            return_value=[
                ("simulation_1", "scenario_1", {"result": 42}),
                ("simulation_2", "scenario_2", {"result": 24}),
            ]
        )

        # Call parallel_processor and capture the result
        result = parallel_processor(tasks=tasks, cpu_count=2, tqdm_info="Processing")

        # Assert that the results are as expected
        assert len(result) == 2
        assert result[0] == ("simulation_1", "scenario_1", {"result": 42})
        assert result[1] == ("simulation_2", "scenario_2", {"result": 24})


# Test parallel_processor with multiple CPU utilization
def test_parallel_processor_cpu_count():
    # Mock models
    mock_model = MockModel(output={"result": 42})

    tasks = [
        ("simulation_1", mock_model, "scenario_1"),
        ("simulation_2", mock_model, "scenario_2"),
        ("simulation_3", mock_model, "scenario_3"),
    ]

    # Patch the ProcessPoolExecutor to simulate parallel processing
    with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
        mock_executor.return_value.__enter__.return_value.map = MagicMock(
            return_value=[
                ("simulation_1", "scenario_1", {"result": 42}),
                ("simulation_2", "scenario_2", {"result": 42}),
                ("simulation_3", "scenario_3", {"result": 42}),
            ]
        )

        # Call parallel_processor and capture the result
        result = parallel_processor(tasks=tasks, cpu_count=3, tqdm_info="Processing")

        # Assert that the results are as expected
        assert len(result) == 3
        assert result[0] == ("simulation_1", "scenario_1", {"result": 42})
        assert result[1] == ("simulation_2", "scenario_2", {"result": 42})
        assert result[2] == ("simulation_3", "scenario_3", {"result": 42})
