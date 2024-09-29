"""
pytest code for the batchprocessing.py class
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

from macrostat.utilities.batchprocessing import BatchProcessing, timeseries_worker

# Mock Model class for testing
class MockModel:
    def __init__(self, output):
        self.output = output

    def simulate(self, *args):
        # Simulate some computation
        return self.output

# Mock BatchProcessing class with tasks and cpu_count attributes
class MockBatchProcessing(BatchProcessing):
    def __init__(self, tasks, cpu_count=2):
        self.tasks = tasks
        self.cpu_count = cpu_count

# Test for timeseries_worker function
def test_timeseries_worker():
    model = MockModel(output={"result": 42})
    task = (model, "simulation_1", "scenario_1")

    # Execute the worker
    simulation_id, scenario_id, output = timeseries_worker(task)

    # Assert that the worker function correctly returns the simulation result
    assert simulation_id == "simulation_1"
    assert scenario_id == "scenario_1"
    assert output == {"result": 42}

# Test for parallel_processor using pytest
def test_parallel_processor():
    # Mock models with different outputs
    mock_model_1 = MockModel(output={"result": 42})
    mock_model_2 = MockModel(output={"result": 24})

    tasks = [
        (mock_model_1, "simulation_1", "scenario_1"),
        (mock_model_2, "simulation_2", "scenario_2"),
    ]

    # Create instance of the mocked BatchProcessing class
    batch_processor = MockBatchProcessing(tasks=tasks, cpu_count=2)

    # Patch the ProcessPoolExecutor to simulate the parallel processing
    with patch('concurrent.futures.ProcessPoolExecutor') as mock_executor:
        # Mock the map function to simulate parallel execution
        mock_executor.return_value.__enter__.return_value.map = MagicMock(
            return_value=[
                ("simulation_1", "scenario_1", {"result": 42}),
                ("simulation_2", "scenario_2", {"result": 24}),
            ]
        )

        # Call parallel_processor and capture the result
        result = batch_processor.parallel_processor(tqdm_info="Processing")

        # Assert that the results are as expected
        assert len(result) == 2
        assert result[0] == ("simulation_1", "scenario_1", {"result": 42})
        assert result[1] == ("simulation_2", "scenario_2", {"result": 24})

# Test if the parallel processing executes all tasks
def test_parallel_processor_full_execution():
    # Mock models with output
    mock_model_1 = MockModel(output={"result": 42})
    mock_model_2 = MockModel(output={"result": 24})

    tasks = [
        (mock_model_1, "simulation_1", "scenario_1"),
        (mock_model_2, "simulation_2", "scenario_2"),
    ]

    # Create instance of the mocked BatchProcessing class
    batch_processor = MockBatchProcessing(tasks=tasks, cpu_count=2)

    # Capture output using the real ProcessPoolExecutor
    result = batch_processor.parallel_processor(tqdm_info="Processing")

    # Assert that the results are as expected
    assert len(result) == 2
    assert result[0] == ("simulation_1", "scenario_1", {"result": 42})
    assert result[1] == ("simulation_2", "scenario_2", {"result": 24})
