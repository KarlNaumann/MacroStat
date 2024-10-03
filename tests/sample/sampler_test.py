"""
pytest code for the Sampler class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

from pathlib import Path
import pytest
from unittest.mock import MagicMock, patch, call

import numpy as np
import pandas as pd
import os
import pickle

from macrostat.sample.sampler import Sampler  # Replace with your actual module path
import macrostat.util.batchprocessing as batchprocessing

# Mock model class for testing
class MockModel:
    def __init__(self, parameters=None):
        self.parameters = parameters if parameters else {}

    def simulate(self, *args):
        return pd.DataFrame({"A": [1, 2], "B": [3, 4]})


# Test the initialization of the Sampler class
def test_sampler_initialization():
    model = MockModel(parameters={"param1": 1, "param2": 2})
    
    # Initialize the sampler
    sampler = Sampler(model=model, output_folder="test_folder", cpu_count=2)
    
    # Assert that the model and base parameters are correctly set
    assert sampler.model == model
    assert sampler.modelclass == type(model)
    assert sampler.base_parameters == model.parameters
    assert sampler.cpu_count == 2
    assert sampler.output_folder == Path("test_folder")
    assert os.path.exists("test_folder")  # Check that the output folder is created


# Test if NotImplementedError is raised for generate_tasks
def test_sampler_generate_tasks():
    model = MockModel(parameters={"param1": 1})
    sampler = Sampler(model=model, output_folder="test_folder", cpu_count=2)
    
    # Ensure NotImplementedError is raised for unimplemented method
    with pytest.raises(NotImplementedError, match="This method should be implemented in a subclass"):
        sampler.generate_tasks()


# Test the sample method (mocking underlying functions)
@patch('macrostat.util.batchprocessing.parallel_processor')
@patch('pandas.DataFrame.to_csv')
@patch('os.makedirs')
def test_sampler_sample(mock_makedirs, mock_to_csv, mock_parallel_processor):
    model = MockModel(parameters={"param1": 1, "param2": 2})
    sampler = Sampler(model=model, output_folder="test_folder", cpu_count=2, batchsize=2)

    # Mock tasks and outputs for sampling
    sampler.generate_tasks = MagicMock()
    sampler.generate_tasks.return_value = [("task_1", model), ("task_2", model)]
    
    # Mock the parallel processing results
    mock_parallel_processor.return_value = [(model, pd.DataFrame({"A": [1, 2], "B": [3, 4]}))] * 2
    
    # Call the sample function
    sampler.sample()

    # Assert that tasks were generated and processed
    sampler.generate_tasks.assert_called_once()
    mock_parallel_processor.assert_called_with(
        tasks=sampler.tasks[:2],
        worker=sampler.worker_function,
        cpu_count=sampler.cpu_count,
        tqdm_info="Sampling"
    )

# Test save_outputs method
@patch('pandas.DataFrame.to_csv')
def test_sampler_save_outputs(mock_to_csv):
    model = MockModel(parameters={"param1": 1, "param2": 2})
    sampler = Sampler(model=model, output_folder="test_folder", batchsize=2)
    
    # Mock raw_outputs with pandas DataFrames
    raw_outputs = [(model, pd.DataFrame({"A": [1, 2], "B": [3, 4]}))] * 2

    # Call the save_outputs method
    sampler.save_outputs(raw_outputs, batch=0)

    # Assert DataFrame was saved correctly
    mock_to_csv.assert_called_once()


# Test the extract method
@patch('pandas.read_csv')
def test_sampler_extract(mock_read_csv):
    model = MockModel(parameters={"param1": 1, "param2": 2})
    sampler = Sampler(model=model, output_folder="test_folder", batchsize=2)

    # Mock the data returned by the pandas reader
    x = pd.DataFrame({"col1": [1, 2], "col2": [3, 4]})
    mock_read_csv.return_value = iter([x, x, x])

    # Call extract method with mock columns
    output = sampler.extract(columns=["col1"], indices=[0, 1])

    # Assert that data was correctly returned
    assert isinstance(output, pd.DataFrame)
    assert "col1" in output.columns

@patch('pandas.read_csv')
def test_sampler_extract_multiindex(mock_read_csv):
    model = MockModel(parameters={"param1": 1, "param2": 2})
    sampler = Sampler(model=model, output_folder="test_folder", batchsize=2)

    y = pd.DataFrame(np.arange(9).reshape(3, 3), columns=["A", "B", "C"], index=pd.MultiIndex.from_tuples([(0, 0), (0, 1), (1, 0)], names=["i", "j"]))
    mock_read_csv.return_value = iter([y, y, y])

    # Call extract method with mock columns
    output = sampler.extract(columns=["A"], indices=[(0,0), (1,0)])

    # Assert that data was correctly returned
    assert isinstance(output, pd.DataFrame)
    assert "A" in output.columns
    assert output.index.names == ["i", "j"]
    assert output.index[0] == (0, 0)
    assert output.index[1] == (1, 0)
    


# Test save method (mocking file operations)
@patch('builtins.open', new_callable=MagicMock)
@patch('pickle.dump')
def test_sampler_save(mock_pickle_dump, mock_open):
    model = MockModel(parameters={"param1": 1, "param2": 2})
    sampler = Sampler(model=model, output_folder="test_folder")

    # Call the save method
    sampler.save(name="test_sampler")

    # Assert that the file was opened and the object was pickled
    mock_open.assert_called_once_with(f"test_folder{os.sep}test_sampler.pkl", "wb")
    mock_pickle_dump.assert_called_once_with(sampler, mock_open.return_value.__enter__.return_value)


# Test load method (mocking file operations)
@patch('builtins.open', new_callable=MagicMock)
@patch('pickle.load')
def test_sampler_load(mock_pickle_load, mock_open):
    # Mock the object returned by pickle.load
    mock_pickle_load.return_value = "mock_sampler_object"

    # Call the load method
    sampler = Sampler.load("test_folder/test_sampler.pkl")

    # Assert that the file was opened and the object was loaded
    mock_open.assert_called_once_with("test_folder/test_sampler.pkl", "rb")
    mock_pickle_load.assert_called_once()
    assert sampler == "mock_sampler_object"