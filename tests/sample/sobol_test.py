"""
pytest code for the SobolSampler class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]


import pytest
import numpy as np
import pandas as pd
from scipy.stats import qmc

from macrostat.sample.sobol import SobolSampler
import macrostat.models.model as msmodel
import macrostat.util.batchprocessing as msbatchprocessing


# Mock the Model class and simulate method for testing purposes
class MockModel(msmodel.Model):
    def __init__(self, parameters=None, hyper_parameters={}):
        self.parameters = parameters if parameters is not None else {"param1": 0.5, "param2": 1.0}
        super().__init__(parameters=self.parameters, hyper_parameters={})

    def simulate(self, *args, **kwargs):
        return {"result": 42}  # Mock result for simulation


# Test the SobolSampler initialization
def test_sobol_sampler_initialization():
    model = MockModel()
    bounds = {"param1": (0.1, 1.0), "param2": (0.5, 2.0)}
    sampler = SobolSampler(model=model, bounds=bounds, sample_power=5)

    assert sampler.model == model
    assert sampler.bounds == bounds
    assert sampler.sample_power == 5
    assert sampler.seed == 0
    assert sampler.logspace is False
    assert sampler.cpu_count == 1
    assert sampler.batchsize is None


# Test Sobol point generation
def test_generate_sobol_points():
    model = MockModel()
    bounds = {"param1": (0.1, 1.0), "param2": (0.5, 2.0)}
    sampler = SobolSampler(model=model, bounds=bounds, sample_power=3)

    points = sampler._generate_sobol_points()

    assert isinstance(points, pd.DataFrame)
    assert len(points) == 2**3  # 2^sample_power points
    assert points.columns.tolist() == ["param1", "param2"]

    # Ensure points are within the bounds
    assert np.all(points["param1"] >= 0.1) and np.all(points["param1"] <= 1.0)
    assert np.all(points["param2"] >= 0.5) and np.all(points["param2"] <= 2.0)


# Test Sobol point generation in logspace
def test_generate_sobol_points_logspace():
    model = MockModel()
    bounds = {"param1": (0.1, 1.0), "param2": (0.5, 2.0)}
    sampler = SobolSampler(model=model, bounds=bounds, sample_power=3, logspace=True)

    points = sampler._generate_sobol_points()

    assert isinstance(points, pd.DataFrame)
    assert len(points) == 2**3  # 2^sample_power points
    assert points.columns.tolist() == ["param1", "param2"]

    # Ensure points are within the logspace bounds
    assert np.all(points["param1"] >= 0.1) and np.all(points["param1"] <= 1.0)
    assert np.all(points["param2"] >= 0.5) and np.all(points["param2"] <= 2.0)


# Test the _verify_bounds method
def test_verify_bounds_valid():
    model = MockModel()
    bounds = {"param1": (0.1, 1.0), "param2": (0.5, 2.0)}
    sampler = SobolSampler(model=model, bounds=bounds, sample_power=3)


def test_verify_bounds_invalid_parameter():
    model = MockModel()
    bounds = {"param1": (0.1, 1.0), "invalid_param": (0.5, 2.0)}

    with pytest.raises(ValueError, match = r"not in the model's parameters"):
        sampler = SobolSampler(model=model, bounds=bounds, sample_power=3)


def test_verify_bounds_invalid_length():
    model = MockModel()
    bounds = {"param1": (0.1,), "param2": (0.5, 2.0)}

    with pytest.raises(ValueError, match=r'Bounds should be a list-like of length 2'):
        sampler = SobolSampler(model=model, bounds=bounds, sample_power=3)


def test_verify_bounds_invalid_order():
    model = MockModel()
    bounds = {"param1": (1.0, 0.1), "param2": (0.5, 2.0)}

    with pytest.raises(ValueError, match="Lower bound should be smaller than the upper bound"):
        sampler = SobolSampler(model=model, bounds=bounds, sample_power=3)


def test_verify_bounds_invalid_logspace():
    model = MockModel()
    bounds = {"param1": (-1.0, 1.0), "param2": (0.5, 2.0)}
    
    with pytest.raises(ValueError, match=r'Bounds should be either both positive or both negative'):
        sampler = SobolSampler(model=model, bounds=bounds, sample_power=3, logspace=True)


def test_verify_bounds_zero_logspace():
    model = MockModel()
    bounds = {"param1": (0.0, 1.0), "param2": (0.5, 2.0)}
    
    with pytest.raises(ValueError, match=r'Bounds cannot be zero when using logspace'):
        sampler = SobolSampler(model=model, bounds=bounds, sample_power=3, logspace=True)

def test_generate_tasks():
    model = MockModel()
    bounds = {"param1": (0.1, 1.0), "param2": (0.5, 2.0)}
    sampler = SobolSampler(model=model, bounds=bounds, sample_power=3)

    tasks = sampler.generate_tasks()

    assert isinstance(tasks, list)
    assert len(tasks) == 2**3  # 2^sample_power tasks

    # Check that each task contains the correct elements
    for task in tasks:
        assert isinstance(task, tuple)
        assert isinstance(task[1], MockModel)
        assert task[0] in np.arange(2**3)


# Test saving and loading the sampler object
def test_save_and_load_sampler(tmp_path):
    model = MockModel()
    bounds = {"param1": (0.1, 1.0), "param2": (0.5, 2.0)}
    sampler = SobolSampler(model=model, bounds=bounds, sample_power=3)
    sampler.output_folder = tmp_path

    # Save the sampler object
    save_path = "sobol_sampler"
    sampler.save(name=str(save_path))

    # Load the sampler object
    loaded_sampler = SobolSampler.load(str(tmp_path / save_path)+".pkl")

    assert isinstance(loaded_sampler, SobolSampler)
    assert loaded_sampler.bounds == sampler.bounds
    assert loaded_sampler.sample_power == sampler.sample_power
    assert loaded_sampler.seed == sampler.seed
