"""
pytest code for the Parameters class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import copy
import logging
import os

import pytest

from macrostat.core import BoundaryError, Parameters


class TestParameters:
    """Tests for the Parameters class found in models/parameters.py"""

    # Sample test parameters
    params = {"param1": 1.0, "param2": 2.0}

    hyper = {
        "timesteps": 100,
        "timesteps_initialization": 10,
        "scenario_trigger": 0,
        "seed": 42,
        "device": "cpu",
        "requires_grad": False,
    }

    bounds = {"param1": (0.0, 2.0), "param2": (1.0, 3.0)}

    def test_init(self):
        """Test initialization with different parameter combinations"""
        # Test with all parameters provided
        p = Parameters(
            parameters=self.params, hyperparameters=self.hyper, bounds=self.bounds
        )
        assert p.values == self.params
        assert p.hyper == self.hyper
        assert p.bounds == self.bounds

        # Test with no parameters provided
        p = Parameters()
        assert isinstance(p.values, dict)
        assert isinstance(p.hyper, dict)
        assert isinstance(p.bounds, dict)

    def test_contains(self):
        """Test the contains magic method"""
        p = Parameters(
            parameters=self.params, hyperparameters=self.hyper, bounds=self.bounds
        )
        assert "param1" in p
        assert "timesteps" in p
        assert "nonexistent" not in p

    def test_getitem(self):
        """Test the getitem magic method"""
        p = Parameters(
            parameters=self.params, hyperparameters=self.hyper, bounds=self.bounds
        )
        assert p["param1"] == 1.0
        assert p["timesteps"] == 100

    def test_setitem(self, caplog):
        """Test the setitem magic method"""
        p = Parameters(
            parameters=self.params, hyperparameters=self.hyper, bounds=self.bounds
        )
        # Test setting parameter value
        p["param1"] = 1.5
        assert p.values["param1"] == 1.5

        # Test setting hyperparameter value as int
        p["timesteps"] = 200
        assert p.hyper["timesteps"] == 200

        # Test setting hyperparameter value that can't be converted to int
        p["device"] = "cuda"
        assert p.hyper["device"] == "cuda"

        # Test setting non-existent parameter
        with caplog.at_level(logging.WARNING):
            p["nonexistent"] = 1.0
        assert (
            "Key nonexistent not found in parameters or hyperparameters." in caplog.text
        )

    def test_boundary_validation(self):
        """Test boundary validation"""
        # Test invalid bounds
        invalid_bounds = copy.deepcopy(self.bounds)
        invalid_bounds["param1"] = (2.0, 1.0)  # Upper < Lower
        with pytest.raises(BoundaryError):
            Parameters(
                parameters=self.params,
                hyperparameters=self.hyper,
                bounds=invalid_bounds,
            )

        # Test parameter outside bounds
        invalid_params = copy.deepcopy(self.params)
        invalid_params["param1"] = 3.0  # Outside (0.0, 2.0)
        with pytest.raises(BoundaryError):
            Parameters(
                parameters=invalid_params,
                hyperparameters=self.hyper,
                bounds=self.bounds,
            )

        # Test missing bounds for parameters
        with pytest.raises(BoundaryError) as exc_info:
            p = Parameters(
                parameters=self.params,
                hyperparameters=self.hyper,
                bounds=self.bounds,
            )
            p.get_default_bounds = lambda: self.bounds
            del p.bounds["param1"]
            p.verify_bounds()
        assert "Missing bounds for parameters" in str(exc_info.value)

    def test_json_io(self, tmpdir):
        """Test JSON file I/O"""
        p = Parameters(
            parameters=self.params, hyperparameters=self.hyper, bounds=self.bounds
        )

        # Save to JSON
        json_path = os.path.join(tmpdir, "test_params.json")
        p.to_json(json_path)

        # Load from JSON
        p2 = Parameters.from_json(json_path)

        assert p.values == p2.values
        assert p.hyper == p2.hyper
        assert p.bounds == p2.bounds

    def test_set_bounds(self):
        """Test setting bounds"""
        p = Parameters(
            parameters=self.params, hyperparameters=self.hyper, bounds=self.bounds
        )

        new_bounds = {"param1": (0.5, 1.5)}
        p.set_bounds(new_bounds)
        assert p.bounds["param1"] == (0.5, 1.5)

        p.set_bound("param2", (1.5, 2.5))
        assert p.bounds["param2"] == (1.5, 2.5)
