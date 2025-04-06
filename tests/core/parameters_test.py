"""
pytest code for the Macrostat Core Parameters class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import copy
import logging

import pytest
import torch

from macrostat.core import BoundaryError, Parameters


class TestParameters:
    """Tests for the Parameters class found in core/parameters.py

    This class does NOT test the following methods:
    - Parameters.__str__

    """

    # Sample test parameters
    params = {
        "param1": {
            "value": 1.0,
            "lower bound": 0.0,
            "upper bound": 2.0,
            "unit": "units",
            "notation": "p_1",
        },
        "param2": {
            "value": 2.0,
            "lower bound": 1.0,
            "upper bound": 3.0,
            "unit": "units",
            "notation": "p_2",
        },
    }

    hyper = {
        "timesteps": 100,
        "timesteps_initialization": 10,
        "scenario_trigger": 0,
        "seed": 42,
        "device": "cpu",
        "requires_grad": False,
    }

    def test_boundary_exception_class(self):
        """Test that the BoundaryError class is defined and returns the correct message"""
        assert isinstance(BoundaryError, type)
        assert issubclass(BoundaryError, Exception)
        assert (
            BoundaryError("test").message
            == "test Please check the Excel, JSON or default bounds."
        )

    def test_init_with_params(self):
        """Test initialization with parameters and hyperparameters provided"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        assert p.values == self.params
        assert p.hyper == self.hyper

    def test_init_empty(self):
        """Test initialization with no parameters provided"""
        p = Parameters()
        assert isinstance(p.values, dict)
        assert isinstance(p.hyper, dict)

    def test_contains(self):
        """Test the contains magic method"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        assert "param1" in p
        assert "timesteps" in p
        assert "nonexistent" not in p

    def test_getitem(self):
        """Test the getitem magic method"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        assert p["param1"] == 1.0
        assert p["timesteps"] == 100
        with pytest.raises(KeyError):
            p["nonexistent"]

    def test_setitem_parameter_value(self):
        """Test setting a parameter value"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        p["param1"] = 1.5
        assert p.values["param1"]["value"] == 1.5

    def test_setitem_hyperparameter_int(self):
        """Test setting a hyperparameter value that should be an int"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        p["timesteps"] = 200
        assert p.hyper["timesteps"] == 200

    def test_setitem_hyperparameter_string(self):
        """Test setting a hyperparameter value that is a string"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        p["device"] = "cuda"
        assert p.hyper["device"] == "cuda"

    def test_setitem_nonexistent(self, caplog):
        """Test setting a non-existent parameter"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        with caplog.at_level(logging.WARNING):
            p["nonexistent"] = 1.0
        assert (
            "Key nonexistent not found in parameters or hyperparameters." in caplog.text
        )

    def test_json_to_file(self, tmp_path):
        """Test saving parameters to JSON file"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)

        json_file = tmp_path / "params.json"
        p.to_json(json_file)
        assert json_file.exists()

    def test_json_from_file(self, tmp_path):
        """Test loading parameters from JSON file"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        json_file = tmp_path / "params.json"
        p.to_json(json_file)

        loaded_params = Parameters.from_json(json_file)
        assert loaded_params.values == p.values
        assert loaded_params.hyper == p.hyper

    def test_json_roundtrip(self, tmp_path):
        """Test JSON serialization roundtrip"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        roundtrip_file = tmp_path / "roundtrip.json"
        p.to_json(roundtrip_file)
        loaded_p = Parameters.from_json(roundtrip_file)

        assert loaded_p.values == p.values
        assert loaded_p.hyper == p.hyper

    def test_csv_to_file(self, tmp_path):
        """Test saving parameters to CSV file"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        csv_file = tmp_path / "params.csv"
        p.to_csv(csv_file)
        assert csv_file.exists()

    def test_csv_from_file(self, tmp_path):
        """Test loading parameters from CSV file"""
        h = copy.deepcopy(self.hyper)
        h["hypertrue"] = True
        p = Parameters(parameters=self.params, hyperparameters=h)
        csv_file = tmp_path / "params.csv"
        p.to_csv(csv_file)

        loaded_params = Parameters.from_csv(csv_file)
        assert loaded_params.values == p.values
        assert loaded_params.hyper == p.hyper

    def test_csv_roundtrip(self, tmp_path):
        """Test CSV serialization roundtrip"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        roundtrip_file = tmp_path / "roundtrip.csv"
        p.to_csv(roundtrip_file)
        loaded_p = Parameters.from_csv(roundtrip_file)

        assert loaded_p.values == p.values
        assert loaded_p.hyper == p.hyper

    def test_excel_to_file_not_implemented(self, tmp_path):
        """Test that the excel_to_file method is not implemented"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        with pytest.raises(NotImplementedError):
            p.to_excel(tmp_path / "params.xlsx")

    def test_excel_from_file_not_implemented(self, tmp_path):
        """Test that the excel_from_file method is not implemented"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        with pytest.raises(NotImplementedError):
            p.from_excel(tmp_path / "params.xlsx")

    def test_get_default_hyperparameters(self):
        """Test that the get_default_hyperparameters method returns the correct default hyperparameters"""
        p = Parameters()
        assert p.hyper == Parameters().get_default_hyperparameters()

    def test_get_default_hyperparameters_keys(self):
        """Test that the get_default_hyperparameters method returns the correct default hyperparameters"""
        keylist = [
            "timesteps",
            "timesteps_initialization",
            "scenario_trigger",
            "seed",
            "device",
            "requires_grad",
        ]
        p = Parameters()
        assert set(p.hyper.keys()) == set(keylist)

    def test_get_default_parameters(self):
        """Test that the get_default_parameters method correctly returns an empty dictionary"""
        p = Parameters()
        assert p.values == {}

    def test_boundary_validation_missing_bounds(self):
        """Test validation fails when bounds are missing"""
        invalid_params = copy.deepcopy(self.params)
        invalid_params["param1"].pop("lower bound")
        invalid_params["param1"].pop("upper bound")

        # Make a parameters child class implementing the get_default_parameters method
        # that returns the correct default parameters
        class TestParameters(Parameters):
            def get_default_parameters(self):
                return {
                    "param1": {"lower bound": 0.0, "upper bound": 2.0},
                    "param2": {"lower bound": 1.0, "upper bound": 3.0},
                }

        with pytest.raises(BoundaryError):
            TestParameters(parameters=invalid_params, hyperparameters=self.hyper)

    def test_boundary_validation_invalid_bounds(self):
        """Test validation fails when upper bound is less than lower bound"""
        invalid_params = copy.deepcopy(self.params)
        invalid_params["param1"]["lower bound"] = 2.0
        invalid_params["param1"]["upper bound"] = 1.0  # Upper < Lower
        with pytest.raises(BoundaryError):
            Parameters(parameters=invalid_params, hyperparameters=self.hyper)

    def test_boundary_validation_value_outside_bounds(self):
        """Test validation fails when parameter value is outside bounds"""
        invalid_params = copy.deepcopy(self.params)
        invalid_params["param1"]["value"] = 3.0  # Outside (0.0, 2.0)
        with pytest.raises(BoundaryError):
            Parameters(parameters=invalid_params, hyperparameters=self.hyper)

    def test_set_bound(self):
        """Test setting bounds"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)

        p.set_bound("param1", (0.5, 1.5))
        assert p.values["param1"]["Lower Bound"] == 0.5
        assert p.values["param1"]["Upper Bound"] == 1.5

    def test_set_notation(self):
        """Test setting notation"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        p.set_notation("param1", "new_notation")
        assert p.values["param1"]["notation"] == "new_notation"

    def test_set_unit(self):
        """Test setting unit"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        p.set_unit("param1", "new_unit")
        assert p.values["param1"]["unit"] == "new_unit"

    def test_vectorize_parameters(self):
        """Test vectorizing parameters"""
        p = Parameters(parameters=self.params, hyperparameters=self.hyper)
        pvectors = p.vectorize_parameters()
        assert isinstance(pvectors, dict)
        assert len(pvectors) == len(self.params)
        assert isinstance(pvectors["param1"], torch.Tensor)
