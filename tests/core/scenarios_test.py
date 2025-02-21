"""
pytest code for the Scenarios class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import os

import numpy as np
import pandas as pd
import pytest
import torch

from macrostat.core import Parameters, Scenarios


class ScenarioTestClass(Scenarios):
    """Test class for the Scenarios class"""

    def get_default_scenario_values(self):
        """Get the default values for the scenarios"""
        return {k: 0.0 for k in ["shock1", "shock2", "shock3"]}


class TestScenarios:
    """Tests for the Scenarios class found in models/scenarios.py"""

    # Sample parameters
    params = Parameters(
        parameters={"param1": 1.0},
        hyperparameters={
            "timesteps": 100,
            "timesteps_initialization": 10,
            "scenario_trigger": 50,
            "seed": 42,
            "device": "cpu",
            "requires_grad": False,
            "T": 100,
            "scenario_trigger": 50,
        },
        bounds={"param1": (0.0, 2.0)},
    )

    # Sample scenarios
    scenarios = {
        "test_scenario": {
            "shock1": 1.0,
            "shock2": torch.ones(50),
            "shock3": [1.0] * 50,
        }
    }

    def test_init(self):
        """Test initialization with different scenario combinations"""
        # Test with no scenarios provided
        s = Scenarios(parameters=self.params)
        assert isinstance(s.timeseries, dict)
        assert isinstance(s.info, dict)
        assert s.current_scenario == 0

    def test_add_scenario(self):
        """Test adding scenarios and name assignment"""
        s = ScenarioTestClass(parameters=self.params)

        # Test adding scenario with different types of timeseries
        test_data = {
            "var1": 1.0,  # constant
            "var2": torch.ones(50),  # tensor
            "var3": [1.0] * 50,  # list
            "var4": pd.Series([1.0] * 50, index=range(50)),  # pandas series
        }

        # Override get_default_scenario_values to include test variables
        def get_default_values():
            return {k: 0.0 for k in test_data.keys()}

        s.get_default_scenario_values = get_default_values

        # Test auto-naming when name=None
        s.add_scenario(None, test_data)
        assert 0 in s.timeseries
        assert s.info[len(s.info) - 1]["Name"] == 1

        # Test explicit naming
        s.add_scenario("test", test_data)
        assert 1 in s.timeseries
        assert "test" == s.info[len(s.info) - 1]["Name"]

        # Check each type of timeseries was handled correctly
        trigger = self.params["scenario_trigger"]

        # Check constant value
        assert torch.all(s.timeseries[1]["var1"][trigger:] == 1.0)

        # Check tensor
        t = min(50, self.params["T"] - trigger)
        assert torch.allclose(
            s.timeseries[1]["var2"][trigger : trigger + t, 0], torch.ones(t)
        )

        # Check list
        assert torch.allclose(
            s.timeseries[1]["var3"][trigger : trigger + t, 0], torch.ones(t)
        )

        # Check pandas series
        assert torch.allclose(
            s.timeseries[1]["var4"][trigger : trigger + t, 0], torch.ones(t)
        )

        # Check pandas series index was stored
        assert isinstance(s.info[1]["Index"], np.ndarray)
        assert len(s.info[1]["Index"]) == test_data["var4"].shape[0]

        # Test adding invalid variable raises error
        with pytest.raises(KeyError):
            s.add_scenario("test2", {"invalid_var": 1.0})

    def test_json_io(self, tmpdir):
        """Test JSON file I/O"""
        s = ScenarioTestClass(parameters=self.params, scenarios=self.scenarios)

        # Save to JSON
        json_path = os.path.join(tmpdir, "test_scenarios.json")
        s.to_json(json_path)

        # Load from JSON
        s2 = ScenarioTestClass.from_json(json_path, self.params)

        # Compare timeseries
        for sc_id in s.timeseries:
            for var in s.timeseries[sc_id]:
                assert torch.allclose(
                    s.timeseries[sc_id][var], s2.timeseries[sc_id][var]
                )

    def test_to_nn_parameters(self):
        """Test conversion to PyTorch parameters"""
        s = ScenarioTestClass(
            parameters=self.params,
            scenarios=self.scenarios,
            calibration_variables=["shock1"],
        )

        params = s.to_nn_parameters()
        assert isinstance(params, torch.nn.ParameterDict)
        assert params["shock1"].requires_grad
        assert not params["shock2"].requires_grad
