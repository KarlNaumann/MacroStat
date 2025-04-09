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

    def get_default_scenario_values(self) -> dict:
        """Get the default values for the scenarios"""
        return {k: 0.0 for k in ["shock1", "shock2", "shock3"]}


class TestScenarios:
    """Tests for the Scenarios class found in models/scenarios.py"""

    # Sample parameters
    params = Parameters(
        parameters={
            "param1": {
                "value": 1.0,
                "lower bound": 0.0,
                "upper bound": 2.0,
                "unit": "units",
                "notation": "p_1",
            }
        },
        hyperparameters={
            "timesteps": 100,
            "timesteps_initialization": 10,
            "scenario_trigger": 50,
            "seed": 42,
            "device": "cpu",
            "requires_grad": False,
        },
    )

    # Sample scenarios
    scenarios = {
        "test_scenario": {
            "shock1": 1.0,
            "shock2": torch.ones(50),
            "shock3": [1.0] * 50,
        }
    }

    scenario_info = {
        1: {
            "Name": "TestScenario1",
            "Colour": "#000000",
            "Index": torch.arange(params["timesteps"] - params["scenario_trigger"]),
        }
    }

    def test_init(self):
        """Test initialization with different scenario combinations"""
        # Test with no scenarios provided
        s = Scenarios(parameters=self.params)
        # Scenario timeseries should be a dictionary
        assert isinstance(s.timeseries, dict)
        # Timeseries should have one key (default scenario)
        assert set(s.timeseries.keys()) == {0}
        # Scenario info should be a dictionary
        assert isinstance(s.info, dict)
        # Scenario info should have one key (default scenario)
        assert set(s.info.keys()) == set(s.timeseries.keys())
        # Current scenario should be 0
        assert s.current_scenario == 0
        # Calibration variables should be empty
        assert s.calibration_variables == []

    def test_init_with_scenarios(self):
        """Test initialization with scenarios"""
        s = ScenarioTestClass(parameters=self.params, scenarios=self.scenarios)
        # Scenario timeseries should be a dictionary
        assert isinstance(s.timeseries, dict)
        # Timeseries should have one key (default scenario)
        assert set(s.timeseries.keys()) == {0, 1}
        # Scenario info should be a dictionary
        assert isinstance(s.info, dict)
        # Scenario info should have one key (default scenario)
        assert set(s.info.keys()) == set(s.timeseries.keys())
        # Current scenario should be 0
        assert s.current_scenario == 0
        # Calibration variables should be empty
        assert s.calibration_variables == []

    def test_init_with_scenario_info(self):
        """Test initialization with scenario info"""
        s = ScenarioTestClass(
            parameters=self.params,
            scenarios=self.scenarios,
            scenario_info=self.scenario_info,
        )
        # Scenario info should be a dictionary
        assert isinstance(s.info, dict)
        # Scenario info should have two keys (default scenario and test scenario)
        assert len(s.info) == 2
        assert set(s.info.keys()) == set(s.timeseries.keys())
        # Current scenario should be 0
        assert s.current_scenario == 0
        # Calibration variables should be empty
        assert s.calibration_variables == []

    def test_getitem(self):
        """Test getting scenario timeseries"""
        s = ScenarioTestClass(parameters=self.params, scenarios=self.scenarios)

        # Test getting by index and name
        assert torch.all(s[0, "shock1"] == s.timeseries[0]["shock1"])
        assert torch.all(s["test_scenario", "shock1"] == s.timeseries[1]["shock1"])

        # Test invalid scenario name
        with pytest.raises(ValueError):
            s["invalid_scenario", "shock1"]

        # Test invalid item format
        with pytest.raises(Exception):
            s["invalid_format"]

    def test_add_scenario_auto_naming(self):
        """Test scenario auto-naming when no name provided"""
        s = ScenarioTestClass(parameters=self.params)
        test_data = {"shock1": 1.0}
        s.add_scenario(test_data)
        assert 0 in s.timeseries
        assert s.info[len(s.info) - 1]["Name"] == f"Scenario.{len(s.info) - 1}"

    def test_add_scenario_explicit_name(self):
        """Test adding scenario with explicit name and colour"""
        s = ScenarioTestClass(parameters=self.params)
        test_data = {"shock1": 1.0}
        s.add_scenario(test_data, name="test", colour="red")
        assert 1 in s.timeseries
        assert "test" == s.info[len(s.info) - 1]["Name"]
        assert "red" == s.info[len(s.info) - 1]["Colour"]

    def test_add_scenario_constant_value(self):
        """Test adding scenario with constant value"""
        s = ScenarioTestClass(parameters=self.params)
        test_data = {"shock1": 1.0}
        s.add_scenario(test_data)
        trigger = self.params["scenario_trigger"]
        assert torch.all(s[1, "shock1"][trigger:] == 1.0)

    def test_add_scenario_tensor(self):
        """Test adding scenario with tensor timeseries"""
        s = ScenarioTestClass(parameters=self.params)
        test_data = {"shock1": torch.ones(50)}
        s.add_scenario(test_data)
        trigger = self.params["scenario_trigger"]
        t = min(50, self.params["timesteps"] - trigger)
        assert torch.allclose(s[1, "shock1"][trigger : trigger + t, 0], torch.ones(t))

    def test_add_scenario_list(self):
        """Test adding scenario with list timeseries"""
        s = ScenarioTestClass(parameters=self.params)
        test_data = {"shock1": [1.0] * 50}
        s.add_scenario(test_data)
        trigger = self.params["scenario_trigger"]
        t = min(50, self.params["timesteps"] - trigger)
        assert torch.allclose(s[1, "shock1"][trigger : trigger + t, 0], torch.ones(t))

    def test_add_scenario_pandas_series(self):
        """Test adding scenario with pandas Series timeseries"""
        s = ScenarioTestClass(parameters=self.params)
        test_data = {"shock1": pd.Series([1.0] * 50, index=range(100, 150))}
        s.add_scenario(test_data)
        trigger = self.params["scenario_trigger"]
        t = min(50, self.params["timesteps"] - trigger)
        assert torch.allclose(s[1, "shock1"][trigger : trigger + t, 0], torch.ones(t))
        assert isinstance(s.info[1]["Index"], np.ndarray)
        assert len(s.info[1]["Index"]) == test_data["shock1"].shape[0]
        assert np.allclose(s.info[1]["Index"], torch.arange(100, 150))

    def test_add_scenario_invalid_variable(self):
        """Test adding scenario with invalid variable raises error"""
        s = ScenarioTestClass(parameters=self.params)
        with pytest.raises(KeyError):
            s.add_scenario({"invalid_var": 1.0})

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

    def test_get_default_scenario(self):
        """Test getting the default scenario"""
        s = ScenarioTestClass(parameters=self.params)
        default_scenario = s.get_default_scenario()

        for s in ["shock1", "shock2", "shock3"]:
            assert torch.allclose(default_scenario[s], torch.zeros(100))
            assert not default_scenario[s].requires_grad
            assert default_scenario[s].shape == (100, 1)

    def test_get_default_scenario_values(self):
        """Test getting the default scenario values"""
        s = ScenarioTestClass(parameters=self.params)
        default_values = s.get_default_scenario_values()
        assert default_values == {"shock1": 0.0, "shock2": 0.0, "shock3": 0.0}

        s = Scenarios(parameters=self.params)
        assert s.get_default_scenario_values() == {}

    def test_get_scenario_index(self):
        """Test getting the scenario index"""
        s = ScenarioTestClass(parameters=self.params)
        s.add_scenario(self.scenarios["test_scenario"], name="test_scenario")
        assert s.get_scenario_index("test_scenario") == 1
        with pytest.raises(ValueError):
            s.get_scenario_index("invalid_scenario")

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
        assert not params["shock3"].requires_grad

    def test_verify_scenario_info_missing_scenario_info(self):
        """Test verifying scenario info"""
        s = ScenarioTestClass(
            parameters=self.params,
            scenarios=self.scenarios,
        )
        s.info.pop(1)
        with pytest.raises(ValueError):
            s.verify_scenario_info()

    def test_verify_scenario_info_missing_scenario_info_key(self):
        """Test verifying scenario info"""
        with pytest.raises(ValueError):
            ScenarioTestClass(
                parameters=self.params,
                scenarios=self.scenarios,
                scenario_info={1: {"Name": "TestScenario1"}},
            )
