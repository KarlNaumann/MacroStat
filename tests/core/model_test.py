"""
pytest code for the Macrostat Core Model class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import pytest
import torch

from macrostat.core import Behavior, Model, Parameters, Scenarios, Variables


class ScenarioTestClass(Scenarios):
    """Test class for the Scenarios class"""

    def get_default_scenario_values(self) -> dict:
        """Get the default values for the scenarios"""
        return {k: 0.0 for k in ["shock1", "shock2", "shock3"]}


class VariableTestClass(Variables):
    """Test class for the Variables class"""

    def get_default_variable_values(self) -> dict:
        """Get the default values for the variables"""
        return {k: 0.0 for k in ["variable1", "variable2", "variable3"]}


class TestModel:
    """Tests for the Model class found in core/model.py"""

    # Sample test parameters
    params = {
        "param1": {
            "value": 1.0,
            "lower bound": 0.0,
            "upper bound": 2.0,
            "unit": "units",
            "notation": "p_1",
        }
    }

    hyper = {
        "timesteps": 500,
        "timesteps_initialization": 10,
        "scenario_trigger": 0,
        "seed": 42,
        "device": "cpu",
        "requires_grad": False,
    }

    scenarios = ScenarioTestClass(
        parameters=Parameters(params),
        scenarios={
            "test_scenario": {
                "shock1": 1.0,
                "shock2": torch.ones(50),
                "shock3": [1.0] * 50,
            }
        },
    )
    variables = VariableTestClass(parameters=Parameters(params))

    def test_init_defaults(self):
        """Test initialization with defaults"""
        model = Model()
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_hyper_dict(self):
        """Test initialization with parameters and hyperparameters provided"""
        params = Parameters()
        model = Model(hyperparameters=self.hyper, parameters=params)
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"
        assert model.parameters.hyper["timesteps"] == 500

    def test_init_with_param_class(self):
        """Test initialization with parameters class"""
        model = Model(parameters=Parameters(self.params))
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_scenarios(self):
        """Test initialization with scenario dictionary"""
        model = Model(scenarios=ScenarioTestClass(parameters=Parameters(self.params)))
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_variables(self):
        """Test initialization with variables class"""
        model = Model(variables=Variables(parameters=Parameters(self.params)))
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_behavior(self):
        """Test initialization with behavior class"""
        model = Model(behavior=Behavior)
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_none_behavior(self):
        """Test initialization with None behavior"""
        model = Model(behavior=None)
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_dict(self):
        """Test initialization with dictionary parameters"""
        model = Model(parameters=self.params, hyperparameters=self.hyper)
        assert isinstance(model.parameters, Parameters)
        assert model.parameters["param1"] == 1.0

    def test_simulate_default_behavior(self):
        """Test model simulation with default behavior"""
        model = Model(parameters=self.params, hyperparameters=self.hyper)
        with pytest.raises(NotImplementedError):
            model.simulate()

    def test_simulate_named_scenario(self):
        """Test model simulation with named scenario"""
        model = Model(
            parameters=self.params,
            hyperparameters=self.hyper,
            scenarios=self.scenarios,
            variables=self.variables,
        )
        with pytest.raises(NotImplementedError):
            model.simulate(scenario="test_scenario")

    def test_compute_theoretical_steady_state(self):
        """Test model compute_theoretical_steady_state functionality"""
        model = Model(parameters=self.params, hyperparameters=self.hyper)
        with pytest.raises(NotImplementedError):
            model.compute_theoretical_steady_state()

    def test_compute_theoretical_steady_state_named_scenario(self):
        """Test model compute_theoretical_steady_state functionality with named scenario"""
        model = Model(
            parameters=self.params,
            hyperparameters=self.hyper,
            scenarios=self.scenarios,
            variables=self.variables,
        )
        with pytest.raises(NotImplementedError):
            model.compute_theoretical_steady_state(scenario="test_scenario")

    def test_to_json(self, tmp_path):
        """Test model to_json functionality"""
        model = Model(
            parameters=self.params,
            hyperparameters=self.hyper,
            scenarios=self.scenarios,
            variables=self.variables,
        )

        # Save to JSON
        model.to_json(tmp_path / "model")

        # Check files exist
        assert (tmp_path / "model_params.json").exists()
        assert (tmp_path / "model_scenarios.json").exists()
        assert (tmp_path / "model_variables.json").exists()

    def test_from_json(self, tmp_path):
        """Test model from_json functionality"""
        # First create JSON files
        model = Model(parameters=self.params, hyperparameters=self.hyper)
        model.to_json(tmp_path / "model")

        # Load from JSON
        loaded_model = Model.from_json(
            f"{tmp_path}/model_params.json",
            f"{tmp_path}/model_scenarios.json",
            f"{tmp_path}/model_variables.json",
        )

        assert isinstance(loaded_model, Model)
        assert isinstance(loaded_model.parameters, Parameters)
        assert isinstance(loaded_model.scenarios, Scenarios)
        assert isinstance(loaded_model.variables, Variables)

    def test_json_roundtrip(self, tmp_path):
        """Test model JSON roundtrip functionality"""
        original_model = Model(parameters=self.params, hyperparameters=self.hyper)

        # Save to JSON
        original_model.to_json(tmp_path / "model")

        # Load back from JSON
        loaded_model = Model.from_json(
            f"{tmp_path}/model_params.json",
            f"{tmp_path}/model_scenarios.json",
            f"{tmp_path}/model_variables.json",
        )

        # Compare key attributes
        assert loaded_model.parameters["param1"] == original_model.parameters["param1"]
        assert loaded_model.parameters.hyper == original_model.parameters.hyper
        assert loaded_model.name == original_model.name

    def test_save_load(self, tmp_path):
        """Test model save and load functionality"""
        model = Model(parameters=self.params, hyperparameters=self.hyper)
        save_path = tmp_path / "model.pkl"

        # Test save
        model.save(save_path)
        assert save_path.exists()

        # Test load
        loaded_model = Model.load(save_path)
        assert isinstance(loaded_model, Model)
        assert loaded_model.parameters["param1"] == model.parameters["param1"]
