"""
pytest code for the Macrostat Core Model class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import pytest
from conftest import MockModel, MockParameters, MockScenarios, MockVariables

from macrostat.core import Behavior, Model, Parameters, Scenarios, Variables


class TestModel:
    """Tests for the Model class found in core/model.py"""

    mockparameters = MockParameters()
    mockvariables = MockVariables(parameters=mockparameters)
    mockscenarios = MockScenarios(parameters=mockparameters)

    def test_init_defaults(self):
        """Test initialization with defaults"""
        model = MockModel()
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_param_class(self):
        """Test initialization with parameters class"""
        model = Model(parameters=self.mockparameters)
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_scenarios(self):
        """Test initialization with scenario dictionary"""
        model = MockModel(scenarios=MockScenarios(parameters=self.mockparameters))
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_variables(self):
        """Test initialization with variables class"""
        model = MockModel(variables=Variables(parameters=self.mockparameters))
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_behavior(self):
        """Test initialization with behavior class"""
        model = MockModel(behavior=Behavior)
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_none_behavior(self):
        """Test initialization with None behavior"""
        model = MockModel(behavior=None)
        assert isinstance(model.parameters, Parameters)
        assert isinstance(model.scenarios, Scenarios)
        assert isinstance(model.variables, Variables)
        assert model.name == "model"

    def test_init_with_dict(self):
        """Test initialization with dictionary parameters"""
        model = MockModel(parameters=self.mockparameters)
        assert isinstance(model.parameters, Parameters)
        assert model.parameters["param1"] == 1.0

    def test_simulate_default_behavior(self):
        """Test model simulation with default behavior"""
        model = MockModel(parameters=self.mockparameters)
        with pytest.raises(NotImplementedError):
            model.simulate()

    def test_training_instance(self):
        """Test model simulation with default behavior"""
        model = MockModel(parameters=self.mockparameters)
        behav = model.get_model_training_instance()
        assert isinstance(behav, Behavior)

    def test_simulate_named_scenario(self):
        """Test model simulation with named scenario"""
        model = MockModel(
            parameters=self.mockparameters,
            scenarios=self.mockscenarios,
            variables=self.mockvariables,
        )
        model.scenarios.add_scenario(timeseries={"shock1": 1.0}, name="test_scenario")
        with pytest.raises(NotImplementedError):
            model.simulate(scenario="test_scenario")

    def test_compute_theoretical_steady_state(self):
        """Test model compute_theoretical_steady_state functionality"""
        model = MockModel(parameters=self.mockparameters)
        with pytest.raises(NotImplementedError):
            model.compute_theoretical_steady_state()

    def test_compute_theoretical_steady_state_named_scenario(self):
        """Test model compute_theoretical_steady_state functionality with named scenario"""
        model = MockModel(
            parameters=self.mockparameters,
            scenarios=self.mockscenarios,
            variables=self.mockvariables,
        )
        with pytest.raises(NotImplementedError):
            model.compute_theoretical_steady_state(scenario="test_scenario")

    def test_to_json(self, tmp_path):
        """Test model to_json functionality"""
        model = MockModel(
            parameters=self.mockparameters,
            scenarios=self.mockscenarios,
            variables=self.mockvariables,
        )

        # Save to JSON
        model.to_json(tmp_path / "model")

        # Check files exist
        assert (tmp_path / "model_params.json").exists()
        assert (tmp_path / "model_scenarios.json").exists()
        assert (tmp_path / "model_variables.json").exists()
