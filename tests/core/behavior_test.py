"""
pytest code for the Behavior class
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import pytest
import torch

from macrostat.core import Behavior, Parameters, Scenarios, Variables


class TestBehavior:
    """Tests for the Behavior class found in models/behavior.py"""

    @pytest.fixture
    def behavior_instance(self):
        """Create a basic behavior instance for testing"""
        parameters = Parameters()
        scenarios = Scenarios(parameters=parameters)
        variables = Variables(parameters=parameters)

        return Behavior(
            parameters=parameters,
            scenarios=scenarios,
            variables=variables,
            record=False,
            scenario=0,
        )

    def test_init(self, behavior_instance):
        """Test initialization of Behavior class"""
        assert isinstance(behavior_instance, Behavior)
        assert isinstance(behavior_instance.params, torch.nn.ParameterDict)
        assert isinstance(behavior_instance.hyper, dict)
        assert isinstance(behavior_instance.scenarios, torch.nn.ParameterDict)
        assert isinstance(behavior_instance.variables, Variables)
        assert behavior_instance.record is False
        assert behavior_instance.scenarioID == 0
        assert behavior_instance.differentiable is False
        assert behavior_instance.debug is False

    def test_forward_initialization(self, behavior_instance):
        """Test the initialization phase of the forward pass"""
        behavior_instance.hyper["T"] = 10
        behavior_instance.hyper["seed"] = 42
        behavior_instance.hyper["requires_grad"] = False
        behavior_instance.hyper["device"] = "cpu"
        behavior_instance.hyper["timesteps_initialization"] = 2
        behavior_instance.hyper["timesteps"] = 5

        # Mock the initialize method since it's abstract
        behavior_instance.initialize = lambda: None
        behavior_instance.step = lambda t, s: None

        # Run forward pass
        behavior_instance.forward()

        # Check state and history were initialized
        assert isinstance(behavior_instance.state, dict)
        assert isinstance(behavior_instance.history, dict)
        assert isinstance(behavior_instance.prior, dict)

    def test_forward_recording(self, behavior_instance):
        """Test recording functionality during forward pass"""
        behavior_instance.hyper["T"] = 10
        behavior_instance.hyper["seed"] = 42
        behavior_instance.hyper["requires_grad"] = False
        behavior_instance.hyper["device"] = "cpu"
        behavior_instance.hyper["timesteps_initialization"] = 2
        behavior_instance.hyper["timesteps"] = 5
        behavior_instance.record = True

        # Mock required methods
        behavior_instance.initialize = lambda: None
        behavior_instance.step = lambda t, s: None

        # Run forward pass
        behavior_instance.forward()

        # Check timeseries was populated
        assert all(
            isinstance(v, torch.Tensor)
            for v in behavior_instance.variables.timeseries.values()
        )

    def test_forward_scenario_indexing(self, behavior_instance):
        """Test scenario indexing during forward pass"""
        behavior_instance.hyper["T"] = 3
        behavior_instance.hyper["seed"] = 42
        behavior_instance.hyper["requires_grad"] = False
        behavior_instance.hyper["device"] = "cpu"
        behavior_instance.hyper["timesteps_initialization"] = 1
        behavior_instance.hyper["timesteps"] = 3

        # Add a test scenario variable
        behavior_instance.scenarios["test"] = torch.nn.Parameter(torch.ones(3, 1))

        # Mock required methods and track scenario values
        behavior_instance.initialize = lambda: None
        scenario_values = []
        behavior_instance.step = lambda t, s: scenario_values.append(s["test"].item())

        # Run forward pass
        behavior_instance.forward()

        # Check scenario values were correctly indexed
        assert len(scenario_values) == 1  # one series only
        assert all(v == 1.0 for v in scenario_values)

    def test_forward_history_update(self, behavior_instance):
        """Test history updates during forward pass"""
        behavior_instance.hyper["T"] = 5
        behavior_instance.hyper["seed"] = 42
        behavior_instance.hyper["requires_grad"] = False
        behavior_instance.hyper["device"] = "cpu"
        behavior_instance.hyper["timesteps_initialization"] = 2
        behavior_instance.hyper["timesteps"] = 4

        # Mock required methods
        behavior_instance.initialize = lambda: None
        behavior_instance.step = lambda t, s: None

        # Run forward pass
        behavior_instance.forward()

        # Check history was updated
        assert isinstance(behavior_instance.history, dict)
        assert behavior_instance.prior is not None

    def test_diffwhere(self, behavior_instance):
        """Test the differentiable where function"""
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([4.0, 5.0, 6.0])
        condition = torch.tensor([1.0, -1.0, 1.0])

        behavior_instance.hyper["diffwhere"] = True
        behavior_instance.hyper["sigmoid_constant"] = 10.0

        result = behavior_instance.diffwhere(condition, x1, x2)
        assert torch.is_tensor(result)
        assert result.shape == x1.shape

    def test_tanhmask(self, behavior_instance):
        """Test the tanh mask function"""
        behavior_instance.hyper["tanh_constant"] = 10.0
        x = torch.tensor([-1.0, 0.0, 1.0])

        result = behavior_instance.tanhmask(x)
        assert torch.is_tensor(result)
        assert result.shape == x.shape
        assert torch.all(result >= 0) and torch.all(result <= 1)

    def test_diffmin(self, behavior_instance):
        """Test the differentiable min function"""
        behavior_instance.hyper["min_constant"] = 10.0
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([2.0, 1.0, 4.0])

        result = behavior_instance.diffmin(x1, x2)
        assert torch.is_tensor(result)
        assert result.shape == x1.shape

    def test_diffmax(self, behavior_instance):
        """Test the differentiable max function"""
        behavior_instance.hyper["max_constant"] = 10.0
        x1 = torch.tensor([1.0, 2.0, 3.0])
        x2 = torch.tensor([2.0, 1.0, 4.0])

        result = behavior_instance.diffmax(x1, x2)
        assert torch.is_tensor(result)
        assert result.shape == x1.shape

    def test_diffmin_v(self, behavior_instance):
        """Test the vector differentiable min function"""
        behavior_instance.hyper["min_constant"] = 10.0
        x = torch.tensor([1.0, 2.0, 3.0])

        result = behavior_instance.diffmin_v(x)
        assert torch.is_tensor(result)
        assert result.dim() == 0

    def test_diffmax_v(self, behavior_instance):
        """Test the vector differentiable max function"""
        behavior_instance.hyper["max_constant"] = 10.0
        x = torch.tensor([1.0, 2.0, 3.0])

        result = behavior_instance.diffmax_v(x)
        assert torch.is_tensor(result)
        assert result.dim() == 0

    def test_unimplemented_methods(self, behavior_instance):
        """Test that unimplemented methods raise NotImplementedError"""
        with pytest.raises(NotImplementedError):
            behavior_instance.initialize()

        with pytest.raises(NotImplementedError):
            behavior_instance.step(t=0, scenario={})
