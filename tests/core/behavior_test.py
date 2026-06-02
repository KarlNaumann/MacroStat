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
from conftest import (
    MockParameters,
    MockScenarios,
    MockVariables,
    VectorMockParameters,
    VectorMockScenarios,
    VectorMockVariables,
)

from macrostat.core import Behavior, LinearConstraint, Parameters, Variables


class TestBehavior:
    """Tests for the Behavior class found in models/behavior.py"""

    @pytest.fixture
    def behavior_instance(self):
        """Create a basic behavior instance for testing"""
        return Behavior(
            parameters=MockParameters(),
            scenarios=MockScenarios(parameters=MockParameters()),
            variables=MockVariables(parameters=MockParameters()),
            scenario=0,
        )

    @pytest.fixture
    def behavior_instance_simple(self):
        """Create a basic behavior instance for testing"""

        class SimpleBehavior(Behavior):
            def initialize(self):
                self.state["variable1"] = torch.tensor(0.0)
                self.state["variable2"] = torch.tensor(0.0)

            def compute_theoretical_steady_state_per_step(self, **kwargs):
                self.state["variable1"] = torch.tensor(1.0)
                self.state["variable2"] = torch.tensor(2.0)

        return SimpleBehavior(
            parameters=MockParameters(),
            scenarios=MockScenarios(parameters=MockParameters()),
            variables=MockVariables(parameters=MockParameters()),
            scenario=0,
        )

    def test_init(self, behavior_instance):
        """Test initialization of Behavior class"""
        assert isinstance(behavior_instance, Behavior)
        assert isinstance(behavior_instance.params, torch.nn.ParameterDict)
        assert isinstance(behavior_instance.hyper, dict)
        assert isinstance(behavior_instance.scenarios, torch.nn.ParameterDict)
        assert isinstance(behavior_instance.variables, Variables)
        assert behavior_instance.scenarioID == 0
        assert behavior_instance.differentiable is False
        assert behavior_instance.debug is False

    def test_forward_initialization(self, behavior_instance):
        """Test the initialization phase of the forward pass"""
        # Mock the initialize method since it's abstract
        behavior_instance.initialize = lambda: None
        behavior_instance.step = lambda t, scenario, params: None

        # Run forward pass
        behavior_instance.forward()

        # Check state and history were initialized
        assert isinstance(behavior_instance.state, dict)
        assert isinstance(behavior_instance.history, dict)
        assert isinstance(behavior_instance.prior, dict)

    def test_forward_recording(self, behavior_instance):
        """Test recording functionality during forward pass"""
        # Mock required methods
        behavior_instance.initialize = lambda: None
        behavior_instance.step = lambda t, scenario, params: None

        # Run forward pass
        behavior_instance.forward()

        # Check timeseries was populated
        assert all(
            isinstance(v, torch.Tensor)
            for v in behavior_instance.variables.timeseries.values()
        )

    def test_forward_scenario_indexing(self, behavior_instance):
        """Test scenario indexing during forward pass"""
        # Add a test scenario variable
        t = behavior_instance.hyper["timesteps"]
        behavior_instance.scenarios["test"] = torch.nn.Parameter(torch.ones(t, 1))

        # Mock required methods and track scenario values
        behavior_instance.initialize = lambda: None
        scenario_values = []
        behavior_instance.step = lambda t, scenario, params: scenario_values.append(
            scenario["test"].item()
        )

        # Run forward pass
        behavior_instance.forward()

        # Check scenario values were correctly indexed
        assert (
            len(scenario_values)
            == t - behavior_instance.hyper["timesteps_initialization"]
        )  # one series only
        assert all(v == 1.0 for v in scenario_values)

    def test_forward_history_update(self, behavior_instance):
        """Test history updates during forward pass"""
        # Mock required methods
        behavior_instance.initialize = lambda: None
        behavior_instance.step = lambda t, scenario, params: None

        # Run forward pass
        behavior_instance.forward()

        # Check history was updated
        assert isinstance(behavior_instance.history, dict)
        assert behavior_instance.prior is not None

    def test_apply_parameter_shocks_no_shocks(self, behavior_instance):
        """Test the apply_parameter_shocks method with no shocks.

        In this case, the method should just return a dictionary where the
        values and the objects are the same as in the original behavior_instance.params
        dictionary.
        """
        params = behavior_instance.apply_parameter_shocks(t=0, scenario={})
        for key, value in behavior_instance.params.items():
            assert params[key] == value

    def test_apply_parameter_shocks_multiplicative_shock(self, behavior_instance):
        """Test the apply_parameter_shocks method with a multiplicative shock"""

        scenario = {"param1_multiply": 2.0, "param2_multiply": 2.0}
        params = behavior_instance.apply_parameter_shocks(t=0, scenario=scenario)
        assert params["param1"] == 2.0
        assert params["param2"] == 4.0

    def test_apply_parameter_shocks_additive_shock(self, behavior_instance):
        """Test the apply_parameter_shocks method with an additive shock"""

        scenario = {"param1_add": 1.0, "param2_add": 1.0}
        params = behavior_instance.apply_parameter_shocks(t=0, scenario=scenario)
        assert params["param1"] == 2.0
        assert params["param2"] == 3.0

    def test_apply_parameter_shocks_multiplicative_and_additive_shock(
        self, behavior_instance
    ):
        """Test the apply_parameter_shocks method with a multiplicative and additive shock

        In case of a combined shock, it should first apply the multiplicative shock and
        only afterwards the additive shock.
        """

        scenario = {
            "param1_multiply": 2.0,
            "param2_multiply": 2.0,
            "param1_add": 1.0,
            "param2_add": 1.0,
        }
        params = behavior_instance.apply_parameter_shocks(t=0, scenario=scenario)
        assert params["param1"] == 3.0
        assert params["param2"] == 5.0

    def test_compute_theoretical_steady_state(self, behavior_instance):
        """Test the compute_theoretical_steady_state method"""
        behavior_instance.initialize = lambda: None
        with pytest.raises(NotImplementedError):
            behavior_instance.compute_theoretical_steady_state()

    def test_compute_theoretical_steady_state_none_function(
        self, behavior_instance_simple
    ):
        """Test the compute_theoretical_steady_state_per_step method"""
        behavior_instance_simple.compute_theoretical_steady_state()
        assert behavior_instance_simple.state["variable1"] == 1.0
        assert behavior_instance_simple.state["variable2"] == 2.0

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

    def test_tanhmask_sign_correctness(self, behavior_instance):
        """tanhmask saturates to 0 for x<<0, 1 for x>>0, exactly 0.5 at x=0."""
        behavior_instance.hyper["tanh_constant"] = 1.0e3

        result = behavior_instance.tanhmask(torch.tensor([-1.0, -1e-2, 0.0, 1e-2, 1.0]))
        assert result[0].item() == pytest.approx(0.0, abs=1e-6)
        assert result[2].item() == pytest.approx(0.5, abs=1e-9)
        assert result[4].item() == pytest.approx(1.0, abs=1e-6)
        assert result[1].item() < 0.05
        assert result[3].item() > 0.95

    def test_tanhmask_requires_grad_toggle(self, behavior_instance):
        """tanhmask output requires_grad is gated by hyper['requires_grad'].

        Values are identical regardless of the flag — only the autograd
        tracking changes.
        """
        behavior_instance.hyper["tanh_constant"] = 100.0
        x = torch.tensor([-0.5, 0.0, 0.5])

        behavior_instance.hyper["requires_grad"] = False
        out_off = behavior_instance.tanhmask(x)
        assert out_off.requires_grad is False

        behavior_instance.hyper["requires_grad"] = True
        out_on = behavior_instance.tanhmask(x)
        assert out_on.requires_grad is True

        assert torch.allclose(out_off, out_on, atol=1e-12)

    def test_tanhmask_gradient_flow(self, behavior_instance):
        """tanhmask gradient is non-zero near x=0 and decays to 0 for |x|>>0."""
        behavior_instance.hyper["tanh_constant"] = 10.0
        behavior_instance.hyper["requires_grad"] = True

        x = torch.tensor([0.0, 1.0], requires_grad=True)
        out = behavior_instance.tanhmask(x).sum()
        out.backward()

        grad = x.grad
        assert grad is not None
        assert grad[0].abs().item() > 1.0
        assert grad[1].abs().item() < 1e-3

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


class ConstrainedScalarParameters(Parameters):
    """Parameters with a scalar sum-to-1 constraint on three entries."""

    def get_default_parameters(self):
        return {
            "a": {
                "value": 0.3,
                "lower bound": -1.0,
                "upper bound": 1.0,
                "unit": ".",
                "notation": "a",
            },
            "b": {
                "value": 0.5,
                "lower bound": -1.0,
                "upper bound": 1.0,
                "unit": ".",
                "notation": "b",
            },
            "c": {
                "value": 0.2,
                "lower bound": -1.0,
                "upper bound": 1.0,
                "unit": ".",
                "notation": "c",
            },
        }

    def get_constraints(self):
        return (LinearConstraint(param_names=("a", "b", "c"), target=1.0),)


class ConstrainedScalarVariables(Variables):
    def get_default_variables(self) -> dict:
        return {
            "output": {
                "notation": r"Y",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            }
        }


class TestBehaviorConstraints:
    """Step-time constraint enforcement through ``apply_parameter_shocks``."""

    def _make_behavior(self, ParamCls, VarCls, ScnCls):
        p = ParamCls()
        return Behavior(
            parameters=p,
            scenarios=ScnCls(parameters=p),
            variables=VarCls(parameters=p),
            scenario=0,
        )

    def test_does_not_snapshot_constraints(self):
        """Behavior holds the Parameters instance, not a frozen list."""
        b = self._make_behavior(
            ConstrainedScalarParameters,
            ConstrainedScalarVariables,
            MockScenarios,
        )
        assert not hasattr(b, "constraints")
        assert b.parameters is not None

    def test_scalar_constraint_survives_shock(self):
        """Adding-up holds after a free parameter is shocked."""
        b = self._make_behavior(
            ConstrainedScalarParameters,
            ConstrainedScalarVariables,
            MockScenarios,
        )
        scenario = {"a_add": torch.tensor(0.2)}
        params = b.apply_parameter_shocks(t=0, scenario=scenario)
        total = params["a"] + params["b"] + params["c"]
        assert torch.isclose(total, torch.tensor(1.0), atol=1e-6)
        # Derived parameter absorbed the shock: c = 1 - 0.5 - 0.5 = 0
        assert torch.isclose(params["c"], torch.tensor(0.0), atol=1e-6)

    def test_scalar_constraint_noop_without_shock(self):
        """With no shocks, derived value equals the init-time value."""
        b = self._make_behavior(
            ConstrainedScalarParameters,
            ConstrainedScalarVariables,
            MockScenarios,
        )
        params = b.apply_parameter_shocks(t=0, scenario={})
        total = params["a"] + params["b"] + params["c"]
        assert torch.isclose(total, torch.tensor(1.0), atol=1e-6)

    def test_vector_constraint_survives_shock(self):
        """1-D sector-indexed constraint holds after a sectoral shock."""
        b = self._make_behavior(
            VectorMockParameters,
            VectorMockVariables,
            VectorMockScenarios,
        )
        # Initial: Household=0.6, Firm=0.4, summing to 1.
        # Shock one sector's multiplier; derived sector (sorted order
        # makes Household the second entry) absorbs it.
        scenario = {"Firm_Share_multiply": torch.tensor(0.5)}
        params = b.apply_parameter_shocks(t=0, scenario=scenario)
        # Share is 1-D with two sectors; must sum to 1 after apply.
        assert torch.isclose(params["Share"].sum(), torch.tensor(1.0), atol=1e-6)

    def test_vector_constraint_grad_flow(self):
        """Gradient from derived slot flows to free slot with sign -1."""
        p = VectorMockParameters()
        # Make the free parameter a leaf that requires grad. Identify it
        # by asking the resolver which slot the derived parameter is in:
        # free_params[0] is the other one.
        constraint = p.get_constraints()[0]
        resolver = p.get_constraint_resolver()
        free_loc = resolver.locate(constraint.free_params[0])
        derived_loc = resolver.locate(constraint.derived_param)
        assert free_loc.tensor_key == "Share"
        assert derived_loc.tensor_key == "Share"

        # Build a fresh params dict with a grad-tracking free slot.
        share = torch.zeros(2, requires_grad=False)
        free_leaf = torch.tensor(0.6, requires_grad=True)
        # Compose via stack so the resulting tensor is non-leaf and
        # tracks gradients into free_leaf.
        if free_loc.index == (0,):
            share = torch.stack([free_leaf, torch.zeros(())])
        else:
            share = torch.stack([torch.zeros(()), free_leaf])
        params = {"Share": share}
        constraint.apply(params, resolver)

        # Loss depends only on the derived slot; grad should flow back
        # to the free leaf with value -1.
        loss = params["Share"][derived_loc.index[0]]
        loss.backward()
        assert free_leaf.grad is not None
        assert torch.isclose(free_leaf.grad, torch.tensor(-1.0))
