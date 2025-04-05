import importlib
import inspect
from pathlib import Path
from typing import List, Tuple, Type

import pytest

from macrostat.core.behavior import Behavior
from macrostat.core.model import Model
from macrostat.core.parameters import Parameters
from macrostat.core.scenarios import Scenarios
from macrostat.core.variables import Variables


def discover_model_components() -> List[
    Tuple[
        Type[Model],
        Type[Parameters],
        Type[Behavior],
        Type[Variables],
        Type[Scenarios],
    ]
]:
    """Automatically discover all model components in the models directory"""
    models_dir = Path("src/macrostat/models")
    model_components = []

    # Skip special directories and files
    skip_dirs = {"__pycache__", "base"}

    for model_dir in models_dir.glob("*"):
        if not model_dir.is_dir() or model_dir.name in skip_dirs:
            continue

        try:
            # Import all component modules for the model
            model_module = importlib.import_module(
                f"macrostat.models.{model_dir.name}.{model_dir.name.lower()}"
            )
            params_module = importlib.import_module(
                f"macrostat.models.{model_dir.name}.parameters"
            )
            behavior_module = importlib.import_module(
                f"macrostat.models.{model_dir.name}.behavior"
            )
            variables_module = importlib.import_module(
                f"macrostat.models.{model_dir.name}.variables"
            )
            scenarios_module = importlib.import_module(
                f"macrostat.models.{model_dir.name}.scenarios"
            )

            # Get the main classes from each module
            model_class = None
            params_class = None
            behavior_class = None
            variables_class = None
            scenarios_class = None

            # Find the model class
            for name, obj in inspect.getmembers(model_module):
                if inspect.isclass(obj) and issubclass(obj, Model) and obj != Model:
                    model_class = obj
                    break

            # Find the parameters class
            for name, obj in inspect.getmembers(params_module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Parameters)
                    and obj != Parameters
                ):
                    params_class = obj
                    break

            # Find the behavior class
            for name, obj in inspect.getmembers(behavior_module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Behavior)
                    and obj != Behavior
                ):
                    behavior_class = obj
                    break

            # Find the variables class
            for name, obj in inspect.getmembers(variables_module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Variables)
                    and obj != Variables
                ):
                    variables_class = obj
                    break

            # Find the scenarios class
            for name, obj in inspect.getmembers(scenarios_module):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, Scenarios)
                    and obj != Scenarios
                ):
                    scenarios_class = obj
                    break

            if all(
                [
                    model_class,
                    params_class,
                    behavior_class,
                    variables_class,
                    scenarios_class,
                ]
            ):
                model_components.append(
                    (
                        model_class,
                        params_class,
                        behavior_class,
                        variables_class,
                        scenarios_class,
                    )
                )

        except ImportError as e:
            print(f"Skipping {model_dir.name}: {e}")
            continue

    return model_components


class BaseModelTest:
    """Base test class containing common tests for all macrostat models"""

    @pytest.fixture
    def model_components(self, request):
        """Fixture to create model instance and its components"""
        model_class, params_class, behavior_class, vars_class, scenarios_class = (
            request.param
        )

        # Initialize components
        parameters = params_class()
        variables = vars_class(parameters=parameters)
        scenarios = scenarios_class(parameters=parameters)

        # Create model instance
        model = model_class(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
            # behavior=behavior_class,
        )

        return model, parameters, variables, scenarios

    @pytest.mark.parametrize(
        "model_components", discover_model_components(), indirect=True
    )
    def test_model_initialization(self, model_components):
        """Test that model and its components initialize correctly"""
        model, parameters, variables, scenarios = model_components

        assert model is not None, "Model should initialize"
        assert parameters is not None, "Parameters should initialize"
        assert variables is not None, "Variables should initialize"
        assert scenarios is not None, "Scenarios should initialize"

        assert hasattr(model, "parameters"), "Model should have parameters attribute"
        assert hasattr(model, "variables"), "Model should have variables attribute"
        assert hasattr(model, "scenarios"), "Model should have scenarios attribute"
        assert hasattr(model, "behavior"), "Model should have behavior attribute"

    @pytest.mark.parametrize(
        "model_components", discover_model_components(), indirect=True
    )
    def test_simulation_method(self, model_components):
        """Test that model can simulate"""
        model, _, _, _ = model_components
        assert hasattr(model, "simulate"), "Model should have simulate method"
        assert callable(model.simulate), "simulate should be callable"

    @pytest.mark.parametrize(
        "model_components", discover_model_components(), indirect=True
    )
    def test_parameters_structure(self, model_components):
        """Test parameter class structure"""
        _, parameters, _, _ = model_components
        assert hasattr(
            parameters, "get_default_parameters"
        ), "Parameters should have get_default_parameters method"
        assert callable(
            parameters.get_default_parameters
        ), "get_default_parameters should be callable"

        default_params = parameters.get_default_parameters()
        assert isinstance(
            default_params, dict
        ), "Default parameters should be a dictionary"

    @pytest.mark.parametrize(
        "model_components", discover_model_components(), indirect=True
    )
    def test_variables_structure(self, model_components):
        """Test variables class structure"""
        _, _, variables, _ = model_components
        assert hasattr(
            variables, "get_default_variables"
        ), "Variables should have get_default_variables method"
        assert callable(
            variables.get_default_variables
        ), "get_default_variables should be callable"

        default_vars = variables.get_default_variables()
        assert isinstance(
            default_vars, dict
        ), "Default variables should be a dictionary"

    @pytest.mark.parametrize(
        "model_components", discover_model_components(), indirect=True
    )
    def test_scenarios_structure(self, model_components):
        """Test scenarios class structure"""
        _, _, _, scenarios = model_components
        assert hasattr(
            scenarios, "get_default_scenario_values"
        ), "Scenarios should have get_default_scenario_values method"
        assert callable(
            scenarios.get_default_scenario_values
        ), "get_default_scenario_values should be callable"

        default_scenarios = scenarios.get_default_scenario_values()
        assert isinstance(
            default_scenarios, dict
        ), "Default scenarios should be a dictionary"

    @pytest.mark.parametrize(
        "model_components", discover_model_components(), indirect=True
    )
    def test_serialization(self, model_components):
        """Test model serialization capabilities"""
        model, _, _, _ = model_components
        assert hasattr(model, "to_json"), "Model should have to_json method"
        assert callable(model.to_json), "to_json should be callable"
        assert hasattr(model, "save"), "Model should have save method"
        assert callable(model.save), "save should be callable"
