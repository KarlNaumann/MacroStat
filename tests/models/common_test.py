import importlib
import inspect
from pathlib import Path
from typing import List, Tuple, Type

import numpy
import pytest
import torch

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
    # Get the project root directory (where tox.ini is located)
    project_root = Path(__file__).parent.parent.parent
    models_dir = project_root / "src" / "macrostat" / "models"
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

    def setup_method(self):
        """Setup method to create model instance and its components"""
        # Initialize components
        parameters = self.params_class()
        variables = self.vars_class(parameters=parameters)
        scenarios = self.scenarios_class(parameters=parameters)

        # Create model instance
        self.model = self.model_class(
            parameters=parameters,
            variables=variables,
            scenarios=scenarios,
        )
        self.model_default = self.model_class(parameters=None)

        self.classes = {
            "model": self.model_class,
            "parameters": self.params_class,
            "behavior": self.behavior_class,
            "variables": self.vars_class,
            "scenarios": self.scenarios_class,
        }

        var_info = set(
            [
                j
                for i in list(variables.get_default_variables().values())
                for j in i.keys()
            ]
        )
        self.isSFC = "sfc" in var_info

    ##############################################################
    # Tests on the model level
    ##############################################################

    def test_model_exists(self):
        """Test that model initializes"""
        assert self.model is not None, "Model should initialize"

    def test_parameters_initialization(self):
        """Test parameters initialization: they exist and are of the correct type"""
        assert hasattr(self.model, "parameters")
        assert isinstance(self.model.parameters, self.classes["parameters"])
        assert isinstance(self.model_default.parameters, self.classes["parameters"])

    def test_variables_initialization(self):
        """Test variables initialization: they exist and are of the correct type"""
        assert hasattr(self.model, "variables")
        assert isinstance(self.model.variables, self.classes["variables"])
        assert isinstance(self.model_default.variables, self.classes["variables"])

    def test_scenarios_initialization(self):
        """Test scenarios initialization: they exist and are of the correct type"""
        assert hasattr(self.model, "scenarios")
        assert isinstance(self.model.scenarios, self.classes["scenarios"])
        assert isinstance(self.model_default.scenarios, self.classes["scenarios"])

    def test_behavior_initialization(self):
        """Test behavior initialization: it exists and is of the correct type"""
        assert hasattr(self.model, "behavior")
        assert self.model.behavior == self.classes["behavior"]
        assert self.model_default.behavior == self.classes["behavior"]

    @pytest.mark.slow
    def test_default_simulation_is_healthy(self):
        """Test that default simulation is healthy"""
        self.model.simulate()
        assert self.model.variables.check_health()

    @pytest.mark.slow
    def test_theoretical_steadystate_is_healthy(self):
        """Test that theoretical steadystate is healthy"""
        try:
            self.model.compute_theoretical_steady_state()
        except NotImplementedError:
            pass
        else:
            assert self.model.variables.check_health()

    ##############################################################
    # Tests on the parameters level
    ##############################################################

    def test_default_parameters_nonempty(self):
        """Test that default parameters are not empty"""
        assert self.model.parameters.get_default_parameters() is not None
        assert len(self.model.parameters.get_default_parameters()) > 0

    def test_default_parameters_have_value_information(self):
        """Test that default parameters have value information"""
        default_params = self.model.parameters.get_default_parameters()
        for param in default_params:
            assert "value" in default_params[param]
            assert isinstance(
                default_params[param]["value"],
                (int, float, list, tuple, numpy.ndarray, torch.Tensor),
            ), f"Parameter {param} value must be numeric/array/tensor. Otherwise it should be a hyperparameter."

    def test_default_hyperparameters_have_minimal_information(self):
        """Test that default hyperparameters have minimal information, i.e. they
        contain at least the information that is in the core.parameters.Parameters class
        method get_default_hyperparameters"""
        model_default_hp = self.model_default.parameters.get_default_hyperparameters()
        parameters_default_hp = Parameters().get_default_hyperparameters()
        assert model_default_hp is not None
        assert len(model_default_hp) > 0
        # Check that all keys in parameters_default_hp are in model_default_hp
        for key in parameters_default_hp:
            assert (
                key in model_default_hp
            ), f"Model default hyperparameters missing required key: {key}"

    ##############################################################
    # Tests on the variables level
    ##############################################################

    def test_variables_default_initialization(self):
        """Test that default variables are not empty"""
        default_vars = self.vars_class(parameters=None)
        assert default_vars is not None
        assert isinstance(default_vars.parameters, self.classes["parameters"])

    def test_default_variables_nonempty(self):
        """Test that default variables are not empty"""
        assert self.model.variables.get_default_variables() is not None
        assert len(self.model.variables.get_default_variables()) > 0

    def test_default_variables_have_history_information(self):
        """Test that default variables have history information"""
        default_vars = self.model.variables.get_default_variables()
        for var in default_vars:
            assert "history" in default_vars[var]
            assert isinstance(default_vars[var]["history"], int)
            assert default_vars[var]["history"] >= 0

    def test_default_variables_have_sfc_information(self):
        """Test that default variables have sfc information"""
        default_vars = self.model.variables.get_default_variables()
        if self.isSFC:
            for var in default_vars:
                assert "sfc" in default_vars[var]
                assert isinstance(default_vars[var]["sfc"], list)
                assert len(default_vars[var]["sfc"]) > 0

    def test_default_variables_sfc_info_valid(self):
        """Test that default variables have valid sfc information"""
        if self.isSFC:
            assert self.model.variables.verify_sfc_info()

    ##############################################################
    # Tests on the scenarios level
    # Since there can be none or many, we can't test the default
    # values of the scenarios class
    ##############################################################

    def test_scenarios_default_initialization(self):
        """Test that default scenarios are not empty"""
        default_scenarios = self.scenarios_class(parameters=None)
        assert default_scenarios is not None
        assert isinstance(default_scenarios.parameters, self.classes["parameters"])

    ##############################################################
    # Tests on the behavior level
    ##############################################################

    def test_behavior_default_initialization(self):
        """Test that default behavior is not empty"""
        default_behavior = self.behavior_class()
        assert default_behavior is not None
        assert isinstance(default_behavior.params, torch.nn.ParameterDict)
        assert isinstance(default_behavior.scenarios, torch.nn.ParameterDict)
        assert isinstance(default_behavior.variables, self.classes["variables"])

    def test_behavior_default_initialization_with_kwargs(self):
        """Test that default behavior is not empty"""
        default_params = self.params_class()
        default_scenarios = self.scenarios_class()
        default_variables = self.vars_class()
        default_behavior = self.behavior_class(
            parameters=default_params,
            scenarios=default_scenarios,
            variables=default_variables,
        )
        assert default_behavior is not None
        assert isinstance(default_behavior.params, torch.nn.ParameterDict)
        assert isinstance(default_behavior.scenarios, torch.nn.ParameterDict)
        assert isinstance(default_behavior.variables, self.classes["variables"])

    def test_behavior_initialize_method_implemented(self):
        """Test that the initialize method is implemented"""
        # Test that it exists
        assert hasattr(self.behavior_class, "initialize")
        assert callable(self.behavior_class.initialize)

        # Test that it doesn't raise a NotImplementedError
        behavior = self.behavior_class()
        # Get the source code of the initialize method
        initialize_source = inspect.getsource(behavior.initialize)
        # Check that it's not just raising NotImplementedError
        if "raise NotImplementedError" in initialize_source:
            raise AssertionError(
                "initialize method should not raise a NotImplementedError"
            )

    def test_behavior_step_method_implemented(self):
        """Test that the step method is implemented"""
        # Test that it exists
        assert hasattr(self.behavior_class, "step")
        assert callable(self.behavior_class.step)

        # Test that it doesn't raise a NotImplementedError
        behavior = self.behavior_class()
        # Get the source code of the step method
        step_source = inspect.getsource(behavior.step)
        # Check that it's not just raising NotImplementedError
        if "raise NotImplementedError" in step_source:
            raise AssertionError("step method should not raise a NotImplementedError")


# Dynamically create test classes for each model
for (
    model_class,
    params_class,
    behavior_class,
    vars_class,
    scenarios_class,
) in discover_model_components():
    # Create a unique class name based on the model name
    model_name = model_class.__name__
    test_class_name = f"Test{model_name}"

    # Create the test class dynamically
    test_class = type(
        test_class_name,
        (BaseModelTest,),
        {
            "__module__": __name__,
            "model_class": model_class,
            "params_class": params_class,
            "behavior_class": behavior_class,
            "vars_class": vars_class,
            "scenarios_class": scenarios_class,
        },
    )

    # Add the class to the current module's namespace
    globals()[test_class_name] = test_class
