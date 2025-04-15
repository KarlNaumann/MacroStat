"""Model Manager

This module provides a manager for all models in the MacroStat library. It allows for easy access to all models, as well as the ability to get the classes for a specific model.
"""

import os
from typing import NamedTuple, Type

from macrostat.core.behavior import Behavior
from macrostat.core.model import Model
from macrostat.core.parameters import Parameters
from macrostat.core.scenarios import Scenarios
from macrostat.core.variables import Variables


class ModelClasses(NamedTuple):
    """Container for all model component classes of a specific version."""

    Behavior: Type[Behavior]
    Parameters: Type[Parameters]
    Scenarios: Type[Scenarios]
    Variables: Type[Variables]
    Model: Type[Model]


def get_model(modelname: str):
    """Get a model from the models directory.

    Parameters
    ----------
    modelname: str
        The name of the model to get.

    Returns
    -------
    ModelClasses
        Named tuple containing all model component classes

    Raises
    ------
    ValueError
        If model name is invalid or model is not available
    ImportError
        If there are problems importing the model components
    """
    available = get_available_models()
    if modelname not in available:
        raise ValueError(
            f"Invalid or unavailable model: {modelname}\n"
            f"Available models: {', '.join(available)}"
        )

    try:
        base_path = f"macrostat.models.{modelname}"

        # Import the model class
        module = __import__(
            f"{base_path}.{modelname.lower()}", fromlist=[modelname.lower()]
        )
        return getattr(module, modelname)

    except ImportError as e:
        raise ImportError(f"Could not import model {modelname}: {str(e)}")


def get_model_classes(modelname: str):
    """Get a model from the models directory.

    Parameters
    ----------
    modelname: str
        The name of the model to get.

    Returns
    -------
    ModelClasses
        Named tuple containing all model component classes

    Raises
    ------
    ValueError
        If model name is invalid or model is not available
    ImportError
        If there are problems importing the model components
    """
    available = get_available_models()
    if modelname not in available:
        raise ValueError(
            f"Invalid or unavailable model: {modelname}\n"
            f"Available models: {', '.join(available)}"
        )

    try:
        base_path = f"macrostat.models.{modelname}"
        components = {}

        # Import the model class
        module = __import__(
            f"{base_path}.{modelname.lower()}", fromlist=[modelname.lower()]
        )
        components["Model"] = getattr(module, modelname)

        # Import all components
        for component in ["behavior", "parameters", "variables", "scenarios"]:
            name = f"{component.capitalize()}{modelname.upper()}"
            module = __import__(f"{base_path}.{component}", fromlist=[name])
            component_class = getattr(module, name)
            components[component.capitalize()] = component_class

        return ModelClasses(
            Model=components["Model"],
            Behavior=components["Behavior"],
            Parameters=components["Parameters"],
            Variables=components["Variables"],
            Scenarios=components["Scenarios"],
        )

    except ImportError as e:
        raise ImportError(f"Could not import model {modelname}: {str(e)}")


def get_available_models():
    """Get all available models in the models directory.

    Parse the models directory and return a list of all available models,
    which are in subdirectories of the models directory and are named after the model.
    They are valid when they contain a __init__.py, parameters.py, variables.py,
    scenarios.py, behavior.py file.

    Returns
    -------
    list
        A list of all available models.
    """
    models = []
    for file in os.listdir(os.path.dirname(__file__)):
        path = os.path.join(os.path.dirname(__file__), file)
        if os.path.isdir(path):
            # Check if the file is a valid model (has all the required files)
            files = [
                os.path.join(path, i)
                for i in [
                    "__init__.py",
                    "parameters.py",
                    "variables.py",
                    "scenarios.py",
                    "behavior.py",
                    f"{file}.py",
                ]
            ]
            if all(os.path.exists(f) for f in files):
                models.append(file)

    return models
