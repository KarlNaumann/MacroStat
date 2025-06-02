# -*- coding: utf-8 -*-
"""
Generic model class as a wrapper to specific implementations
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__version__ = "0.1.0"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging
import os
import pickle

import torch

from macrostat.core.behavior import Behavior
from macrostat.core.parameters import Parameters
from macrostat.core.scenarios import Scenarios
from macrostat.core.variables import Variables

logger = logging.getLogger(__name__)


class Model:
    """A general class to represent a macroeconomic model.

    This class provides a wrapper for users to write their underlying model
    behavior while maintaining a uniformly accessible interface.

    Attributes
    ----------
    parameters : macrostat.core.parameters.Parameters
        The parameters of the model.
    scenarios : macrostat.core.scenarios.Scenarios
        The scenarios of the model.
    variables : macrostat.core.variables.Variables
        The variables of the model.
    behavior : macrostat.core.behavior.Behavior
        The behavior class of the model.
    name : str
        The name of the model.

    Example
    -------
    A general workflow for a model might look like:

    >>> model = Model()
    >>> output = model.simulate()
    >>> model.save()

    """

    def __init__(
        self,
        parameters: Parameters | dict | None = None,
        hyperparameters: dict | None = None,
        scenarios: Scenarios | dict = None,
        variables: Variables | dict = None,
        behavior: Behavior = Behavior,
        name: str = "model",
        log_level: int = logging.INFO,
        log_file: str = "macrostat_model.log",
    ):
        """Initialization of the model class.


        Parameters
        ----------
        parameters: macrostat.core.parameters.Parameters | dict
            The parameters of the model.
        hyperparameters: dict (optional)
            The hyperparameters of the model.
        scenarios: macrostat.core.scenarios.Scenarios | dict (optional)
            The scenarios of the model.
        variables: macrostat.core.variables.Variables | dict (optional)
            The variables of the model.
        behavior: macrostat.core.behavior.Behavior (optional)
            The behavior of the model.
        name: str (optional)
            The name of the model.
        log_level: int (optional)
            The log level, defaults to logging.INFO but can be set to logging.DEBUG
            for more verbose output.
        log_file: str (optional)
            The log file, defaults to "macrostat_model.log" in the current working
            directory.
        """
        # Essential attributes
        if isinstance(parameters, dict):
            self.parameters = Parameters(
                parameters=parameters, hyperparameters=hyperparameters
            )
        elif isinstance(parameters, Parameters):
            self.parameters = parameters
            if hyperparameters is not None:
                self.parameters.hyper.update(hyperparameters)
        else:
            logger.warning("No parameters provided, using default parameters")
            self.parameters = Parameters()

        if isinstance(scenarios, Scenarios):
            self.scenarios = scenarios
        else:
            logger.warning("No scenarios provided, using default scenarios")
            self.scenarios = Scenarios(parameters=self.parameters)

        if isinstance(variables, Variables):
            self.variables = variables
        else:
            logger.warning("No variables provided, using default variables")
            self.variables = Variables(parameters=self.parameters)

        if behavior is not None and issubclass(behavior, Behavior):
            self.behavior = behavior
        else:
            logger.warning("No behavior provided, using default behavior")
            self.behavior = Behavior

        self.name = name

        logging.basicConfig(level=log_level, filename=log_file)

    @classmethod
    def from_json(
        cls,
        parameter_file: str,
        scenario_file: str,
        variable_file: str,
        *args,
        **kwargs,
    ):
        """Initialize the model from a JSON file."""
        parameters = Parameters.from_json(parameter_file)
        scenarios = Scenarios.from_json(scenario_file, parameters=parameters)
        variables = Variables.from_json(variable_file, parameters=parameters)
        return cls(parameters=parameters, scenarios=scenarios, variables=variables)

    @classmethod
    def load(cls, path: os.PathLike):
        """Class method to load a model instance from a pickled file.

        Parameters
        ----------
        path: os.PathLike
            path to the targeted file containing the model.

        Notes
        -----
        .. note:: This implementation is dependent on your pickling version

        """
        with open(path, "rb") as f:
            model = pickle.load(f)
        return model

    def save(self, path: os.PathLike):
        """Save the model object as a pickled file

        Parameters
        ----------
        path: os.PathLike
            path where the model will be stored. If it is None then
            the model's name will be used and the file stored in the
            working directory.

        Notes
        -----
        .. note:: This implementation is dependent on your pickling version
        """
        with open(path, "wb") as f:
            pickle.dump(self, f)

    def simulate(self, scenario: int | str = 0, *args, **kwargs):
        """Simulate the model.

        Parameters
        ----------
        scenario: int (optional)
            The scenario to use for the model run, defaults to 0, which
            represents the default scenario (no shocks).
        """
        if isinstance(scenario, str):
            scenario = self.scenarios.get_scenario_index(scenario)

        logging.debug(f"Starting simulation. Scenario: {scenario}")
        behavior = self.behavior(
            self.parameters,
            self.scenarios,
            self.variables,
            scenario=scenario,
            *args,
            **kwargs,
        )
        behavior = behavior.to(self.parameters["device"])
        with torch.no_grad():
            return behavior.forward(*args, **kwargs)

    def compute_theoretical_steady_state(
        self, scenario: int | str = 0, *args, **kwargs
    ):
        """Compute the theoretical steady state of the model.

        This process generally follows the structure of the forward() function,
        but instead of simulating the model, the steady state is computed at
        each timestep. Therefore, (1) the model is initialized, and (2) for
        each timestep the parameter and scenario information is passed to the
        compute_theoretical_steady_state_per_step() function that computes the
        steady state at that timestep.

        Parameters
        ----------
        scenario: int (optional)
            The scenario to use for the model run, defaults to 0, which
            represents the default scenario (no shocks).
        """
        if isinstance(scenario, str):
            scenario = self.scenarios.get_scenario_index(scenario)

        logging.info(f"Computing theoretical steady state. Scenario: {scenario}")
        behavior = self.behavior(
            self.parameters,
            self.scenarios,
            self.variables,
            scenario=scenario,
            *args,
            **kwargs,
        )
        with torch.no_grad():
            return behavior.compute_theoretical_steady_state(*args, **kwargs)

    def to_json(self, file_path: os.PathLike, *args, **kwargs):
        """Convert the model to a JSON file split into parameters, scenarios,
        and variables.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the file to save the model to.
        """
        self.parameters.to_json(f"{file_path}_params.json")
        self.scenarios.to_json(f"{file_path}_scenarios.json")
        self.variables.to_json(f"{file_path}_variables.json")
