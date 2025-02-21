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
    """A class representing a macroeconomic model.

    This class provides a wrapper for users to write their underlying model
    behavior while maintaining a uniformly accessible interface. Specifically,
    the user is expected to adapt the model.simulate() function to their needs,
    respecting only that the return of that function is a pandas dataframe.

    Attributes
    ----------
    name: str
        Name of the model, such as "model". Used for file and database names
    parameters : dict
        Dictionary of all parameters
    hyper_parameters : dict
        Dictionary of all hyperparameters
    output : pd.DataFrame
        None, or the latest simulation run for given parameters


    Example
    -------
    A general workflow for a model might look like

    >>> model = Model(parameters, hyper_parameters)
    >>> output = model.simulate()
    >>> model.save()

    """

    def __init__(
        self,
        parameters: Parameters | dict,
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
        parameters: Parameters | dict
            The parameters of the model.
        scenarios: Scenarios | dict
            The scenarios of the model.
        variables: Variables | dict
            The variables of the model.
        debug: bool
            Whether to print debug information.
        """
        # Essential attributes
        if not isinstance(parameters, Parameters):
            self.parameters = Parameters(
                parameters=parameters, hyperparameters=hyperparameters
            )
        else:
            self.parameters = parameters

        if not isinstance(scenarios, Scenarios):
            self.scenarios = Scenarios(parameters=self.parameters, scenarios=scenarios)
        else:
            self.scenarios = scenarios

        if not isinstance(variables, Variables):
            self.variables = Variables(parameters=self.parameters, variables=variables)
        else:
            self.variables = variables

        self.behavior = behavior
        self.name = name

        logging.basicConfig(level=log_level, filename=log_file)

    def forward(self, scenario: int = 0, *args, **kwargs):
        """Forward pass of the model."""
        behavior = self.behavior(
            self.parameters,
            self.scenarios,
            self.variables,
            record=False,
            scenario=scenario,
        )
        return behavior.forward(*args, **kwargs)

    @classmethod
    def from_json(cls, parameter_file: str, scenario_file: str, *args, **kwargs):
        """Initialize the model from a JSON file."""
        parameters = Parameters.from_json(parameter_file)
        scenarios = Scenarios.from_json(scenario_file, parameters=parameters)
        return cls(parameters, scenarios)

    @classmethod
    def load(cls, path=None):
        """Class method to load a model instance from a pickled file.

        Parameters
        ----------
        path, optional
            path to the targeted Sampler

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

    def simulate(self, scenario: int = 0, *args, **kwargs):
        """Simulate the model."""
        logging.info(f"Starting simulation. Scenario: {scenario}")
        behavior = self.behavior(
            self.parameters,
            self.scenarios,
            self.variables,
            record=True,
            scenario=scenario,
        )
        with torch.no_grad():
            return behavior.forward(*args, **kwargs)

    def to_json(self, file_path: os.PathLike, *args, **kwargs):
        """Convert the model to a JSON file."""
        self.parameters.to_json(f"{file_path}_params.json")
        self.scenarios.to_json(f"{file_path}_scenarios.json")
