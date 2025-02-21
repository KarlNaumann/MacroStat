"""
A class for handling parameters for a MacroStat model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

# Default libraries
import json
import logging
import os

# Third-party libraries
import torch

logger = logging.getLogger(__name__)


class BoundaryError(Exception):
    """Exception raised for invalid bounds."""

    def __init__(self, message: str):
        self.message = message + " Please check the Excel, JSON or default bounds."


class Parameters:
    """A class for handling parameters for the MacroStat model."""

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        bounds: dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the parameters for the model. If no parameters are provided,
        the default parameters will be used, and if only some parameters are
        provided, the missing parameters will be set to their default values.

        Parameters
        ----------
        parameters: dict | None
            The parameters to initialize the model with. If None, the default
            parameters will be used.
        hyperparameters: dict | None
            The hyperparameters to initialize the model with. If None, the
            default hyperparameters will be used.
        bounds: dict | None
            The bounds to initialize the model with. If None, the default bounds
            will be used
        """

        self.values = self.get_default_parameters()
        if parameters is not None:
            self.values.update(parameters)

        self.hyper = self.get_default_hyperparameters()
        if hyperparameters is not None:
            self.hyper.update(hyperparameters)

        self.bounds = self.get_default_bounds()
        if bounds is not None:
            self.bounds.update(bounds)

        self.verify_bounds()
        self.verify_parameters()

    def __contains__(self, key: str):
        """Check if a key is in the parameters or hyperparameters.

        Parameters
        ----------
        key: str
            The key to check for.
        """
        return key in self.values or key in self.hyper

    def __getitem__(self, key: str):
        """Get an item from the parameters or hyperparameters.

        Parameters
        ----------
        key: str
            The key to get the item for.
        """
        return self.values[key] if key in self.values else self.hyper[key]

    def __setitem__(self, key: str, value: float):
        """Set an item in the parameters or hyperparameters.

        Parameters
        ----------
        key: str
            The key to set the item for.
        value: float
            The value to set for the item.
        """
        if key in self.values:
            self.values[key] = value
        elif key in self.hyper:
            try:
                self.hyper[key] = int(value)
            except Exception:
                self.hyper[key] = value
        else:
            logger.warning(f"Key {key} not found in parameters or hyperparameters.")

    def __str__(self):
        """Return a string representation of the parameters.

        This function returns a string representation of the parameters,
        with the hyperparameters and parameters aligned.
        """
        # Find the longest key for alignment
        hyper_max_len = max(len(key) for key in self.hyper.keys())
        param_max_len = max(len(key) for key in self.values.keys())
        max_key_length = max(hyper_max_len, param_max_len)

        # Create the output string, hyperparameters first
        output = "Hyperparameters:\n"
        for key, value in self.hyper.items():
            output += f"  {key:.<{max_key_length}} {value}\n"

        # Add the parameters
        output += "\nParameters:\n"
        for key, value in self.values.items():
            output += f"  {key:.<{max_key_length}} {value:.5g}\n"

        return output

    @classmethod
    def from_json(cls, file_path: os.PathLike, *args, **kwargs):
        """Initialize the parameters from a JSON file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the JSON file.
        """
        with open(file_path, "r") as file:
            data = json.load(file)

        # Convert bounds to tuple
        for key, value in data["Bounds"].items():
            data["Bounds"][key] = tuple(value)

        return cls(
            parameters=data["Parameters"],
            hyperparameters=data["HyperParameters"],
            bounds=data["Bounds"],
        )

    @classmethod
    def from_excel(cls, file_path: os.PathLike, *args, **kwargs):
        """Initialize the parameters from an Excel file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the Excel file to load the parameters from.
        """
        raise NotImplementedError("Not implemented")

    def get_default_bounds(self):
        """Return the default bounds."""
        return {}

    def get_default_hyperparameters(self):
        """Return the default hyperparameters.

        The hyperparameters are the parameters that are not directly used in
        the model, but rather for the simulation and calibration. They:
        1. Must include the number of timesteps to simulate
        2. Must include the scenario trigger, i.e. the timestep at which the
           scenario starts
        3. Must include the seed for the random number generator
        4. Must include the device to use for the simulation
        5. May include other parameters, such as flags for the model
        """

        return {
            "timesteps": 100,
            "timesteps_initialization": 10,
            "scenario_trigger": 0,
            "seed": 42,
            "device": "cpu",
            "requires_grad": False,
        }

    def get_default_parameters(self):
        """Return the default parameters."""
        return {}

    def set_bounds(self, bounds: dict):
        """Set the bounds for the parameters.

        Parameters
        ----------
        bounds: dict
            The bounds to set for the parameters.
        """
        self.bounds.update(bounds)
        self.verify_bounds()

    def set_bound(self, key: str, value: tuple):
        """Set the bounds for a single parameter

        Parameters
        ----------
        key: str
            The key of the parameter to set the bounds for.
        value: tuple
            The bounds to set for the parameter.
        """
        self.bounds[key] = value
        self.verify_bounds()

    def to_excel(self, file_path: os.PathLike, *args, **kwargs):
        """Convert the parameters to an Excel file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the Excel file to save the parameters to.
        """
        raise NotImplementedError("Not implemented")

    def to_json(self, file_path: os.PathLike, *args, **kwargs):
        """Convert the parameters to a JSON file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the JSON file to save the parameters to.
        """
        with open(file_path, "w") as file:
            json.dump(
                {
                    "Parameters": self.values,
                    "HyperParameters": self.hyper,
                    "Bounds": self.bounds,
                },
                file,
            )

    def to_nn_parameters(self):
        """Convert the parameters to a nn.ParameterDict."""
        vectorized = self.vectorize_parameters()
        return torch.nn.ParameterDict(vectorized)

    def vectorize_parameters(self):
        """Vectorize the parameters."""
        self.values_vectorized = self.values
        return self.values_vectorized

    def verify_bounds(self):
        """Verify that the bounds are valid. By testing first that all
        parameters have bounds, and then that the bounds are valid.
        """
        # Check that all parameters have bounds
        needed_bounds = set(self.get_default_bounds().keys())
        found_bounds = set(self.bounds.keys())
        if needed_bounds.difference(found_bounds):
            raise BoundaryError(
                f"Missing bounds for parameters: {needed_bounds - found_bounds}"
            )

        # Check that the bounds are valid
        for param, bounds in self.bounds.items():
            if bounds[0] > bounds[1]:
                raise BoundaryError(f"Parameter {param} has invalid bounds: {bounds}")

    def verify_parameters(self):
        """Verify that the parameters are within the bounds."""
        for param, bounds in self.bounds.items():
            value = self.values[param]
            if value < bounds[0] or value > bounds[1]:
                msg = f"Parameter {param} has invalid value: {value}"
                raise BoundaryError(f"{msg} (bounds: {self.bounds[param]})")


if __name__ == "__main__":
    pass
