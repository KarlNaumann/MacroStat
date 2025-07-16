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
import pandas as pd
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

        self.verify_bounds()
        self.verify_parameters()

    def __contains__(self, key: str):
        """Check if a key is in the parameters or hyperparameters.

        Parameters
        ----------
        key: str
            The name of the parameter or hyperparameter to check for.
        """
        return key in self.values or key in self.hyper

    def __getitem__(self, key: str):
        """Get an item from the parameters or hyperparameters.

        Parameters
        ----------
        key: str
            The name of the parameter or hyperparameter to get the item for.
        """
        return self.values[key]["value"] if key in self.values else self.hyper[key]

    def __setitem__(self, key: str, value: float):
        """Set an item in the parameters or hyperparameters.

        Parameters
        ----------
        key: str
            The name of the parameter or hyperparameter to set.
        value: float
            The value to set for the item.
        """
        if key in self.values:
            self.values[key]["value"] = value
        elif key in self.hyper:
            self.hyper[key] = value
        else:
            logger.warning(f"Key {key} not found in parameters or hyperparameters.")

    def __str__(self):  # pragma: no cover
        """Return a string representation of the parameters.

        This function returns a string representation of the parameters,
        with the hyperparameters and parameters aligned.
        """
        # Find the longest key for alignment
        hyper_max_len = max([len(key) for key in self.hyper.keys()] + [10])
        param_max_len = max([len(key) for key in self.values.keys()] + [10])
        max_key_length = max(hyper_max_len, param_max_len)

        # Create the output string, hyperparameters first
        output = "Hyperparameters:\n"
        for key, value in self.hyper.items():
            output += f"  {key:.<{max_key_length}} {value}\n"

        # Add the parameters
        output += "\nParameters:\n"
        for key, info in self.values.items():
            output += (
                f"  {key:.<{max_key_length}} {info['value']:.5g} ({info['unit']})\n"
            )

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

        return cls(
            parameters=data["Parameters"],
            hyperparameters=data["HyperParameters"],
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

    @classmethod
    def from_csv(cls, file_path: os.PathLike, *args, **kwargs):
        """Initialize the parameters from a CSV file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the CSV file to load the parameters from.
        """
        df = pd.read_csv(file_path, index_col=0)

        # Parameters
        par = df[df["ParameterType"] == "Parameter"].drop(columns=["ParameterType"])
        par.columns = [i.lower() for i in par.columns]
        par["value"] = par["value"].astype(float)
        par = par.to_dict(orient="index")

        # Hyperparameters
        hyper = df[df["ParameterType"] == "HyperParameter"]["Value"]
        hyper.index = hyper.index.str.lower()
        hyper = hyper.to_dict()
        # Try to convert to int, else bool, else keep as string
        for key, value in hyper.items():
            if value.isdigit():
                hyper[key] = int(value)
            elif value.lower() == "true":
                hyper[key] = True
            elif value.lower() == "false":
                hyper[key] = False
            else:
                hyper[key] = value

        # Convert the dataframe to a dictionary
        return cls(
            parameters=par,
            hyperparameters=hyper,
        )

    def get_default_hyperparameters(self):
        """Return the default hyperparameters.

        The hyperparameters are the parameters that are not directly used in
        the model, but rather for the simulation and calibration. They must
        include:
        1. The number of timesteps to simulate
        2. The scenario trigger, i.e. the timestep at which the scenario starts
        3. The seed for the random number generator
        4. The device to use for the simulation
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
        """Return the default parameters.

        Should return a dictionary with the keys being the parameter names,
        and a subdictionary with the keys including:
        - "Value": The value of the parameter.
        - "Lower": The lower bound of the parameter.
        - "Upper": The upper bound of the parameter.
        - "Unit": The unit of the parameter.
        - "Notation": The notation of the parameter.
        """
        return {}

    def get_bounds(self):
        """Return the bounds for the parameters."""
        return {
            key: (info["lower bound"], info["upper bound"])
            for key, info in self.values.items()
        }

    def get_values(self):
        """Return the values for the parameters."""
        return {key: info["value"] for key, info in self.values.items()}

    def is_equal(self, other: "Parameters"):
        """Compare the parameters to another Parameters object."""
        return self.values == other.values and self.hyper == other.hyper

    def set_bound(self, key: str, value: tuple):
        """Set the bounds for a single parameter

        Parameters
        ----------
        key: str
            The key of the parameter to set the bounds for.
        value: tuple
            The bounds to set for the parameter.
        """
        self.values[key]["Lower Bound"] = value[0]
        self.values[key]["Upper Bound"] = value[1]
        self.verify_bounds()

    def set_notation(self, key: str, value: str):
        """Set the notation for a single parameter."""
        self.values[key]["notation"] = value

    def set_unit(self, key: str, value: str):
        """Set the unit for a single parameter."""
        self.values[key]["unit"] = value

    def to_csv(self, file_path: os.PathLike, sphinx_math: bool = False):
        """Convert the parameters to a CSV file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the CSV file to save the parameters to.
        sphinx_math: bool
            Whether to use Sphinx math notation in the CSV file.
        """
        par = pd.DataFrame.from_dict(self.values, orient="index").sort_index()
        par = par[["notation", "unit", "value", "lower bound", "upper bound"]]
        par.columns = ["Notation", "Unit", "Value", "Lower Bound", "Upper Bound"]
        if sphinx_math:  # pragma: no cover
            par["Notation"] = par["Notation"].apply(lambda x: r":math:`" + x + r"`")
        par["ParameterType"] = "Parameter"

        hyper = pd.DataFrame.from_dict(self.hyper, orient="index").sort_index()
        hyper.columns = ["Value"]
        hyper["ParameterType"] = "HyperParameter"

        df = pd.concat([par, hyper], axis=0)
        df.to_csv(file_path)
        return df

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
                },
                file,
            )

    def to_nn_parameters(self):
        """Convert the parameters to a nn.ParameterDict."""
        vectorized = self.vectorize_parameters()
        return torch.nn.ParameterDict(vectorized)

    def vectorize_parameters(self):
        """Vectorize the parameters."""
        pvectors = {}
        for key, info in self.values.items():
            pvectors[key.replace(".", "_")] = torch.tensor(
                info["value"], device=self.hyper["device"], dtype=torch.float
            )
        return pvectors

    def verify_bounds(self):
        """Verify that the bounds are valid. By testing first that all
        parameters have bounds, and then that the bounds are valid.
        """
        # Check that all parameters have bounds
        needed_bounds = set(self.get_default_parameters().keys())
        found_bounds = {}
        for key, info in self.values.items():
            conditions = [
                "lower bound" in info and info["lower bound"] is not None,
                "upper bound" in info and info["upper bound"] is not None,
            ]
            if all(conditions):
                found_bounds[key] = (info["lower bound"], info["upper bound"])

        found_bound_params = set(found_bounds.keys())
        if needed_bounds.difference(found_bound_params):
            raise BoundaryError(
                f"Missing bounds for parameters: {needed_bounds - found_bound_params}"
            )

        # Check that the bounds are valid
        for param, bounds in found_bounds.items():
            if bounds[0] > bounds[1]:
                raise BoundaryError(f"Parameter {param} has invalid bounds: {bounds}")

    def verify_parameters(self):
        """Verify that the parameters are within the bounds."""
        for param, info in self.values.items():
            if (
                info["value"] < info["lower bound"]
                or info["value"] > info["upper bound"]
            ):
                msg = f"Parameter {param} has invalid value: {info['value']}"
                raise BoundaryError(
                    f"{msg} (bounds: {info['lower bound']}, {info['upper bound']})"
                )


if __name__ == "__main__":
    pass
