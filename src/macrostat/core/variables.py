"""
A class for handling variables for a MacroStat model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import json
import logging
import os
from typing import Self

import pandas as pd
import torch

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class Variables:
    """Variables class for the MacroStat model.

    This class contains the variables of a MacroStat model, specifically the
    output tensors from the simulation. Furthermore, it contains the methods
    to export the variables to different formats, and holds important information
    on the characteristics of each of the variables, such as their dimension,
    long-form name, unit, description and notation.
    """

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: Parameters | dict | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables for the model. If no variables are provided,
        the default variables will be used, and if only some variables are
        provided, the missing variables will be set to their default values.

        Parameters
        ----------
        variable_info: dict | None
            The variable information to use for the model.
        timeseries: dict | None
            The timeseries to use for the model.
        parameters: dict | None
            The parameters to use for the model.
        """
        if parameters is not None:
            self.parameters = parameters
        else:
            self.parameters = Parameters()

        if variable_info is None:
            self.info = self.get_default_variables()
        else:
            self.info = variable_info

        self.timeseries = timeseries

    def compare(self, other: Self | pd.DataFrame):
        """Compare the variables to another Variables object or DataFrame.

        Parameters
        ----------
        other: pd.DataFrame
            The DataFrame to compare the variables to.
        """
        if isinstance(other, Variables):
            other = other.to_pandas()

        df = self.to_pandas()

        # Compare columns and indices
        logger.info(f"Columns that don't match: {set(df.columns) - set(other.columns)}")
        logger.info(f"Indices that don't match: {set(df.index) - set(other.index)}")

        # Compare values
        diff = df.sub(other)
        rel_diff = df.sub(other).div(other).mul(100)
        rel_diff = rel_diff[other != 0]

        return diff, rel_diff

    @classmethod
    def from_excel(cls, file_path: os.PathLike, *args, **kwargs):
        """Initialize the variables from an Excel file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the Excel file to read the variables from.
        """
        raise NotImplementedError("Not implemented yet")

    @classmethod
    def from_json(cls, file_path: os.PathLike, *args, **kwargs):
        """Read the timeseries from a JSON file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the JSON file to read the timeseries from.
        """
        with open(file_path, "r") as file:
            data = json.load(file)
        timeseries = {k: torch.tensor(v) for k, v in data.items()}
        return cls(timeseries=timeseries)

    def get_default_variables(self):
        """Return the default variables information dictionary.

        This function returns a dictionary of the variable information with
        their default values. Users should implement this function in their
        model class, and it should return a dictionary with the variable names
        as keys and the variable information as values. The variable information
        should contain at least the following keys:
        - "history": int - The number of periods that the variable requires information from.
        - "sectors": list - The sectors that the variable is associated with.
        - "unit": str - The unit of the variable.
        - "notation": str - The notation of the variable.
        """
        return {}

    def info_to_csv(self, file_path: str, sphinx_math: bool = False):
        """Convert the variables information to a CSV file.

        Parameters
        ----------
        file_path: str
            The path to the CSV file to save the variables information to.
        sphinx_math: bool
            Whether to add a ":math:" marker to the notation column, e.g. for
            usage in the documentation
        """
        df = pd.DataFrame.from_dict(self.info, orient="index")
        df["sectors"] = df["sectors"].apply(lambda x: ", ".join(x))
        df["history"] = df["history"].astype(int)
        if sphinx_math:
            df["notation"] = df["notation"].apply(lambda x: r":math:`" + x + r"`")
        df.columns = [i.title() for i in df.columns]
        df.to_csv(file_path)

    def initialize_tensors(self, t: int, **kwargs):
        """Initialize the output tensors, creating two different dictionaries.
        First, a dictionary for the state variables (i.e. those that require
        only t-1 information, but no history) and second a dictionary for the
        history variables (i.e. those that require information from further
        previous periods). This distinction is important for PyTorch based
        simulations to reduce memory usage.

        Parameters
        ----------
        t: int
            The number of periods to initialize the tensors for.
        """
        # State variables (only t-1 information)
        state_vars = self.new_state(**kwargs)

        # History variables (v["history"] rows)
        self.history = {}
        for k, v in self.info.items():
            if "history" in v and v["history"] > 0:
                self.history[k] = []

        # Initialize the timeseries
        self.timeseries = {
            k: torch.zeros(t, len(v["sectors"])) for k, v in self.info.items()
        }

        return state_vars, self.history

    def new_state(self, **kwargs):
        """Initialize the state variables for the given period."""

        state = {}
        for k, v in self.info.items():
            state[k] = torch.zeros(len(v["sectors"]), **kwargs)

        return state

    def update_history(self, state: dict):
        """Update the history variables for the given period.

        Parameters
        ----------
        state: dict
            The state variables for the given period.
        history: dict
            The history variables for the given period.
        """
        for k, v in self.history.items():
            steps = self.info[k]["history"]

            if len(v) < steps:
                v.insert(0, state[k].squeeze())
            else:
                del v[-1]
                v.insert(0, state[k].squeeze())

        vhistory = {}
        for k, v in self.history.items():
            vhistory[k] = torch.stack(v, dim=0)

        return vhistory

    def record_state(
        self,
        t: int,
        state_vars: dict,
    ):
        """Record the state variables for the given period.

        Parameters
        ----------
        t: int
            The period to record the state variables for.
        state_vars: dict
            The state variables to record.
        """
        key_state = set(state_vars.keys())
        key_series = set(self.timeseries.keys())

        # Warn if there are keys that are in the state variables
        # but not in the timeseries
        if len(key_state - key_series) > 0:
            msg = "keys in state variables but not timeseries"
            logger.warning(f"{msg}: {key_state - key_series}")

        # Only keep the keys that are in both dictionaries
        for k in list(key_state.intersection(key_series)):
            try:
                self.timeseries[k][t, :] = state_vars[k].clone().detach()
            except Exception as e:
                logger.error(f"Error recording {k}:")
                logger.error(f"State: {state_vars[k].clone().detach()}")
                logger.error(f"Timeseries: {self.timeseries[k][t, :]}")
                raise e

    def to_excel(self, file_path: os.PathLike):
        """Convert the variables to an Excel file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the Excel file to save the variables to.
        """
        raise NotImplementedError("Not implemented yet")

    def to_json(self, file_path: os.PathLike):
        """Convert the parameters to a JSON file.

        Parameters
        ----------
        file_path: os.PathLike
            The path to the JSON file to save the timeseries to.
        """
        dicts = {k: v.tolist() for k, v in self.timeseries.items()}
        with open(file_path, "w") as file:
            json.dump(dicts, file)

    def to_pandas(self):
        """Convert the variables to a pandas DataFrame."""
        df = pd.concat({k: pd.DataFrame(v) for k, v in self.timeseries.items()}, axis=1)
        return df


if __name__ == "__main__":
    pass
