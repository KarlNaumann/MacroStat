"""
A class for handling variables for a MacroStat model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import copy
import json
import logging
import os
import re

import pandas as pd
import torch
from typing_extensions import Self

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

        if timeseries is None:
            self.initialize_tensors(self.parameters.hyper["timesteps"])
        else:
            self.timeseries = timeseries

    ############################################################################
    # Accounting Functions
    ############################################################################

    def get_stock_variables(self):
        """Get all the stock variables from the info dictionary. Stock variables
        are those that are assets or liabilities, i.e. their "sfc" tuple starts
        with "asset" or "liability".
        """
        return {
            k: v["sfc"]
            for k, v in self.info.items()
            if v["sfc"][0][0].lower() in ["asset", "liability"]
        }

    def get_flow_variables(self):
        """Get all the flow variables from the info dictionary. Flow variables
        are those that are flows between sectors i.e. their "sfc" tuple starts
        with "inflow" or "outflow".
        """
        return {
            k: v["sfc"]
            for k, v in self.info.items()
            if v["sfc"][0][0].lower() in ["inflow", "outflow"]
        }

    def get_index_variables(self):
        """Get all the index variables from the info dictionary. Index variables
        are those that are indices, i.e. their "sfc" tuple starts with "index".
        """
        return {
            k: v["sfc"]
            for k, v in self.info.items()
            if v["sfc"][0][0].lower() == "index"
        }

    def balance_sheet_theoretical(
        self,
        mathfmt: str = "sphinx",
        non_camel_case: bool = False,
    ):
        """Calculate the theoretical balance sheet of the model based on the
        information in the info dictionary.

        Parameters
        ----------
        mathfmt: str
            The format to use for the math. Can be "sphinx", "myst", or "latex".
        non_camel_case: bool
            Whether to convert variable names to non-camel case.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the theoretical balance sheet of the model.
        """
        if not self.verify_sfc_info():
            raise ValueError("SFC information is not complete")

        bs = {}
        for k, v in self.get_stock_variables().items():
            for kind, sector in v:
                # Set the default balance sheet section to "Current"
                if isinstance(sector, list):
                    sector = tuple(sector)
                elif not isinstance(sector, tuple):
                    sector = (sector, "Current")

                # Convert variable name to non-camel case if requested
                item = k.replace(sector[0], "").replace(sector[0].lower(), "")
                if non_camel_case:
                    item = re.sub(r"([A-Z])", r" \1", item).strip()

                if kind.lower() == "asset":
                    notation = f"+{self.info[k]['notation']}"
                else:
                    # Only other option is that kind.lower() == "liability":
                    notation = f"-{self.info[k]['notation']}"

                # Add the item to the balance sheet
                if item not in bs:
                    bs[item] = {sector: notation}
                else:
                    bs[item][sector] = notation

        if len(bs) > 0:
            order = list(bs.keys())
            # Generate the balance sheet (in order of stocks)
            bs = pd.DataFrame.from_dict(bs, orient="index")
            bs = bs.loc[order]

            # Add columns for any other sectors that are not in the sfc
            for sector in self.parameters.hyper["sectors"]:
                if sector not in bs.columns:
                    bs[(sector, "Current")] = None
        else:
            # If there are no stocks, create a DataFrame with the sectors and Current
            bs = pd.DataFrame(
                columns=pd.MultiIndex.from_product(
                    [self.parameters.hyper["sectors"], ["Current"]]
                )
            )

        # Sort the columns by the order of the sectors
        bs = bs[self.parameters.hyper["sectors"]]

        # Apply the math format
        bs = self._apply_math_format(bs, mathfmt)

        # Add the total column to the end
        bs["Total"] = 0

        return bs

    def balance_sheet_actual(self):
        """Calculate the actual balance sheet of the model."""
        raise NotImplementedError("Not implemented yet")

    def transaction_matrix_theoretical(
        self,
        mathfmt: str = "sphinx",
        non_camel_case: bool = False,
    ):
        """Calculate the theoretical transaction matrix of the model based on the
        information in the info dictionary.

        Parameters
        ----------
        mathfmt: str
            The format to use for the math. Can be "sphinx", "myst", or "latex".
        non_camel_case: bool
            Whether to convert variable names to non-camel case.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the theoretical balance sheet of the model.
        """
        if not self.verify_sfc_info():
            raise ValueError("SFC information is not complete")

        tm = {}
        # Capture the flows
        for k, v in self.get_flow_variables().items():
            for kind, sector in v:
                sector = self._convert_sector_to_tuples(sector)

                # Convert variable name to non-camel case if requested
                if non_camel_case:
                    item = re.sub(r"([A-Z])", r" \1", k).strip()
                else:
                    item = k

                # Add item to the transaction matrix
                if kind.lower() == "inflow":
                    notation = f"+{self.info[k]['notation']}"
                else:
                    # Only other option is that kind.lower() == "outflow":
                    notation = f"-{self.info[k]['notation']}"

                # Add the item to the transaction matrix
                if item not in tm:
                    tm[item] = {sector: notation}
                else:
                    tm[item][sector] = notation

        # Capture the change in stocks
        for k, v in self.get_stock_variables().items():
            for kind, sector in v:
                sector = self._convert_sector_to_tuples(sector)

                # Change in wealth is not considered a flow
                if "wealth" in k.lower():
                    continue

                # Convert variable name to non-camel case if requested
                item = k.replace(sector[0], "").replace(sector[0].lower(), "")
                if non_camel_case:
                    item = re.sub(r"([A-Z])", r" \1", item).strip()

                item = f"Change in {item}"

                if kind.lower() == "asset":
                    notation = f"+{self.info[k]['notation']}"
                else:
                    # Only other option is that kind.lower() == "liability":
                    notation = f"-{self.info[k]['notation']}"

                # Add the item to the transaction matrix
                if item not in tm:
                    tm[item] = {sector: notation}
                else:
                    tm[item][sector] = notation

        # Maintain the order of flows then changes in stocks
        order = list(tm.keys())
        tm = pd.DataFrame.from_dict(tm, orient="index")
        tm = tm.loc[order]

        # Add columns for any other sectors that are not in the sfc
        for sector in self.parameters.hyper["sectors"]:
            if sector not in tm.columns:
                tm[(sector, "Current")] = None

        # Sort the columns by the order of the sectors
        tm = tm.loc[:, self.parameters.hyper["sectors"]]

        # Add the total column to the end
        tm["Total"] = 0

        # Add total row
        tm.loc["Total"] = 0

        # Apply the math format
        tm = self._apply_math_format(tm, mathfmt)

        return tm

    def transaction_matrix_actual(self):
        """Calculate the actual transaction matrix of the model."""
        raise NotImplementedError("Not implemented yet")

    ############################################################################
    # Comparison Functions
    ############################################################################

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

    ############################################################################
    # IO Functions
    ############################################################################

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
        varinfo = {k: v["info"] for k, v in data.items()}
        timeseries = {k: torch.tensor(v["timeseries"]) for k, v in data.items()}
        return cls(variable_info=varinfo, timeseries=timeseries)

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
        dicts = {
            k: {"info": self.info[k], "timeseries": v.tolist()}
            for k, v in self.timeseries.items()
        }
        with open(file_path, "w") as file:
            json.dump(dicts, file)

    def to_pandas(self):
        """Convert the variables to a pandas DataFrame."""
        # Copy deep so we can delete/add without affecting core var
        timeseries = copy.deepcopy(self.timeseries)

        # Flatten matrix variables: a timeseries per row of the matrix
        for k, v in self.timeseries.items():
            if "matrix" in self.info[k]:
                del timeseries[k]
                for i, subvar in enumerate(self.info[k]["matrix"]):
                    key = f"{k}{subvar}"
                    timeseries[key] = v[:, i, :]

        df = pd.concat({k: pd.DataFrame(v) for k, v in timeseries.items()}, axis=1)
        return df

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

    ############################################################################
    # General Functions
    ############################################################################

    def check_health(self):
        """Check the health of the variables. This is where the user may want to
        implement checks for consistency of the variables, e.g. whether the
        balance sheet is in balance, or whether the redundant equations hold.

        By default, this function returns True, indicating that the variables
        are healthy. This is to facilitate usage of the variables object in other
        functions.
        """
        logger.warning("Check health not implemented for this model")
        return True

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
        self.timeseries = {}
        for k, v in self.info.items():
            if "matrix" in v and len(v["sectors"]) > 0:
                self.timeseries[k] = torch.zeros(t, len(v["sectors"]), len(v["matrix"]))
            elif "sectors" in v and len(v["sectors"]) > 0:
                self.timeseries[k] = torch.zeros(t, len(v["sectors"]))
            else:
                self.timeseries[k] = torch.zeros(t, 1)

        return state_vars, self.history

    def new_state(self, **kwargs):
        """Initialize the state variables for the given period."""

        state = {}
        for k, v in self.info.items():
            if "matrix" in v and len(v["sectors"]) > 0:
                state[k] = torch.zeros(len(v["sectors"]), len(v["matrix"]), **kwargs)
            elif "sectors" in v and len(v["sectors"]) > 0:
                state[k] = torch.zeros(len(v["sectors"]), **kwargs)
            else:
                state[k] = torch.zeros(1, **kwargs)

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
            try:
                steps = self.info[k]["history"]

                if len(v) < steps:
                    v.insert(0, state[k].squeeze())
                else:
                    del v[-1]
                    v.insert(0, state[k].squeeze())
            except Exception as e:
                logger.error(f"Update history failed for {k}. Value is {v}")
                raise e

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
                print(k)
                raise e

    def verify_sfc_info(self):
        """Verify that the sfc information in the info dictionary is complete.

        This function checks first whether there is an sfc entry in the info
        dictionary for each variable. If there is, it then checks whether the
        sfc information makes sense, i.e. if they are flows they should have
        and "inflow" and "outflow" tuple in the list, otherwise they should
        contain at least one tuple with the first element being either "index",
        "asset" or "liability".
        """

        for k, v in self.info.items():
            if "sfc" not in v:
                logger.warning(f"No SFC information for {k}")
                return False

            if not isinstance(v["sfc"], (tuple, list)):
                logger.warning(f"Sfc information for {k} is not a list or tuple")
                return False
            elif isinstance(v["sfc"], tuple):
                if self._verify_sfc_item(v["sfc"], k):
                    continue
                else:
                    return False
            else:  # isinstance(v["sfc"], list):
                for sfc in v["sfc"]:
                    if self._verify_sfc_item(sfc, k):
                        continue
                    else:
                        return False

        return True

    ############################################################################
    # Helper Functions
    ############################################################################

    @staticmethod
    def _apply_math_format(df: pd.DataFrame, mathfmt: str):
        """Apply a math format to a DataFrame."""
        # Optionally wrap the notation in math mode
        if mathfmt == "sphinx":
            nonemask = df.isna()
            df = df.map(lambda x: r":math:`" + str(x) + r"`")
            df[nonemask] = ""
        elif mathfmt in ["myst", "latex"]:
            nonemask = df.isna()
            df = df.map(lambda x: r"$" + str(x) + r"$")
            df[nonemask] = ""
        else:
            raise ValueError(f"Invalid math format: {mathfmt}")

        return df

    def _verify_sfc_item(self, sfc: tuple, key: str):
        """Verify that an sfc item is valid by checking the first element is
        an accepted stock/flow/index type and that there are two elements in
        the tuple.

        Parameters
        ----------
        sfc: tuple
            The sfc item to verify.
        key: str
            The key of the variable.
        """
        if not isinstance(sfc[0], str):
            logger.warning(f"sfc information for {key} is not a valid item")
            return False
        elif sfc[0].lower() not in [
            "inflow",
            "outflow",
            "index",
            "asset",
            "liability",
        ]:
            logger.warning(f"sfc information for {key} is not a valid item")
            return False
        if len(sfc) != 2:
            logger.warning(
                f"sfc information for {key} is not a valid tuple of length 2: {sfc}"
            )
            return False
        return True

    @staticmethod
    def _convert_sector_to_tuples(sector):
        """The point of this method is to ensure that for the SFC tuples, there
        is always a sector and balance sheet section, if there is no balance
        sheet section, it is assumed to be the current account.

        Parameters
        ----------
        sector: str | tuple | list
            The sector to convert.

        Returns
        -------
        tuple
            A tuple of the sector and balance sheet section.
        """
        # Convert the sector to a tuple from str or list
        if isinstance(sector, list):
            sector = tuple(sector)
        elif isinstance(sector, str):
            sector = (sector, "Current")

        # Check that the sector is a tuple
        if not isinstance(sector, tuple):
            raise ValueError(f"Sector {sector} is not a tuple")
        elif len(sector) != 2:
            raise ValueError(f"Sector {sector} is not a tuple of length 2")
        else:
            return sector


if __name__ == "__main__":
    pass
