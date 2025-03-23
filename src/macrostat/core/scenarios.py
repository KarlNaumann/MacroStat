"""
Scenarios class for the MacroStat model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import json
import logging
import random

import numpy as np
import pandas as pd
import torch

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)


class Scenarios:
    """Scenarios class for the MacroStat model.

    The aim of this class is to provide a uniform interface for handling
    scenarios, in particular for exogeneous shocks (e.g. also for the case
    where the shocks are calibrated to fit the data, such as using productivity
    shocks to fit the trajectory of GDP). It also contains user-specified scenarios
    such as exogenous supply shocks or the like.
    """

    def __init__(
        self,
        parameters: Parameters,
        scenarios: dict | None = None,
        scenario_info: dict | None = None,
        calibration_variables: list[str] | None = None,
    ):
        """Initialize the scenarios for the model.

        If no scenarios are provided, the default scenarios will be used.

        Parameters
        ----------
        parameters: Parameters
            The parameters of the model. These are necessary to create the
            default scenarios.
        scenarios: dict | None
            The scenarios to initialize the model with. If None, the default
            scenarios will be used.
        scenario_info: dict | None
            The colors to use for the scenario variables. If None, the default
            colors will be used.
        calibration_variables: list[str] | None
            The scenario variables that can be used in the calibration. If
            None, calibration will be based on the LabourProductivityGeneral,
            ProductivityShockGeneral, and ProductionLossGeneral scenario variables.
        """
        self.parameters = parameters
        self.trigger = self.parameters["scenario_trigger"]
        self.scenario_duration = self.parameters["timesteps"] - self.trigger + 1

        self.timeseries = {0: self.get_default_scenario()}
        self.info = {
            0: {
                "Name": "Scenario.0",
                "Colour": "#000000",
                "Index": torch.arange(self.scenario_duration),
            }
        }

        if scenarios is not None:
            for name, timeseries in scenarios.items():
                self.add_scenario(name, timeseries)

        self.calibration_variables = (
            [] if calibration_variables is None else calibration_variables
        )

        if scenario_info is None:
            self.info = {
                k: dict(
                    Name=f"Scenario {k}",
                    Colour=f"#{random.randint(0, 0xFFFFFF):06x}",
                    Index=torch.arange(self.scenario_duration),
                )
                for k in self.timeseries.keys()
            }
        else:
            self.info = scenario_info

        self.current_scenario = 0

    def add_scenario(self, name: str, timeseries: dict, colour: str = None):
        """Add a scenario to the model.

        Parameters
        ----------
        name: str
            The name of the scenario.
        timeseries: dict
            The timeseries of the scenario.
        colour: str
            The colour of the scenario.
        """
        # Add the scenario info
        scID = len(self.info)
        if name is None:
            name = len(self.info)
        if colour is None:
            colour = f"#{random.randint(0, 0xFFFFFF):06x}"

        self.info[scID] = {
            "Name": name,
            "Colour": colour,
            "Index": np.arange(self.parameters["T"]),
        }

        # Copy default scenario as a starting point
        self.timeseries[scID] = self.get_default_scenario()
        trigger = self.parameters["scenario_trigger"]

        # Update the scenario timeseries using the user-provided timeseries
        for k, v in timeseries.items():
            if k not in self.timeseries[scID].keys():
                raise KeyError(
                    f"Key {k} not found in scenario {scID}. Add it to the default scenario."
                )

            # If the timeseries is a number, replace the default value
            if isinstance(v, (int, float)):
                self.timeseries[scID][k][trigger:] = v
            # If the timeseries is a vector, assume it starts at the trigger
            elif isinstance(v, torch.Tensor):
                t = min(len(v), self.parameters["T"] - trigger)
                self.timeseries[scID][k][trigger : trigger + t, 0] = v.squeeze()[:t]
            else:
                t = min(len(v), self.parameters["T"] - trigger)
                self.timeseries[scID][k][trigger : trigger + t, 0] = torch.tensor(v[:t])

            if isinstance(v, (pd.Series, pd.DataFrame)):
                self.info[scID]["Index"] = v.index.to_numpy()

    @classmethod
    def from_excel(cls, excel_path: str, parameters: Parameters):
        """Initialize the scenarios from an Excel file.

        Parameters
        ----------
        excel_path: str
            The path to the Excel file containing the scenarios.
        parameters: Parameters
            The parameters of the model. These are necessary to create the
            default scenarios.
        """
        raise NotImplementedError("Not implemented")

    @classmethod
    def from_json(cls, json_path: str, parameters: Parameters):
        """Initialize the scenarios from a JSON file.

        Parameters
        ----------
        json_path: str
            The path to the JSON file containing the scenarios.
        parameters: Parameters
            The parameters of the model. These are necessary to create the
            default scenarios.
        """
        with open(json_path, "r") as f:
            data = json.load(f)

        info = data.pop("ScenarioDetails")
        info = {int(float(k)): v for k, v in info.items()}

        timeseries = {}
        for k, v in info.items():
            if k != 0:
                timeseries[k] = data[f"Scenario.{k}"]

        return cls(parameters=parameters, scenarios=timeseries, scenario_info=info)

    def get_default_scenario(self) -> dict:
        """Return the default scenario variable in vectorized form."""
        default_values = self.get_default_scenario_values()

        vectorized = {}
        for k, v in default_values.items():
            vectorized[k] = v * torch.ones(self.parameters.hyper["timesteps"], 1)

        return vectorized

    def get_default_scenario_values(self) -> dict:
        """Return the default scenario values.

        This function returns a dictionary of the scenario values with their
        default values.
        """
        return {}

    def to_excel(self, excel_path: str):
        """Save the scenarios to an Excel file.

        Parameters
        ----------
        excel_path: str
            The path to the Excel file to save the scenarios to.
        """
        raise NotImplementedError("Not implemented")

    def to_json(self, json_path: str):
        """Save the scenarios to a JSON file.

        Parameters
        ----------
        json_path: str
            The path to the JSON file to save the scenarios to.
        """
        # Convert timeseries to dict of lists
        trigger = self.parameters["scenario_trigger"]
        data = {
            f"Scenario.{sc}": {k: v.squeeze()[trigger:].tolist() for k, v in ts.items()}
            for sc, ts in self.timeseries.items()
        }

        # Convert scenario info to dict of lists
        data["ScenarioDetails"] = {}
        for k, v in self.info.items():
            data["ScenarioDetails"][k] = {**v, "Index": v["Index"].tolist()}

        # Save the data to a JSON file
        with open(json_path, "w") as f:
            json.dump(data, f)

    def to_nn_parameters(self, scenario: int = 0):
        """Convert the scenarios to a PyTorch ParameterDict.

        Parameters
        ----------
        scenario: int
            The scenario to convert to PyTorch parameters.
        """
        # Set the current scenario
        self.current_scenario = scenario

        # In general, we keep scenarios fixed, so we set requires_grad to False
        vscenarios = torch.nn.ParameterDict(self.timeseries[scenario])

        for k, tensor in vscenarios.items():
            tensor.requires_grad = k in self.calibration_variables

        return vscenarios


if __name__ == "__main__":
    pass
