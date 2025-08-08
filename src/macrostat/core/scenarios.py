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
            scenarios will be used. Scenarios should be a str:dict (name:timeseries)
            dictionary.
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
                self.add_scenario(timeseries=timeseries, name=name)

        self.calibration_variables = (
            [] if calibration_variables is None else calibration_variables
        )

        if scenario_info is not None:
            self.info.update(scenario_info)

        self.verify_scenario_info()

        self.current_scenario = 0

    def __getitem__(self, item: tuple[int | str, str]) -> torch.Tensor:
        """Get a scenario timeseries from the model.

        Parameters
        ----------
        item: tuple[int, str] | int
            The index or name of the scenario.
        """
        try:
            scenario, variable = item
        except Exception as e:
            logger.error(
                f"Error getting scenario: {item} should be a tuple of (name, variable) or (index, variable)"
            )
            raise e

        if isinstance(scenario, str):
            scenario = self.get_scenario_index(scenario)

        return self.timeseries[scenario][variable]

    def add_scenario(self, timeseries: dict, name: str = None, colour: str = None):
        """Add a scenario to the model.

        Parameters
        ----------
        timeseries: dict
            The timeseries of the scenario.
        name: str | None
            The name of the scenario. If None, the scenario will be named
            "Scenario.N" where N is the number of scenarios.
        colour: str | None
            The colour of the scenario. If None, a random colour will be generated.
        """
        # Add the scenario info
        scID = len(self.info)
        name = f"Scenario.{scID}" if name is None else name
        colour = f"#{random.randint(0, 0xFFFFFF):06x}" if colour is None else colour

        self.info[scID] = {
            "Name": name,
            "Colour": colour,
            "Index": np.arange(
                self.parameters["timesteps"] - self.parameters["scenario_trigger"]
            ),
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
                t = min(len(v), self.parameters["timesteps"] - trigger)
                self.timeseries[scID][k][trigger : trigger + t, 0] = v.squeeze()[:t]
            elif isinstance(v, (pd.Series, pd.DataFrame)):
                t = min(len(v), self.parameters["timesteps"] - trigger)
                self.timeseries[scID][k][trigger : trigger + t, 0] = torch.tensor(
                    v.values[:t]
                )
                self.info[scID]["Index"] = v.index.to_numpy()
            else:
                t = min(len(v), self.parameters["timesteps"] - trigger)
                self.timeseries[scID][k][trigger : trigger + t, 0] = torch.tensor(v[:t])

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

        # Get the scenario details
        info = data.pop("ScenarioDetails")
        info = {int(float(k)): v for k, v in info.items()}

        # Get the calibration variables
        calibration_variables = data.pop("CalibrationVariables")

        # Get the timeseries
        timeseries = {}
        for k, v in info.items():
            if k != 0:
                timeseries[k] = data[f"Scenario.{k}"]

        return cls(
            parameters=parameters,
            scenarios=timeseries,
            scenario_info=info,
            calibration_variables=calibration_variables,
        )

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

    def get_scenario_index(self, scenario: str) -> int:
        """Get the index of a scenario by name.

        Parameters
        ----------
        scenario: str
            The name of the scenario.

        Returns
        -------
        int
            The index of the scenario.
        """
        for scenario_id, info in self.info.items():
            if info["Name"] == scenario:
                return scenario_id

        raise ValueError(f"Scenario {scenario} not found")

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

        # Add the calibration variables
        data["CalibrationVariables"] = self.calibration_variables

        # Convert scenario info to dict of lists
        data["ScenarioDetails"] = {}
        for k, v in self.info.items():
            data["ScenarioDetails"][k] = {**v, "Index": v["Index"].tolist()}

        # Save the data to a JSON file
        with open(json_path, "w") as f:
            json.dump(data, f)

    def vectorize_scenarios(self, timeseries: dict):
        """User-defined vectorization operations on the scenario timeseries.

        By default, scenarios are defined as single-column vectors where the
        rows (dim 0) matches the number of timesteps and the column is the
        scenario variable or paramter. However, for users with vectorized
        implementations, one can modify this function to create vectors of
        shape TxK as needed

        Parameters
        ----------
        timeseries: dict[str:torch.tensor]
            dictionary of the scenario timeseries

        Returns
        -------
        vectors: dict[str:torch.tensor]
            modified scenario timeseries

        """
        return timeseries

    def to_nn_parameters(self, scenario: int = 0):
        """Convert the scenarios to a PyTorch ParameterDict.

        Parameters
        ----------
        scenario: int
            The scenario to convert to PyTorch parameters.
        """
        self.current_scenario = scenario

        vectors = self.vectorize_scenarios(self.timeseries[scenario])
        vscenarios = torch.nn.ParameterDict(vectors)

        # In general, we keep scenarios fixed, so we set requires_grad to False
        for k, tensor in vscenarios.items():
            tensor.requires_grad = k in self.calibration_variables

        return vscenarios

    def verify_scenario_info(self):
        """Verify that the scenario info is consistent:
        1. There should be a one-to-one mapping between scenario info and timeseries
        2. The scenario info should be a subset of the timeseries keys
        3. The scenario info should have the ["Name", "Colour", "Index"] keys
        """

        # Check that there is a one-to-one mapping between scenario info and timeseries
        if set(self.info.keys()) != set(self.timeseries.keys()):
            raise ValueError(
                "There should be a one-to-one mapping between scenario info and timeseries"
            )

        # Check that the scenario info is a subset of the timeseries keys
        for k, v in self.info.items():
            if set(v.keys()) != {"Name", "Colour", "Index"}:
                raise ValueError(
                    f"Scenario info for scenario {k} should have the ['Name', 'Colour', 'Index'] keys"
                )


if __name__ == "__main__":
    pass
