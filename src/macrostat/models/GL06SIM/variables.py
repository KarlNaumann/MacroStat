"""
Variables class for the Godley-Lavoie 2006 SIM model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.variables import Variables
from macrostat.models.GL06SIM.parameters import ParametersGL06SIM

logger = logging.getLogger(__name__)


class VariablesGL06SIM(Variables):
    """Variables class for the Godley-Lavoie 2006 SIM model."""

    version = "SIM"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersGL06SIM | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables of the Godley-Lavoie 2006 SIM model."""

        if parameters is None:
            parameters = ParametersGL06SIM()

        super().__init__(
            variable_info=variable_info,
            timeseries=timeseries,
            parameters=parameters,
            *args,
            **kwargs,
        )

    def get_default_variables(self):
        """Return the default variables information dictionary."""
        return {
            "ConsumptionDemand": {
                "notation": r"C_d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "ConsumptionSupply": {
                "notation": r"C_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Inflow", "Production"), ("Outflow", "Household")],
            },
            "GovernmentDemand": {
                "notation": r"G_d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Index", "Government")],
            },
            "GovernmentSupply": {
                "notation": r"G_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Inflow", "Production"), ("Outflow", "Government")],
            },
            "TaxDemand": {
                "notation": r"T_d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Index", "Government")],
            },
            "TaxSupply": {
                "notation": r"T_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Outflow", "Household"), ("Inflow", "Government")],
            },
            "LabourDemand": {
                "notation": r"N_d(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Production")],
            },
            "LabourSupply": {
                "notation": r"N_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "LabourEarnings": {
                "notation": r"W(t)\cdot N_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Inflow", "Household"), ("Outflow", "Production")],
            },
            "DisposableIncome": {
                "notation": r"YD(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "Wages": {
                "notation": r"W(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "GovernmentMoneyStock": {
                "notation": r"H_s(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
                "sfc": [("Liability", "Government")],
            },
            "HouseholdMoneyStock": {
                "notation": r"H_h(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Asset", "Household")],
            },
            "NationalIncome": {
                "notation": r"Y(t)",
                "unit": "USD",
                "history": 0,
                "sectors": ["Macroeconomy"],
                "sfc": [("Index", "Macroeconomy")],
            },
        }
