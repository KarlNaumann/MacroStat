"""
Variables class for the Godley-Lavoie 2006 SIM model.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import logging

from macrostat.core.variables import Variables
from macrostat.models.GL06SIMEX.parameters import ParametersGL06SIMEX

logger = logging.getLogger(__name__)


class VariablesGL06SIMEX(Variables):
    """Variables class for the Godley-Lavoie 2006 SIMEX model."""

    version = "SIMEX"

    def __init__(
        self,
        variable_info: dict | None = None,
        timeseries: dict | None = None,
        parameters: ParametersGL06SIMEX | None = None,
        *args,
        **kwargs,
    ):
        """Initialize the variables of the Godley-Lavoie 2006 SIMEX model."""

        if parameters is None:
            parameters = ParametersGL06SIMEX()

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
                "notation": r"C_d",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "ConsumptionSupply": {
                "notation": r"C_s",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "GovernmentDemand": {
                "notation": r"G_d",
                "unit": "MU",
                "history": 0,
                "sectors": ["Government"],
            },
            "GovernmentSupply": {
                "notation": r"G_s",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "TaxSupply": {
                "notation": r"T_s",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "TaxDemand": {
                "notation": r"T_d",
                "unit": "MU",
                "history": 0,
                "sectors": ["Government"],
            },
            "LabourSupply": {
                "notation": r"N_s",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "LabourDemand": {
                "notation": r"N_d",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "DisposableIncome": {
                "notation": r"YD",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "ExpectedDisposableIncome": {
                "notation": r"YD^e",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "Wages": {
                "notation": r"W",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "GovernmentMoneyStock": {
                "notation": r"H_s",
                "unit": "MU",
                "history": 0,
                "sectors": ["Government"],
            },
            "HouseholdMoneyDemand": {
                "notation": r"H_d",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "HouseholdMoneyStock": {
                "notation": r"H_h",
                "unit": "MU",
                "history": 0,
                "sectors": ["Household"],
            },
            "NationalIncome": {
                "notation": r"Y",
                "unit": "MU",
                "history": 0,
                "sectors": ["Macroeconomy"],
            },
        }
