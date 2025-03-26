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
                "notation": "C_d",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
            },
            "ConsumptionSupply": {
                "notation": "C_s",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
            },
            "GovernmentDemand": {
                "notation": "G_d",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
            },
            "GovernmentSupply": {
                "notation": "G_s",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
            },
            "TaxSupply": {
                "notation": "T_s",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
            },
            "TaxDemand": {
                "notation": "T_d",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
            },
            "LabourSupply": {
                "notation": "N_s",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
            },
            "LabourDemand": {
                "notation": "N_d",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
            },
            "DisposableIncome": {
                "notation": "YD",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
            },
            "Wages": {
                "notation": "W",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
            },
            "GovernmentMoneyStock": {
                "notation": "H_s",
                "unit": "USD",
                "history": 0,
                "sectors": ["Government"],
            },
            "HouseholdMoneyStock": {
                "notation": "H_h",
                "unit": "USD",
                "history": 0,
                "sectors": ["Household"],
            },
            "NationalIncome": {
                "notation": "Y",
                "unit": "USD",
                "history": 0,
                "sectors": ["Macroeconomy"],
            },
        }
