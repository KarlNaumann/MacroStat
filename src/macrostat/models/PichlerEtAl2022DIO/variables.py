"""
Variables for the Pichler et al. (2022) Dynamic Input-Output model.
"""

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import logging

from macrostat.core.variables import Variables

from .parameters import ParametersPichlerEtAl2022DIO

logger = logging.getLogger(__name__)


class VariablesPichlerEtAl2022DIO(Variables):
    """Variables for the Pichler et al. (2022) Dynamic IO model.

    Each variable is either a vector of length N (one entry per IO sector)
    or a scalar (household-level aggregates).  Matrix state variables
    (inventories, intermediate consumption, orders) carry an additional
    ``matrix`` key that denotes the second dimension.

    Parameters
    ----------
    parameters : ParametersPichlerEtAl2022DIO or None
        Model parameters.  If ``None``, a default instance is created.
    """

    def __init__(self, parameters=None, *args, **kwargs):
        if parameters is None:
            parameters = ParametersPichlerEtAl2022DIO()
        super().__init__(parameters=parameters, *args, **kwargs)

    def get_default_variables(self):
        """Return the default variable specification dictionary.

        Returns
        -------
        dict
            Mapping from variable name to a dict with keys ``notation``,
            ``unit``, ``history``, ``sectors``, ``sfc``, and optionally
            ``matrix``.
        """
        ios = self.parameters.hyper["iosectors"]
        return {
            "GrossOutput": {
                "notation": r"x_{i,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "AggregateDemand": {
                "notation": r"d_{i,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "ConsumptionDemand": {
                "notation": r"c^d_{i,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "RealizedConsumption": {
                "notation": r"c_{i,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "LabourCompensation": {
                "notation": r"l_{i,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "Profits": {
                "notation": r"\pi_{i,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "ProductiveCapacity": {
                "notation": r"x^{cap}_{i,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "InputCapacity": {
                "notation": r"x^{inp}_{i,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "Inventories": {
                "notation": r"S_{ji,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "matrix": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "IntermediateConsumption": {
                "notation": r"Z_{ji,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "matrix": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "IntermediateOrders": {
                "notation": r"O_{ji,t}",
                "unit": "USD mn",
                "history": 0,
                "sectors": ios,
                "matrix": ios,
                "sfc": [("Index", "IOSectors")],
            },
            "TotalConsumptionDemand": {
                "notation": r"\tilde{c}^d_t",
                "unit": "USD mn",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
            "Savings": {
                "notation": r"s_t",
                "unit": "USD mn",
                "history": 0,
                "sectors": ["Household"],
                "sfc": [("Index", "Household")],
            },
        }
