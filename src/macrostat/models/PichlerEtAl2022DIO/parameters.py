"""
Parameters for the Pichler et al. (2022) Dynamic Input-Output model.

Reference: Pichler, Pangallo, Del Rio-Chanona, Lafond & Farmer (2022),
"Forecasting the propagation of pandemic shocks with a dynamic
input-output model", Journal of Economic Dynamics & Control.
"""

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import logging
import os
from pathlib import Path

import pandas as pd
import torch

from macrostat.core.parameters import Parameters

logger = logging.getLogger(__name__)

WIOD_SECTOR_CODES = [
    "A01",
    "A02",
    "A03",
    "B",
    "C10-C12",
    "C13-C15",
    "C16",
    "C17",
    "C18",
    "C19",
    "C20",
    "C21",
    "C22",
    "C23",
    "C24",
    "C25",
    "C26",
    "C27",
    "C28",
    "C29",
    "C30",
    "C31_C32",
    "C33",
    "D35",
    "E36",
    "E37-E39",
    "F",
    "G45",
    "G46",
    "G47",
    "H49",
    "H50",
    "H51",
    "H52",
    "H53",
    "I",
    "J58",
    "J59_J60",
    "J61",
    "J62_J63",
    "K64",
    "K65",
    "K66",
    "L68",
    "M69_M70",
    "M71",
    "M72",
    "M73",
    "M74_M75",
    "N",
    "O84",
    "P85",
    "Q",
    "R_S",
    "T",
]


class ParametersPichlerEtAl2022DIO(Parameters):
    """Parameters for the Pichler et al. (2022) Dynamic IO model.

    Scalar parameters (e.g. adjustment speeds, rates) live in the standard
    ``parameters`` dict.  Large tensor-valued parameters (IO matrices,
    initial-condition vectors) live in ``data_parameters`` and are loaded
    from external CSV files via the ``from_wiod_uk`` class method.

    Parameters
    ----------
    parameters : dict or None
        Scalar parameter overrides.
    hyperparameters : dict or None
        Hyperparameter overrides.
    data_parameters : dict or None
        Data-parameter overrides (tensors).
    """

    version = "PichlerEtAl2022DIO"

    def __init__(
        self,
        parameters: dict | None = None,
        hyperparameters: dict | None = None,
        data_parameters: dict | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(
            parameters=parameters,
            hyperparameters=hyperparameters,
            data_parameters=data_parameters,
            *args,
            **kwargs,
        )

    @classmethod
    def from_wiod_uk(
        cls,
        data_dir: os.PathLike,
        ihs_dir: os.PathLike | None = None,
        inv_file: os.PathLike | None = None,
        **kwargs,
    ):
        """Load parameters from WIOD UK CSV files.

        Parameters
        ----------
        data_dir : os.PathLike
            Path to the ``io_data_detail/`` directory containing the
            ``GBR_*.csv`` files.
        ihs_dir : os.PathLike or None
            Path to the ``IHS_matrices_processed/`` directory.  If ``None``,
            the ``CriticalInputMatrix`` is left at its default (zeros).
        inv_file : os.PathLike or None
            Path to the ``ons_table_ratio_inv_go.csv`` file.  If ``None``,
            the ``InventoryTargetDays`` is left at its default (zeros).
        **kwargs
            Additional keyword arguments passed to the constructor (e.g.
            ``parameters``, ``hyperparameters``).

        Returns
        -------
        ParametersPichlerEtAl2022DIO
            Fully populated parameter instance.
        """
        obj = cls(**kwargs)
        data_dir = Path(data_dir)

        wiod_mapping = {
            "TechnicalCoefficients": "GBR_A.csv",
            "IntermediateConsumptionMatrix": "GBR_Z.csv",
        }
        obj.load_data_parameters_from_directory(data_dir, mapping=wiod_mapping)

        obj._load_vector_from_csv("InitialGrossOutput", data_dir / "GBR_x.csv")
        obj._load_vector_from_csv("InitialLabourCompensation", data_dir / "GBR_lab.csv")
        obj._load_vector_from_csv("InitialProfits", data_dir / "GBR_cap.csv")

        f = pd.read_csv(data_dir / "GBR_f.csv", index_col=0, header=0)
        c0 = torch.tensor(f["C1"].values, dtype=torch.float)
        obj.data["InitialHouseholdConsumption"]["value"] = c0

        other_fd_cols = [c for c in f.columns if c != "C1"]
        other_fd = torch.tensor(f[other_fd_cols].sum(axis=1).values, dtype=torch.float)
        obj.data["InitialOtherFinalDemand"]["value"] = other_fd

        firm_taxes = cls._read_vector_csv(data_dir / "GBR_firm_taxes.csv")
        other_expenses = cls._read_vector_csv(data_dir / "GBR_other_expenses.csv")
        z_imports = cls._read_vector_csv(data_dir / "GBR_Z_imports.csv")
        x = obj.data["InitialGrossOutput"]["value"]
        expense = firm_taxes + other_expenses + z_imports
        safe_x = torch.where(x != 0, x, torch.ones_like(x))
        obj.data["OtherCostCoefficients"]["value"] = expense / safe_x

        f_imported = pd.read_csv(data_dir / "GBR_f_imported.csv", index_col=0, header=0)
        consumer_taxes = cls._read_vector_csv(data_dir / "GBR_consumer_taxes.csv")
        c_imports = f_imported.iloc[0, 0]
        total_cons_expenditure = (
            c_imports + consumer_taxes.sum().item() + c0.sum().item()
        )
        consumer_expenses_other = c_imports + consumer_taxes.sum().item()
        obj.data["HouseholdOtherCostCoefficient"]["value"] = torch.tensor(
            [consumer_expenses_other / total_cons_expenditure], dtype=torch.float
        )

        if ihs_dir is not None:
            ihs_dir = Path(ihs_dir)
            ihs = pd.read_csv(ihs_dir / "A.essential2.csv", index_col=0, header=0)
            n = obj.hyper["n_sectors"]
            tensor = torch.tensor(ihs.values[:n, :n], dtype=torch.float)
            obj.data["CriticalInputMatrix"]["value"] = tensor

        if inv_file is not None:
            inv = pd.read_csv(inv_file)
            n = obj.hyper["n_sectors"]
            obj.data["InventoryTargetDays"]["value"] = torch.tensor(
                inv["invratio"].values[:n], dtype=torch.float
            )

        obj.verify_parameters()
        return obj

    def _load_vector_from_csv(self, name: str, file_path: os.PathLike):
        """Load a vector data parameter from a WIOD-format CSV.

        Parameters
        ----------
        name : str
            Name of the data parameter to populate.
        file_path : os.PathLike
            Path to a CSV with a single value column and optional row index.
        """
        self.data[name]["value"] = self._read_vector_csv(file_path)[
            : self.hyper["n_sectors"]
        ]

    @staticmethod
    def _read_vector_csv(file_path: os.PathLike) -> torch.Tensor:
        """Read a single-column CSV into a 1-D tensor.

        Parameters
        ----------
        file_path : os.PathLike
            Path to the CSV file (first column is the index).

        Returns
        -------
        torch.Tensor
            1-D float tensor of the values.
        """
        return torch.tensor(
            pd.read_csv(file_path, index_col=0, header=0).iloc[:, 0].values,
            dtype=torch.float,
        )

    def get_default_parameters(self):
        """Return the default scalar parameters.

        Returns
        -------
        dict
            Mapping from parameter name to a dict with keys ``value``,
            ``lower bound``, ``upper bound``, ``notation``, and ``unit``.
        """
        return {
            "InventoryAdjustmentSpeed": {
                "lower bound": 1.0,
                "upper bound": 30.0,
                "notation": r"\tau",
                "unit": "days",
                "value": 5.0,
            },
            "HiringRate": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\gamma_H",
                "unit": "per day",
                "value": 1.0 / 30.0,
            },
            "FiringRate": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\gamma_F",
                "unit": "per day",
                "value": 2.0 / 30.0,
            },
            "ConsumptionPersistence": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\rho",
                "unit": ".",
                "value": 1.0 - (1.0 - 0.6) / 90.0,
            },
            "PropensityToConsume": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"m",
                "unit": ".",
                "value": 0.82,
            },
            "BenefitRate": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"b",
                "unit": ".",
                "value": 0.80,
            },
            "SavingsRedirection": {
                "lower bound": 0.0,
                "upper bound": 1.0,
                "notation": r"\Delta s",
                "unit": ".",
                "value": 0.50,
            },
        }

    def get_default_hyperparameters(self):
        """Return the default hyperparameters.

        Returns
        -------
        dict
            Mapping from hyperparameter name to its default value.
        """
        hyper = super().get_default_hyperparameters()
        hyper["timesteps"] = 182
        hyper["timesteps_initialization"] = 1
        hyper["scenario_trigger"] = 0
        hyper["n_sectors"] = 55
        hyper["sector_names"] = list(WIOD_SECTOR_CODES)
        hyper["sectors"] = list(WIOD_SECTOR_CODES) + ["Household"]
        hyper["iosectors"] = list(WIOD_SECTOR_CODES)
        hyper["production_function"] = "half_critical"
        hyper["hiring_firing"] = True
        hyper["firm_priority"] = False
        return hyper

    def get_default_data_parameters(self):
        """Return the default (zero-initialized) data parameters.

        Returns
        -------
        dict
            Mapping from data-parameter name to a dict with keys ``value``
            (a ``torch.Tensor``), ``lower_bound``, ``upper_bound``,
            ``notation``, and ``unit``.
        """
        n = self.hyper["n_sectors"]
        z = torch.zeros
        return {
            "TechnicalCoefficients": {
                "value": z(n, n),
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "notation": r"A",
                "unit": ".",
            },
            "CriticalInputMatrix": {
                "value": z(n, n),
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "notation": r"A^{ess}",
                "unit": ".",
            },
            "IntermediateConsumptionMatrix": {
                "value": z(n, n),
                "lower_bound": -1e12,
                "upper_bound": 1e12,
                "notation": r"Z",
                "unit": "USD mn",
            },
            "InitialGrossOutput": {
                "value": z(n),
                "lower_bound": -1e12,
                "upper_bound": 1e12,
                "notation": r"x_0",
                "unit": "USD mn",
            },
            "InitialLabourCompensation": {
                "value": z(n),
                "lower_bound": -1e12,
                "upper_bound": 1e12,
                "notation": r"l_0",
                "unit": "USD mn",
            },
            "InitialHouseholdConsumption": {
                "value": z(n),
                "lower_bound": -1e12,
                "upper_bound": 1e12,
                "notation": r"c_0",
                "unit": "USD mn",
            },
            "InitialProfits": {
                "value": z(n),
                "lower_bound": -1e12,
                "upper_bound": 1e12,
                "notation": r"\pi_0",
                "unit": "USD mn",
            },
            "InitialOtherFinalDemand": {
                "value": z(n),
                "lower_bound": -1e12,
                "upper_bound": 1e12,
                "notation": r"f_0",
                "unit": "USD mn",
            },
            "InventoryTargetDays": {
                "value": z(n),
                "lower_bound": 0.0,
                "upper_bound": 365.0,
                "notation": r"n",
                "unit": "days",
            },
            "OtherCostCoefficients": {
                "value": z(n),
                "lower_bound": -10.0,
                "upper_bound": 10.0,
                "notation": r"e_i/x_i",
                "unit": ".",
            },
            "HouseholdOtherCostCoefficient": {
                "value": z(1),
                "lower_bound": 0.0,
                "upper_bound": 1.0,
                "notation": r"c^{other}",
                "unit": ".",
            },
        }
