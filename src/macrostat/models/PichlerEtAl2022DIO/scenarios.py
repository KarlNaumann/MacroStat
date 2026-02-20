"""
Scenarios for the Pichler et al. (2022) Dynamic Input-Output model.

Encodes the time-varying exogenous shock processes (supply shocks, demand
shocks, fear-of-infection, permanent income expectations, other final demand).
"""

__author__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"

import logging
import os

import numpy as np
import pandas as pd
import torch

from macrostat.core.scenarios import Scenarios

from .parameters import ParametersPichlerEtAl2022DIO

logger = logging.getLogger(__name__)


class ScenariosPichlerEtAl2022DIO(Scenarios):
    """Scenarios for the Pichler et al. (2022) Dynamic IO model.

    Scenario variables are either scalar ``(T, 1)`` or vector-valued
    ``(T, N)`` tensors.  The base-class ``get_default_scenario`` broadcasts
    the values returned by ``get_default_scenario_values`` across time
    using ``v * torch.ones(T, 1)``, which gives the correct shape for
    both cases.

    Parameters
    ----------
    parameters : ParametersPichlerEtAl2022DIO or None
        Model parameters.  If ``None``, a default instance is created.
    scenarios : dict or None
        Pre-built scenario timeseries, keyed by scenario ID.
    scenario_info : dict or None
        Metadata for each scenario (name, colour, etc.).
    """

    version = "PichlerEtAl2022DIO"

    def __init__(
        self,
        parameters=None,
        scenarios: dict | None = None,
        scenario_info: dict | None = None,
        *args,
        **kwargs,
    ):
        if parameters is None:
            parameters = ParametersPichlerEtAl2022DIO()

        super().__init__(
            parameters=parameters,
            scenarios=scenarios,
            scenario_info=scenario_info,
            *args,
            **kwargs,
        )

    def get_default_scenario_values(self) -> dict:
        """Return the steady-state (no-shock) scenario values.

        Household demand preferences are set to the initial household
        consumption vector shares, or uniform if the initial consumption
        vector is zero.

        Returns
        -------
        dict
            Mapping from scenario variable name to a scalar or 1-D tensor
            representing the per-timestep default value.
        """
        N = self.parameters.hyper["n_sectors"]
        c0 = self.parameters["InitialHouseholdConsumption"]
        c_sum = c0.sum()
        prefs = c0 / c_sum if c_sum > 0 else torch.ones(N) / N

        return {
            "SupplyShock": torch.zeros(N),
            "DemandPreferences": prefs,
            "FearOfInfection": 0.0,
            "PermanentIncomeExpectation": 1.0,
            "OtherFinalDemand": self.parameters["InitialOtherFinalDemand"].clone(),
        }

    def add_vector_scenario(
        self,
        timeseries: dict[str, torch.Tensor],
        name: str = "Baseline",
        colour: str = "#CC0000",
    ):
        """Add a scenario with full pre-built ``(T, K)`` timeseries tensors.

        This bypasses the base-class ``add_scenario`` which does not
        support multi-dimensional tensor assignment.

        Parameters
        ----------
        timeseries : dict[str, torch.Tensor]
            Mapping from scenario variable name to a ``(T, K)`` tensor.
        name : str
            Human-readable scenario name.
        colour : str
            Hex colour code for plotting.
        """
        scID = len(self.info)
        self.timeseries[scID] = self.get_default_scenario()

        for k, v in timeseries.items():
            if k not in self.timeseries[scID]:
                raise KeyError(
                    f"Key {k} not in default scenario. "
                    f"Valid keys: {list(self.timeseries[scID].keys())}"
                )
            self.timeseries[scID][k] = v

        self.info[scID] = {
            "Name": name,
            "Colour": colour,
            "Index": np.arange(self.scenario_duration),
        }

    @classmethod
    def from_shocks_csv(
        cls,
        parameters,
        shock_csv: os.PathLike,
        final_demand_csv: os.PathLike,
        supply_scenario: str = "S4",
        demand_scenario: str = "D",
        other_fd_scenario: str = "strong",
        delta_srate: float = 0.5,
        rho_bar: float = 0.6,
        l_shape_believe: str = "50%",
    ):
        """Build scenario tensors replicating the R ``initialize_shocks()``.

        Parameters
        ----------
        parameters : ParametersPichlerEtAl2022DIO
            The model parameters (needed for sector names, timesteps, etc.).
        shock_csv : os.PathLike
            Path to ``shock_scenarios.csv``.
        final_demand_csv : os.PathLike
            Path to ``GBR_f.csv`` (final demand matrix).
        supply_scenario : str
            Supply shock scenario code (``S1``--``S6``).
        demand_scenario : str
            Demand shock scenario code (``D``).
        other_fd_scenario : str
            Other final demand scenario (``weak`` / ``strong``).
        delta_srate : float
            Fraction of fear-of-infection averted consumption that is saved.
        rho_bar : float
            Quarterly consumption persistence (converted to daily internally).
        l_shape_believe : str
            Permanent income belief (``"0%"``, ``"50%"``, ``"100%"``).

        Returns
        -------
        ScenariosPichlerEtAl2022DIO
            Instance with the default (no-shock) scenario at index 0 and
            the constructed shock scenario at index 1.
        """
        T = parameters["timesteps"]
        N = parameters["n_sectors"]
        sector_names = parameters.hyper["sector_names"]

        shocks = pd.read_csv(shock_csv, index_col=0, header=0)
        f = pd.read_csv(final_demand_csv, index_col=0, header=0)
        c0 = f["C1"].values[:N]
        l0 = parameters["InitialLabourCompensation"].numpy()

        # R uses 1-indexed arrays; convert to 0-indexed for Python:
        # R `+2` dates -> Python `+1`; R plain dates -> Python `-1`
        t_shock = (pd.Timestamp("2020-03-23") - pd.Timestamp("2020-01-01")).days + 1
        t_open = (pd.Timestamp("2020-05-13") - pd.Timestamp("2020-01-01")).days + 1
        t_open_retail = (
            pd.Timestamp("2020-06-15") - pd.Timestamp("2020-01-01")
        ).days + 1

        # --- Supply shocks (N x T) ---
        delta_ = _build_supply_shocks(
            shocks,
            sector_names,
            supply_scenario,
            T,
            N,
            t_shock,
            t_open,
            t_open_retail,
        )

        # --- Demand shocks (theta_: N x T, epsilon_: T) ---
        # R demand code uses `as.Date(...) - as.Date(...)` without +2
        time_reopen_retail = (
            pd.Timestamp("2020-06-15") - pd.Timestamp("2020-01-01")
        ).days - 1
        theta_, epsilon_ = _build_demand_shocks(
            shocks,
            sector_names,
            c0,
            demand_scenario,
            delta_srate,
            T,
            N,
            t_shock,
            t_open,
            time_reopen_retail,
        )

        # --- Permanent income expectations (T,) ---
        supply_shock_vector = shocks[supply_scenario].values[:N]
        rho1 = 1.0 - (1.0 - rho_bar) / 90.0
        rho0 = 1.0 - rho1
        delta_L = (l0[:N] * supply_shock_vector).sum() / l0[:N].sum()

        if l_shape_believe == "50%":
            etat = -delta_L / 4.0 * (1.0 - rho1)
        elif l_shape_believe == "0%":
            etat = 0.0
        else:
            etat = -delta_L / 2.0 * (1.0 - rho1)

        xi_ = torch.ones(T)
        xi_[t_shock:t_open] = 1.0 - delta_L / 2.0
        for t in range(t_open, T):
            xi_[t] = rho0 + rho1 * xi_[t - 1] + etat

        # --- Other final demand (N x T) ---
        fd_other_ = _build_other_final_demand(
            shocks,
            f,
            other_fd_scenario,
            T,
            N,
            t_shock,
        )

        # Reshape to (T, K) for the framework
        shock_timeseries = {
            "SupplyShock": delta_.T,  # (T, N)
            "DemandPreferences": theta_.T,  # (T, N)
            "FearOfInfection": epsilon_.unsqueeze(1),  # (T, 1)
            "PermanentIncomeExpectation": xi_.unsqueeze(1),  # (T, 1)
            "OtherFinalDemand": fd_other_.T,  # (T, N)
        }

        obj = cls(parameters=parameters)
        obj.add_vector_scenario(shock_timeseries, name="Baseline")
        return obj


# ======================================================================
# Helper functions for building shock matrices
# ======================================================================


def _build_supply_shocks(
    shocks,
    sector_names,
    scenario,
    T,
    N,
    t_shock,
    t_open,
    t_open_retail,
):
    """Build the ``(N, T)`` supply-shock matrix ``delta_``.

    Parameters
    ----------
    shocks : pandas.DataFrame
        Shock scenarios table (rows = sectors, columns = scenario codes).
    sector_names : list[str]
        Ordered list of sector codes.
    scenario : str
        Supply shock scenario code (``S1``--``S6``).
    T : int
        Number of timesteps.
    N : int
        Number of sectors.
    t_shock : int
        0-indexed timestep when shocks begin.
    t_open : int
        0-indexed timestep when the economy starts reopening.
    t_open_retail : int
        0-indexed timestep when retail sectors reopen.

    Returns
    -------
    torch.Tensor
        ``(N, T)`` supply-shock matrix with values in ``[0, 1]``.
    """
    supply_vec = torch.tensor(shocks[scenario].values[:N], dtype=torch.float)
    rli = torch.tensor(shocks["RLI"].values[:N], dtype=torch.float)
    uk_ess = torch.tensor(shocks["UK essential"].values[:N], dtype=torch.float)

    retail_idx = [i for i, s in enumerate(sector_names) if s in ("G45", "G47")]

    if scenario in ("S1",):
        delta_ = supply_vec.unsqueeze(1).expand(N, T).clone()
        delta_[:, :t_shock] = 0.0
        for i in retail_idx:
            delta_[i, t_open_retail:] = 0.0
    elif scenario in ("S5", "S6"):
        delta_ = supply_vec.unsqueeze(1).expand(N, T).clone()
        delta_[:, :t_shock] = 0.0
        delta_[:, t_open:] = 0.0
    elif scenario in ("S2", "S3", "S4"):
        ess_mat = uk_ess.unsqueeze(1).expand(N, T).clone()
        ess_mat[:, :t_shock] = 1.0
        for i in retail_idx:
            ess_mat[i, t_open_retail:] = 1.0

        rli_mat = (1.0 - rli).unsqueeze(1).expand(N, T).clone()

        ppi_mat = torch.zeros(N, T)
        length = t_open - t_shock + 1
        ramp = torch.linspace(1.0, 0.0, length)
        ppi_mat[:, t_shock : t_open + 1] = supply_vec.unsqueeze(1) * ramp.unsqueeze(0)

        delta_ = rli_mat * (1.0 - ess_mat * (1.0 - ppi_mat))
    else:
        delta_ = torch.zeros(N, T)

    return delta_


def _build_demand_shocks(
    shocks,
    sector_names,
    c0,
    scenario,
    delta_srate,
    T,
    N,
    t_shock,
    t_open,
    t_open_retail,
):
    """Build demand-shock tensors ``theta_`` and ``epsilon_``.

    Parameters
    ----------
    shocks : pandas.DataFrame
        Shock scenarios table.
    sector_names : list[str]
        Ordered list of sector codes.
    c0 : numpy.ndarray
        Initial household consumption vector (length ``N``).
    scenario : str
        Demand shock scenario code (``D``).
    delta_srate : float
        Fraction of fear-of-infection averted consumption that is saved.
    T : int
        Number of timesteps.
    N : int
        Number of sectors.
    t_shock : int
        0-indexed timestep when demand shocks begin.
    t_open : int
        0-indexed timestep when demand shocks start tapering.
    t_open_retail : int
        0-indexed timestep when non-durable retail demand shocks taper.

    Returns
    -------
    theta_ : torch.Tensor
        ``(N, T)`` demand preference matrix.
    epsilon_ : torch.Tensor
        ``(T,)`` fear-of-infection saving rate.
    """
    demand_vec = torch.tensor(shocks["D"].values[:N], dtype=torch.float)

    non_durable_sectors = {"C13-C15", "C18", "C20", "C26", "C29", "C31_C32"}
    idx = [i for i, s in enumerate(sector_names) if s not in non_durable_sectors]
    nonidx = [i for i, s in enumerate(sector_names) if s in non_durable_sectors]

    demand_mat = demand_vec.unsqueeze(1).expand(N, T).clone()
    demand_mat[:, :t_shock] = 0.0

    if t_open < T:
        length = T - t_open
        ramp = torch.linspace(1.0, 0.5, length)
        for i in idx:
            demand_mat[i, t_open:] = demand_vec[i] * ramp

    if t_open_retail < T:
        for i in nonidx:
            demand_mat[i, t_open_retail:] = demand_mat[idx[0], t_open_retail:]

    c0_t = torch.tensor(c0[:N], dtype=torch.float)
    c0_share = c0_t / c0_t.sum()

    theta_ = (1.0 - demand_mat) * c0_share.unsqueeze(1)
    epsilon_ = 1.0 - theta_.sum(dim=0)
    safe_denom = torch.where(
        (1.0 - epsilon_) != 0, 1.0 - epsilon_, torch.ones_like(epsilon_)
    )
    theta_ = theta_ / safe_denom.unsqueeze(0)
    epsilon_ = epsilon_ * delta_srate

    return theta_, epsilon_


def _build_other_final_demand(shocks, f, scenario, T, N, t_shock):
    """Build the ``(N, T)`` other-final-demand matrix.

    Parameters
    ----------
    shocks : pandas.DataFrame
        Shock scenarios table.
    f : pandas.DataFrame
        Full final-demand matrix (columns include ``C1``, ``G``, ``I``, etc.).
    scenario : str
        Other final demand scenario (``weak`` / ``strong``).
    T : int
        Number of timesteps.
    N : int
        Number of sectors.
    t_shock : int
        0-indexed timestep when shocks begin.

    Returns
    -------
    torch.Tensor
        ``(N, T)`` other-final-demand matrix.
    """
    demand_vec = torch.tensor(shocks["D"].values[:N], dtype=torch.float)

    if scenario == "weak":
        inv_shock, exp_shock = -0.1, -0.1
    else:
        inv_shock, exp_shock = -0.15, -0.15

    other_cols = [c for c in f.columns if c != "C1"]
    initial_fd = torch.tensor(f[other_cols].sum(axis=1).values[:N], dtype=torch.float)

    shocked_cols = {
        "C2": lambda v: torch.tensor(v, dtype=torch.float) * (1.0 - demand_vec),
        "G": lambda v: torch.tensor(v, dtype=torch.float),
        "I": lambda v: torch.tensor(v, dtype=torch.float) * (1.0 + inv_shock),
        "inventory": lambda v: torch.tensor(v, dtype=torch.float),
        "Export": lambda v: torch.tensor(v, dtype=torch.float) * (1.0 + exp_shock),
        "RoW_C": lambda v: torch.tensor(v, dtype=torch.float) * (1.0 + exp_shock),
        "RoW_G": lambda v: torch.tensor(v, dtype=torch.float) * (1.0 + exp_shock),
        "RoW_I": lambda v: torch.tensor(v, dtype=torch.float) * (1.0 + exp_shock),
        "RoW_inventory": lambda v: torch.tensor(v, dtype=torch.float)
        * (1.0 + exp_shock),
    }

    shocked_total = torch.zeros(N)
    for col, fn in shocked_cols.items():
        if col in f.columns:
            shocked_total = shocked_total + fn(f[col].values[:N])

    fd_other_ = torch.zeros(N, T)
    fd_other_[:, :t_shock] = initial_fd.unsqueeze(1).expand(N, t_shock)
    fd_other_[:, t_shock:] = shocked_total.unsqueeze(1).expand(N, T - t_shock)

    return fd_other_
