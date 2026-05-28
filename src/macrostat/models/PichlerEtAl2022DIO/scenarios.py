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
        n_sectors = self.parameters.hyper["n_sectors"]
        initial_consumption = self.parameters["InitialHouseholdConsumption"]
        consumption_sum = initial_consumption.sum()
        if consumption_sum > 0:
            preference_share = initial_consumption / consumption_sum
        else:
            preference_share = torch.ones(n_sectors) / n_sectors

        return {
            "SupplyShock": torch.zeros(n_sectors),
            "DemandPreferences": preference_share,
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
        scenario_id = len(self.info)
        self.timeseries[scenario_id] = self.get_default_scenario()

        for key, value in timeseries.items():
            if key not in self.timeseries[scenario_id]:
                raise KeyError(
                    f"Key {key} not in default scenario. "
                    f"Valid keys: {list(self.timeseries[scenario_id].keys())}"
                )
            self.timeseries[scenario_id][key] = value

        self.info[scenario_id] = {
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
        n_timesteps = parameters["timesteps"]
        n_sectors = parameters["n_sectors"]
        sector_names = parameters.hyper["sector_names"]

        shocks = pd.read_csv(shock_csv, index_col=0, header=0)
        final_demand_df = pd.read_csv(final_demand_csv, index_col=0, header=0)
        initial_consumption = final_demand_df["C1"].values[:n_sectors]
        initial_labour = parameters["InitialLabourCompensation"].numpy()

        # R uses 1-indexed arrays; convert to 0-indexed for Python:
        # R `+2` dates -> Python `+1`; R plain dates -> Python `-1`
        t_shock = (pd.Timestamp("2020-03-23") - pd.Timestamp("2020-01-01")).days + 1
        t_open = (pd.Timestamp("2020-05-13") - pd.Timestamp("2020-01-01")).days + 1
        t_open_retail = (
            pd.Timestamp("2020-06-15") - pd.Timestamp("2020-01-01")
        ).days + 1

        # --- Supply shocks (n_sectors x n_timesteps) ---
        supply_shock_matrix = _build_supply_shocks(
            shocks=shocks,
            sector_names=sector_names,
            scenario=supply_scenario,
            n_timesteps=n_timesteps,
            n_sectors=n_sectors,
            t_shock=t_shock,
            t_open=t_open,
            t_open_retail=t_open_retail,
        )

        # --- Demand shocks: preference matrix and fear-of-infection vector ---
        # R demand code uses `as.Date(...) - as.Date(...)` without +2
        time_reopen_retail = (
            pd.Timestamp("2020-06-15") - pd.Timestamp("2020-01-01")
        ).days - 1
        preference_matrix, fear_of_infection = _build_demand_shocks(
            shocks=shocks,
            sector_names=sector_names,
            initial_consumption=initial_consumption,
            scenario=demand_scenario,
            delta_srate=delta_srate,
            n_timesteps=n_timesteps,
            n_sectors=n_sectors,
            t_shock=t_shock,
            t_open=t_open,
            t_open_retail=time_reopen_retail,
        )

        # --- Permanent income expectations (n_timesteps,) ---
        supply_shock_vector = shocks[supply_scenario].values[:n_sectors]
        rho1 = 1.0 - (1.0 - rho_bar) / 90.0
        rho0 = 1.0 - rho1
        delta_labour = (
            initial_labour[:n_sectors] * supply_shock_vector
        ).sum() / initial_labour[:n_sectors].sum()

        if l_shape_believe == "50%":
            etat = -delta_labour / 4.0 * (1.0 - rho1)
        elif l_shape_believe == "0%":
            etat = 0.0
        else:
            etat = -delta_labour / 2.0 * (1.0 - rho1)

        permanent_income = torch.ones(n_timesteps)
        permanent_income[t_shock:t_open] = 1.0 - delta_labour / 2.0
        for t in range(t_open, n_timesteps):
            permanent_income[t] = rho0 + rho1 * permanent_income[t - 1] + etat

        # --- Other final demand (n_sectors x n_timesteps) ---
        other_final_demand = _build_other_final_demand(
            shocks=shocks,
            final_demand_df=final_demand_df,
            scenario=other_fd_scenario,
            n_timesteps=n_timesteps,
            n_sectors=n_sectors,
            t_shock=t_shock,
        )

        # Reshape to (T, K) for the framework
        shock_timeseries = {
            "SupplyShock": supply_shock_matrix.T,
            "DemandPreferences": preference_matrix.T,
            "FearOfInfection": fear_of_infection.unsqueeze(1),
            "PermanentIncomeExpectation": permanent_income.unsqueeze(1),
            "OtherFinalDemand": other_final_demand.T,
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
    n_timesteps,
    n_sectors,
    t_shock,
    t_open,
    t_open_retail,
):
    """Build the ``(n_sectors, n_timesteps)`` supply-shock matrix.

    Parameters
    ----------
    shocks : pandas.DataFrame
        Shock scenarios table (rows = sectors, columns = scenario codes).
    sector_names : list[str]
        Ordered list of sector codes.
    scenario : str
        Supply shock scenario code (``S1``--``S6``).
    n_timesteps : int
        Number of timesteps.
    n_sectors : int
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
        ``(n_sectors, n_timesteps)`` supply-shock matrix with values in ``[0, 1]``.
    """
    supply_vec = torch.tensor(shocks[scenario].values[:n_sectors], dtype=torch.float)
    rli = torch.tensor(shocks["RLI"].values[:n_sectors], dtype=torch.float)
    uk_ess = torch.tensor(shocks["UK essential"].values[:n_sectors], dtype=torch.float)

    retail_idx = [i for i, s in enumerate(sector_names) if s in ("G45", "G47")]

    if scenario in ("S1",):
        supply_shock = supply_vec.unsqueeze(1).expand(n_sectors, n_timesteps).clone()
        supply_shock[:, :t_shock] = 0.0
        for i in retail_idx:
            supply_shock[i, t_open_retail:] = 0.0
    elif scenario in ("S5", "S6"):
        supply_shock = supply_vec.unsqueeze(1).expand(n_sectors, n_timesteps).clone()
        supply_shock[:, :t_shock] = 0.0
        supply_shock[:, t_open:] = 0.0
    elif scenario in ("S2", "S3", "S4"):
        ess_mat = uk_ess.unsqueeze(1).expand(n_sectors, n_timesteps).clone()
        ess_mat[:, :t_shock] = 1.0
        for i in retail_idx:
            ess_mat[i, t_open_retail:] = 1.0

        rli_mat = (1.0 - rli).unsqueeze(1).expand(n_sectors, n_timesteps).clone()

        ppi_mat = torch.zeros(n_sectors, n_timesteps)
        length = t_open - t_shock + 1
        ramp = torch.linspace(1.0, 0.0, length)
        ppi_mat[:, t_shock : t_open + 1] = supply_vec.unsqueeze(1) * ramp.unsqueeze(0)

        supply_shock = rli_mat * (1.0 - ess_mat * (1.0 - ppi_mat))
    else:
        supply_shock = torch.zeros(n_sectors, n_timesteps)

    return supply_shock


def _build_demand_shocks(
    shocks,
    sector_names,
    initial_consumption,
    scenario,
    delta_srate,
    n_timesteps,
    n_sectors,
    t_shock,
    t_open,
    t_open_retail,
):
    """Build demand-shock tensors: preference matrix and fear-of-infection vector.

    Parameters
    ----------
    shocks : pandas.DataFrame
        Shock scenarios table.
    sector_names : list[str]
        Ordered list of sector codes.
    initial_consumption : numpy.ndarray
        Initial household consumption vector (length ``n_sectors``).
    scenario : str
        Demand shock scenario code (``D``).
    delta_srate : float
        Fraction of fear-of-infection averted consumption that is saved.
    n_timesteps : int
        Number of timesteps.
    n_sectors : int
        Number of sectors.
    t_shock : int
        0-indexed timestep when demand shocks begin.
    t_open : int
        0-indexed timestep when demand shocks start tapering.
    t_open_retail : int
        0-indexed timestep when non-durable retail demand shocks taper.

    Returns
    -------
    preference_matrix : torch.Tensor
        ``(n_sectors, n_timesteps)`` demand preference matrix.
    fear_of_infection : torch.Tensor
        ``(n_timesteps,)`` fear-of-infection saving rate.
    """
    demand_vec = torch.tensor(shocks["D"].values[:n_sectors], dtype=torch.float)

    non_durable_sectors = {"C13-C15", "C18", "C20", "C26", "C29", "C31_C32"}
    idx = [i for i, s in enumerate(sector_names) if s not in non_durable_sectors]
    nonidx = [i for i, s in enumerate(sector_names) if s in non_durable_sectors]

    demand_mat = demand_vec.unsqueeze(1).expand(n_sectors, n_timesteps).clone()
    demand_mat[:, :t_shock] = 0.0

    if t_open < n_timesteps:
        length = n_timesteps - t_open
        ramp = torch.linspace(1.0, 0.5, length)
        for i in idx:
            demand_mat[i, t_open:] = demand_vec[i] * ramp

    if t_open_retail < n_timesteps:
        for i in nonidx:
            demand_mat[i, t_open_retail:] = demand_mat[idx[0], t_open_retail:]

    initial_consumption_tensor = torch.tensor(
        initial_consumption[:n_sectors], dtype=torch.float
    )
    consumption_share = initial_consumption_tensor / initial_consumption_tensor.sum()

    preference_matrix = (1.0 - demand_mat) * consumption_share.unsqueeze(1)
    fear_of_infection = 1.0 - preference_matrix.sum(dim=0)
    safe_denom = torch.where(
        (1.0 - fear_of_infection) != 0,
        1.0 - fear_of_infection,
        torch.ones_like(fear_of_infection),
    )
    preference_matrix = preference_matrix / safe_denom.unsqueeze(0)
    fear_of_infection = fear_of_infection * delta_srate

    return preference_matrix, fear_of_infection


def _build_other_final_demand(
    shocks,
    final_demand_df,
    scenario,
    n_timesteps,
    n_sectors,
    t_shock,
):
    """Build the ``(n_sectors, n_timesteps)`` other-final-demand matrix.

    Parameters
    ----------
    shocks : pandas.DataFrame
        Shock scenarios table.
    final_demand_df : pandas.DataFrame
        Full final-demand matrix (columns include ``C1``, ``G``, ``I``, etc.).
    scenario : str
        Other final demand scenario (``weak`` / ``strong``).
    n_timesteps : int
        Number of timesteps.
    n_sectors : int
        Number of sectors.
    t_shock : int
        0-indexed timestep when shocks begin.

    Returns
    -------
    torch.Tensor
        ``(n_sectors, n_timesteps)`` other-final-demand matrix.
    """
    demand_vec = torch.tensor(shocks["D"].values[:n_sectors], dtype=torch.float)

    if scenario == "weak":
        inv_shock, exp_shock = -0.1, -0.1
    else:
        inv_shock, exp_shock = -0.15, -0.15

    other_cols = [c for c in final_demand_df.columns if c != "C1"]
    initial_fd = torch.tensor(
        final_demand_df[other_cols].sum(axis=1).values[:n_sectors], dtype=torch.float
    )

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

    shocked_total = torch.zeros(n_sectors)
    for col, fn in shocked_cols.items():
        if col in final_demand_df.columns:
            shocked_total = shocked_total + fn(final_demand_df[col].values[:n_sectors])

    other_final_demand = torch.zeros(n_sectors, n_timesteps)
    other_final_demand[:, :t_shock] = initial_fd.unsqueeze(1).expand(n_sectors, t_shock)
    other_final_demand[:, t_shock:] = shocked_total.unsqueeze(1).expand(
        n_sectors, n_timesteps - t_shock
    )

    return other_final_demand
