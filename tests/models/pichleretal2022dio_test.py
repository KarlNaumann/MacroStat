"""
Tests for the PichlerEtAl2022DIO model.

Tests the data loading, scenario construction, steady-state preservation,
and replication against the R reference implementation across multiple
model configurations (production functions, supply scenarios, labour
adjustment).

Data-dependent tests require the environment variable
``PICHLER2022_DATA_DIR`` to point to the root of the replication-code
directory (the one containing ``data/`` and ``output/`` sub-folders).
When unset, those tests are skipped automatically.
"""

import os
from pathlib import Path

import pandas as pd
import pytest
import torch

from macrostat.models.PichlerEtAl2022DIO import (
    ParametersPichlerEtAl2022DIO,
    PichlerEtAl2022DIO,
    ScenariosPichlerEtAl2022DIO,
    VariablesPichlerEtAl2022DIO,
)

# ---------------------------------------------------------------------------
# Resolve data paths from an environment variable
# ---------------------------------------------------------------------------

_DATA_ROOT_ENV = "PICHLER2022_DATA_DIR"
_data_root_str = os.environ.get(_DATA_ROOT_ENV)

if _data_root_str is not None:
    _DATA_ROOT = Path(_data_root_str)
    DATA_DIR = _DATA_ROOT / "data" / "io_data_detail"
    IHS_DIR = _DATA_ROOT / "data" / "IHS_matrices_processed"
    INV_FILE = _DATA_ROOT / "data" / "ons_table_ratio_inv_go.csv"
    SHOCK_CSV = _DATA_ROOT / "data" / "shocks" / "shock_scenarios.csv"
    R_OUTPUT_DIR = _DATA_ROOT / "output"
    FD_CSV = DATA_DIR / "GBR_f.csv"
    HAS_DATA = DATA_DIR.exists()
else:
    DATA_DIR = IHS_DIR = INV_FILE = SHOCK_CSV = R_OUTPUT_DIR = FD_CSV = None
    HAS_DATA = False

skip_no_data = pytest.mark.skipif(
    not HAS_DATA,
    reason=f"Set {_DATA_ROOT_ENV} to the replication-code root to enable",
)


# ======================================================================
# Unit tests (no data dependency)
# ======================================================================


class TestParametersPichlerEtAl2022DIO:
    def test_default_parameters(self):
        p = ParametersPichlerEtAl2022DIO()
        assert p.hyper["n_sectors"] == 3
        assert p.hyper["iosectors"] == ["A", "B", "C"]
        assert p.hyper["timesteps"] == 182
        assert "TechnicalCoefficients" in p
        assert p["TechnicalCoefficients"].shape == (3, 3)

    def test_data_parameters_default_shapes(self):
        p = ParametersPichlerEtAl2022DIO()
        assert p["InitialGrossOutput"].shape == (3,)
        assert p["IntermediateConsumptionMatrix"].shape == (3, 3)
        assert p["InventoryTargetDays"].shape == (3,)

    def test_default_spectral_radius(self):
        assert (
            torch.linalg.eigvals(
                ParametersPichlerEtAl2022DIO()["TechnicalCoefficients"]
            )
            .abs()
            .max()
            .item()
            < 0.5
        )

    def test_default_io_closure(self):
        p = ParametersPichlerEtAl2022DIO()
        assert (
            p["InitialGrossOutput"]
            - p["IntermediateConsumptionMatrix"].sum(dim=1)
            - p["InitialHouseholdConsumption"]
            - p["InitialOtherFinalDemand"]
        ).abs().max().item() < 1e-3

    def test_default_accounting_closure(self):
        p = ParametersPichlerEtAl2022DIO()
        assert (
            p["InitialProfits"]
            - (
                p["InitialGrossOutput"]
                - p["IntermediateConsumptionMatrix"].sum(dim=0)
                - p["InitialLabourCompensation"]
                - p["OtherCostCoefficients"] * p["InitialGrossOutput"]
            )
        ).abs().max().item() < 1e-3

    def test_hyperparameter_override(self):
        p = ParametersPichlerEtAl2022DIO(
            hyperparameters={"production_function": "leontief", "hiring_firing": False}
        )
        assert p.hyper["production_function"] == "leontief"
        assert p.hyper["hiring_firing"] is False
        assert p.hyper["timesteps"] == 182

    @skip_no_data
    def test_from_wiod_uk(self):
        p = ParametersPichlerEtAl2022DIO.from_wiod_uk(
            data_dir=DATA_DIR, ihs_dir=IHS_DIR, inv_file=INV_FILE
        )
        assert p["InitialGrossOutput"].sum().item() > 5e6
        assert p["TechnicalCoefficients"].sum().item() > 0
        assert p["InitialHouseholdConsumption"].sum().item() > 1e6
        assert (p["InventoryTargetDays"] > 0).all()

    @skip_no_data
    def test_from_wiod_uk_with_hyperparameter_override(self):
        assert (
            ParametersPichlerEtAl2022DIO.from_wiod_uk(
                data_dir=DATA_DIR,
                ihs_dir=IHS_DIR,
                inv_file=INV_FILE,
                hyperparameters={"production_function": "linear"},
            ).hyper["production_function"]
            == "linear"
        )

    @skip_no_data
    def test_io_balance(self):
        """x = c + rowSum(Z) + fd_other at initial conditions."""
        p = ParametersPichlerEtAl2022DIO.from_wiod_uk(
            data_dir=DATA_DIR, ihs_dir=IHS_DIR, inv_file=INV_FILE
        )
        assert torch.allclose(
            p["InitialGrossOutput"],
            p["InitialHouseholdConsumption"]
            + p["IntermediateConsumptionMatrix"].sum(dim=1)
            + p["InitialOtherFinalDemand"],
            atol=0.1,
        )


class TestScenariosPichlerEtAl2022DIO:
    def test_default_scenario(self):
        assert (
            "SupplyShock"
            in ScenariosPichlerEtAl2022DIO(
                parameters=ParametersPichlerEtAl2022DIO()
            ).timeseries[0]
        )

    @skip_no_data
    def test_from_shocks_csv(self):
        supply = ScenariosPichlerEtAl2022DIO.from_shocks_csv(
            parameters=ParametersPichlerEtAl2022DIO.from_wiod_uk(
                data_dir=DATA_DIR, ihs_dir=IHS_DIR, inv_file=INV_FILE
            ),
            shock_csv=SHOCK_CSV,
            final_demand_csv=FD_CSV,
        ).timeseries[1]["SupplyShock"]
        assert supply.shape == (182, 55)
        assert (supply[:83] == 0).all()
        assert supply[83:].abs().sum() > 0


class TestVariablesPichlerEtAl2022DIO:
    def test_default_variables(self):
        defaults = VariablesPichlerEtAl2022DIO().get_default_variables()
        assert "GrossOutput" in defaults
        assert "Inventories" in defaults
        assert "Savings" in defaults


# ======================================================================
# Simulation tests (require data files)
# ======================================================================


class TestSimulation:
    def test_default_instantiation_runs(self):
        """Zero-config instantiation simulates without error or NaN."""
        gross_output = PichlerEtAl2022DIO().simulate()["GrossOutput"]
        assert gross_output.shape[1] == 3
        assert not torch.isnan(gross_output).any()
        assert (gross_output > 0).all()

    def test_default_shock_propagates(self):
        """A supply shock to sector A reduces output downstream."""
        model = PichlerEtAl2022DIO()
        shock_timeseries = {
            "SupplyShock": torch.zeros(model.parameters.hyper["timesteps"], 3)
        }
        shock_start = 10
        shock_timeseries["SupplyShock"][shock_start : shock_start + 30, 0] = 0.5
        model.scenarios.add_vector_scenario(shock_timeseries, name="UpstreamShock")
        baseline = model.simulate(scenario=0)["GrossOutput"]
        shocked = model.simulate(scenario=1)["GrossOutput"]
        assert (
            baseline[shock_start + 5 :, 2].sum() - shocked[shock_start + 5 :, 2].sum()
        ).item() > 0

    @skip_no_data
    def test_no_shock_steady_state(self):
        """Without shocks, the economy should remain at steady state."""
        p = ParametersPichlerEtAl2022DIO.from_wiod_uk(
            data_dir=DATA_DIR, ihs_dir=IHS_DIR, inv_file=INV_FILE
        )
        x = PichlerEtAl2022DIO(
            parameters=p,
            scenarios=ScenariosPichlerEtAl2022DIO(parameters=p),
            variables=VariablesPichlerEtAl2022DIO(parameters=p),
        ).simulate(scenario=0)["GrossOutput"]
        for t in [5, 50, 100, 181]:
            assert (
                abs(x[t].sum().item() - x[0].sum().item()) < 1.0
            ), f"Steady state violated at t={t}"

    @skip_no_data
    def test_shock_reduces_output(self):
        """Supply + demand shocks should reduce aggregate output."""
        p = ParametersPichlerEtAl2022DIO.from_wiod_uk(
            data_dir=DATA_DIR, ihs_dir=IHS_DIR, inv_file=INV_FILE
        )
        x = PichlerEtAl2022DIO(
            parameters=p,
            scenarios=ScenariosPichlerEtAl2022DIO.from_shocks_csv(
                parameters=p, shock_csv=SHOCK_CSV, final_demand_csv=FD_CSV
            ),
            variables=VariablesPichlerEtAl2022DIO(parameters=p),
        ).simulate(scenario=1)["GrossOutput"]
        assert x[90].sum().item() < 0.85 * x[1].sum().item(), (
            f"Expected >15% output drop during lockdown, got "
            f"{(1 - x[90].sum().item() / x[1].sum().item()) * 100:.1f}%"
        )


# ======================================================================
# Replication tests against R reference (parametrized)
# ======================================================================

VARIANT_CONFIGS = [
    pytest.param(
        "r_baseline_aggregate.csv",
        "half_critical",
        "S4",
        True,
        0.001,
        id="baseline-half_critical-S4-hirefire",
    ),
    pytest.param(
        "r_variant_leontief.csv",
        "leontief",
        "S4",
        True,
        0.001,
        id="leontief-S4-hirefire",
    ),
    pytest.param(
        "r_variant_linear.csv",
        "linear",
        "S4",
        True,
        0.001,
        id="linear-S4-hirefire",
    ),
    pytest.param(
        "r_variant_S1.csv",
        "half_critical",
        "S1",
        True,
        0.001,
        id="half_critical-S1-hirefire",
    ),
    pytest.param(
        "r_variant_nohirefire.csv",
        "half_critical",
        "S4",
        False,
        0.001,
        id="half_critical-S4-nohirefire",
    ),
]


class TestReplicationAgainstR:
    @skip_no_data
    @pytest.mark.parametrize(
        "r_file, production_function, supply_scenario, hiring_firing, tol",
        VARIANT_CONFIGS,
    )
    def test_variant(
        self,
        r_file,
        production_function,
        supply_scenario,
        hiring_firing,
        tol,
    ):
        """Aggregate gross output matches R within float32 tolerance."""
        r_path = R_OUTPUT_DIR / r_file
        if not r_path.exists():
            pytest.skip(f"R output file not found: {r_file}")

        r_agg = pd.read_csv(r_path)

        params = ParametersPichlerEtAl2022DIO.from_wiod_uk(
            data_dir=DATA_DIR,
            ihs_dir=IHS_DIR,
            inv_file=INV_FILE,
            hyperparameters={
                "production_function": production_function,
                "hiring_firing": hiring_firing,
            },
        )
        scenarios = ScenariosPichlerEtAl2022DIO.from_shocks_csv(
            parameters=params,
            shock_csv=SHOCK_CSV,
            final_demand_csv=FD_CSV,
            supply_scenario=supply_scenario,
        )
        variables = VariablesPichlerEtAl2022DIO(parameters=params)
        model = PichlerEtAl2022DIO(
            parameters=params,
            scenarios=scenarios,
            variables=variables,
        )
        result = model.simulate(scenario=1)
        py_x = result["GrossOutput"]

        max_rel_err = 0.0
        worst_t = 0
        for r_row in range(len(r_agg)):
            r_val = r_agg["x"].iloc[r_row]
            py_val = py_x[r_row + 1].sum().item()
            if abs(r_val) > 1e-6:
                rel_err = abs(py_val - r_val) / abs(r_val)
                if rel_err > max_rel_err:
                    max_rel_err = rel_err
                    worst_t = r_row + 1

        assert max_rel_err < tol, (
            f"Max relative error {max_rel_err:.6f} at time {worst_t} "
            f"exceeds tolerance {tol}"
        )
