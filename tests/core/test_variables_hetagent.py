"""Tests for the heterogeneous-agent extensions of the Variables class.

Covers the four cases enumerated in the MacroStat_HetAgentInfra dispatch:

1. Variable-shape resolution from hyper (``sectors=["n_a", "n_b"]``).
2. Multi-level SFC routing via the ``level`` field.
3. Variable with no ``sfc`` key (non-SFC path; balance-sheet builder skips).
4. Backwards-compat: literal-string ``sectors=["Households"]`` (no hyper match)
   falls back to a size-1 axis.
"""

# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Karl Naumann-Woleske

import pytest

from macrostat.core import Parameters, Variables


@pytest.fixture
def params_with_hetagent_hyper():
    """Parameters whose hyper dict carries het-agent axis sizes.

    Default hyper does not contain ``n_a``/``n_b``; we inject them directly
    so the resolver in ``new_state()`` can find them via the fall-through
    in ``Parameters.__getitem__``.
    """
    p = Parameters()
    p.hyper["n_a"] = 10
    p.hyper["n_b"] = 5
    p.hyper["n_deciles"] = 4
    return p


def test_new_state_resolves_hyper_axes(params_with_hetagent_hyper):
    """Case 1: ``sectors=["n_a", "n_b"]`` → tensor of shape ``(10, 5)``."""
    info = {
        "DisposableIncome": {
            "sectors": ["n_a", "n_b"],
            "history": 0,
            "unit": "currency",
            "notation": "Y_d",
        }
    }
    v = Variables(variable_info=info, parameters=params_with_hetagent_hyper)
    state = v.new_state()
    assert state["DisposableIncome"].shape == (10, 5)


def test_new_state_resolves_int_literal(params_with_hetagent_hyper):
    """Int literal in ``sectors`` is used as-is for the axis size."""
    info = {"X": {"sectors": [7], "history": 0, "unit": "u", "notation": "x"}}
    v = Variables(variable_info=info, parameters=params_with_hetagent_hyper)
    state = v.new_state()
    assert state["X"].shape == (7,)


def test_balance_sheet_routes_via_level(params_with_hetagent_hyper):
    """Case 2: a variable with ``level: "Households"`` and a het-agent shape
    appears in the balance sheet under the Households sector.
    """
    info = {
        "HouseholdDeposits": {
            "sectors": ["n_deciles"],
            "level": "Households",
            "history": 0,
            "unit": "currency",
            "notation": "D_h",
            "sfc": [("asset", "Households")],
        }
    }
    params_with_hetagent_hyper.hyper["sectors"] = ["Households", "Banks"]
    v = Variables(variable_info=info, parameters=params_with_hetagent_hyper)
    bs = v.balance_sheet_theoretical()
    assert "Households" in bs.columns.get_level_values(0)


def test_non_sfc_variable_skipped(params_with_hetagent_hyper):
    """Case 3: a variable without ``sfc`` is silently skipped by
    ``verify_sfc_info`` and by every ``get_*_variables`` accessor.
    """
    info = {
        "AgentBalance": {
            "sectors": ["n_a"],
            "history": 0,
            "unit": "currency",
            "notation": "b",
        }
    }
    v = Variables(variable_info=info, parameters=params_with_hetagent_hyper)
    assert v.verify_sfc_info()
    assert v.get_stock_variables() == {}
    assert v.get_flow_variables() == {}
    assert v.get_index_variables() == {}


def test_legacy_literal_string_sector_size_one(params_with_hetagent_hyper):
    """Case 4: an unresolvable literal-string entry (e.g. ``"Households"``)
    falls back to the legacy ``len(sectors)`` interpretation, giving a
    size-1 axis for a single-element list.
    """
    info = {
        "HouseholdWealth": {
            "sectors": ["Households"],
            "history": 0,
            "unit": "currency",
            "notation": "W_h",
        }
    }
    v = Variables(variable_info=info, parameters=params_with_hetagent_hyper)
    state = v.new_state()
    assert state["HouseholdWealth"].shape == (1,)


def test_legacy_multi_label_sector(params_with_hetagent_hyper):
    """Backwards-compat: ``sectors=["Households","Firms","Banks"]`` (label
    list, no hyper match) gives ``len(sectors) = 3``.
    """
    info = {
        "WealthBySector": {
            "sectors": ["Households", "Firms", "Banks"],
            "history": 0,
            "unit": "currency",
            "notation": "W",
        }
    }
    v = Variables(variable_info=info, parameters=params_with_hetagent_hyper)
    state = v.new_state()
    assert state["WealthBySector"].shape == (3,)
