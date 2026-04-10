"""
pytest code for macrostat.core.constraints

Covers ParameterLocation, ConstraintResolver, LinearConstraint
validation, and LinearConstraint.apply across scalar, 1-D and 2-D paths.
"""

__author__ = ["Karl Naumann-Woleske"]
__credits__ = ["Karl Naumann-Woleske"]
__license__ = "MIT"
__maintainer__ = ["Karl Naumann-Woleske"]

import math

import pytest
import torch

from macrostat.core.constraints import (
    ConstraintError,
    ConstraintResolver,
    LinearConstraint,
    ParameterLocation,
)

# ----------------------------------------------------------------------
# ParameterLocation
# ----------------------------------------------------------------------


class TestParameterLocation:
    """Read/write semantics for scalar, 1-D and 2-D slots."""

    def test_scalar_read(self):
        loc = ParameterLocation(tensor_key="TaxRate", index=None)
        params = {"TaxRate": torch.tensor(0.2)}
        assert loc.read(params).item() == pytest.approx(0.2)

    def test_scalar_write(self):
        loc = ParameterLocation(tensor_key="TaxRate", index=None)
        params = {"TaxRate": torch.tensor(0.2)}
        loc.write(params, torch.tensor(0.5))
        assert params["TaxRate"].item() == pytest.approx(0.5)

    def test_1d_read(self):
        loc = ParameterLocation(tensor_key="Share", index=(1,))
        params = {"Share": torch.tensor([0.3, 0.4, 0.3])}
        assert loc.read(params).item() == pytest.approx(0.4)

    def test_1d_write_out_of_place(self):
        loc = ParameterLocation(tensor_key="Share", index=(1,))
        base = torch.tensor([0.3, 0.4, 0.3])
        params = {"Share": base}
        loc.write(params, torch.tensor(0.6))
        # index_put is out-of-place; the original tensor is untouched
        assert base[1].item() == pytest.approx(0.4)
        assert params["Share"][1].item() == pytest.approx(0.6)
        # other entries preserved
        assert params["Share"][0].item() == pytest.approx(0.3)
        assert params["Share"][2].item() == pytest.approx(0.3)

    def test_2d_write(self):
        loc = ParameterLocation(tensor_key="Flow", index=(0, 1))
        base = torch.zeros(2, 2)
        params = {"Flow": base}
        loc.write(params, torch.tensor(0.7))
        assert params["Flow"][0, 1].item() == pytest.approx(0.7)
        # other cells still zero
        assert params["Flow"][0, 0].item() == 0.0
        assert params["Flow"][1, 0].item() == 0.0
        assert params["Flow"][1, 1].item() == 0.0

    def test_1d_write_is_differentiable(self):
        """Critical path: autograd flows through index_put."""
        loc = ParameterLocation(tensor_key="Share", index=(1,))
        base = torch.tensor([0.3, 0.4, 0.3], requires_grad=True)
        params = {"Share": base}
        new_val = torch.tensor(0.6, requires_grad=True)
        loc.write(params, new_val)
        # scalar loss depends only on the derived slot
        loss = params["Share"][1] * 2.0
        loss.backward()
        # Gradient flows to new_val, not to the old base[1]
        assert new_val.grad is not None
        assert new_val.grad.item() == pytest.approx(2.0)

    def test_1d_write_dtype_float64(self):
        """Device/dtype path: the write must inherit the base tensor's dtype.

        Catches the bug where torch.tensor([i], dtype=torch.long) is
        built without a device argument.
        """
        loc = ParameterLocation(tensor_key="Share", index=(1,))
        base = torch.tensor([0.3, 0.4, 0.3], dtype=torch.float64)
        params = {"Share": base}
        new_val = torch.tensor(0.6, dtype=torch.float64)
        loc.write(params, new_val)
        assert params["Share"].dtype == torch.float64
        assert params["Share"][1].item() == pytest.approx(0.6)


# ----------------------------------------------------------------------
# ConstraintResolver
# ----------------------------------------------------------------------


class TestConstraintResolver:
    """Resolver lookup and error messages."""

    def test_locate_known(self):
        loc = ParameterLocation(tensor_key="TaxRate", index=None)
        r = ConstraintResolver({"TaxRate": loc})
        assert r.locate("TaxRate") is loc

    def test_locate_unknown_raises(self):
        r = ConstraintResolver({"TaxRate": ParameterLocation("TaxRate", None)})
        with pytest.raises(ConstraintError, match="Unknown constraint parameter"):
            r.locate("Missing")

    def test_contains(self):
        r = ConstraintResolver({"TaxRate": ParameterLocation("TaxRate", None)})
        assert "TaxRate" in r
        assert "Missing" not in r

    def test_len(self):
        r = ConstraintResolver(
            {
                "a": ParameterLocation("a", None),
                "b": ParameterLocation("b", None),
            }
        )
        assert len(r) == 2

    def test_resolver_is_not_aliased(self):
        """Mutating the input mapping after construction must not affect it."""
        src = {"a": ParameterLocation("a", None)}
        r = ConstraintResolver(src)
        src["b"] = ParameterLocation("b", None)
        assert "b" not in r


# ----------------------------------------------------------------------
# LinearConstraint.__post_init__
# ----------------------------------------------------------------------


class TestLinearConstraintPostInit:
    """Construction-time validation."""

    def test_accepts_valid(self):
        c = LinearConstraint(param_names=("a", "b", "c"), target=1.0)
        assert c.free_params == ("a", "b")
        assert c.derived_param == "c"
        assert c.target == 1.0

    def test_coerces_list_to_tuple(self):
        c = LinearConstraint(param_names=["a", "b"], target=0.0)
        assert c.param_names == ("a", "b")
        assert isinstance(c.param_names, tuple)

    def test_rejects_too_few_names(self):
        with pytest.raises(ConstraintError, match="at least two"):
            LinearConstraint(param_names=("a",), target=1.0)

    def test_rejects_empty_names(self):
        with pytest.raises(ConstraintError, match="at least two"):
            LinearConstraint(param_names=(), target=1.0)

    def test_rejects_duplicate_names(self):
        with pytest.raises(ConstraintError, match="unique"):
            LinearConstraint(param_names=("a", "b", "a"), target=1.0)

    def test_rejects_non_string_name(self):
        with pytest.raises(ConstraintError, match="strings"):
            LinearConstraint(param_names=("a", 2), target=1.0)

    def test_rejects_non_sequence_param_names(self):
        with pytest.raises(ConstraintError, match="sequence"):
            LinearConstraint(param_names=42, target=1.0)

    def test_rejects_bool_target(self):
        with pytest.raises(ConstraintError, match="bool"):
            LinearConstraint(param_names=("a", "b"), target=True)

    def test_rejects_non_numeric_target(self):
        with pytest.raises(ConstraintError, match="number"):
            LinearConstraint(param_names=("a", "b"), target="1.0")

    def test_rejects_nan_target(self):
        with pytest.raises(ConstraintError, match="finite"):
            LinearConstraint(param_names=("a", "b"), target=math.nan)

    def test_rejects_inf_target(self):
        with pytest.raises(ConstraintError, match="finite"):
            LinearConstraint(param_names=("a", "b"), target=math.inf)

    def test_accepts_int_target(self):
        c = LinearConstraint(param_names=("a", "b"), target=1)
        assert c.target == 1


# ----------------------------------------------------------------------
# LinearConstraint.apply
# ----------------------------------------------------------------------


def _scalar_resolver(*names):
    return ConstraintResolver(
        {name: ParameterLocation(tensor_key=name, index=None) for name in names}
    )


class TestLinearConstraintApply:
    """End-to-end constraint enforcement across all paths."""

    def test_apply_scalar_path(self):
        c = LinearConstraint(("a", "b", "c"), target=1.0)
        resolver = _scalar_resolver("a", "b", "c")
        params = {
            "a": torch.tensor(0.3),
            "b": torch.tensor(0.2),
            "c": torch.tensor(0.99),  # will be overwritten
        }
        c.apply(params, resolver)
        assert params["c"].item() == pytest.approx(0.5)

    def test_apply_scalar_differentiable(self):
        c = LinearConstraint(("a", "b", "c"), target=1.0)
        resolver = _scalar_resolver("a", "b", "c")
        a = torch.tensor(0.3, requires_grad=True)
        b = torch.tensor(0.2, requires_grad=True)
        params = {"a": a, "b": b, "c": torch.tensor(0.5)}
        c.apply(params, resolver)
        # downstream loss depends on derived parameter
        loss = 4.0 * params["c"]
        loss.backward()
        # d(loss)/d(a) = d(loss)/d(c) * d(c)/d(a) = 4 * (-1) = -4
        assert a.grad.item() == pytest.approx(-4.0)
        assert b.grad.item() == pytest.approx(-4.0)

    def test_apply_1d_path(self):
        """1-D constraint: residual sector slot gets recomputed."""
        c = LinearConstraint(
            ("Household.Share", "Firm.Share", "Bank.Share"),
            target=1.0,
        )
        # Three sectors, one free per free name, last is derived.
        # tensor_key is "Share" with sector indices 0/1/2.
        resolver = ConstraintResolver(
            {
                "Household.Share": ParameterLocation("Share", (0,)),
                "Firm.Share": ParameterLocation("Share", (1,)),
                "Bank.Share": ParameterLocation("Share", (2,)),
            }
        )
        params = {"Share": torch.tensor([0.4, 0.2, 0.99])}
        c.apply(params, resolver)
        # Bank.Share = 1.0 - 0.4 - 0.2 = 0.4
        assert params["Share"][2].item() == pytest.approx(0.4)
        assert params["Share"].sum().item() == pytest.approx(1.0)

    def test_apply_1d_autograd_correctness(self):
        """The critical test: derived gradient is zero, free gradients are -dL/d(derived).

        Build a 1-D Share tensor from leaf tensors via index_put, apply
        the constraint, then take a scalar loss depending on the
        derived slot and check that autograd routes the gradient back
        to the free slots with exactly the expected sign and magnitude.
        """
        # Build Share tensor from free leaves so it tracks gradients
        free_h = torch.tensor(0.4, requires_grad=True)
        free_f = torch.tensor(0.2, requires_grad=True)
        # Derived starts as zero placeholder; apply will overwrite
        zero = torch.zeros(())
        # Compose via stack so the whole tensor is on the graph
        share = torch.stack([free_h, free_f, zero])
        params = {"Share": share}
        resolver = ConstraintResolver(
            {
                "Household.Share": ParameterLocation("Share", (0,)),
                "Firm.Share": ParameterLocation("Share", (1,)),
                "Bank.Share": ParameterLocation("Share", (2,)),
            }
        )
        c = LinearConstraint(
            ("Household.Share", "Firm.Share", "Bank.Share"), target=1.0
        )
        c.apply(params, resolver)

        # Loss depends only on the derived slot.
        loss = 3.0 * params["Share"][2]
        loss.backward()

        # d(loss)/d(free_h) = 3 * d(derived)/d(free_h) = 3 * (-1) = -3
        assert free_h.grad.item() == pytest.approx(-3.0)
        assert free_f.grad.item() == pytest.approx(-3.0)

    def test_apply_2d_path(self):
        """2-D constraint: one row's derived cell recomputed."""
        c = LinearConstraint(
            (
                "Household.Firm.Flow",
                "Household.Bank.Flow",
                "Household.Household.Flow",
            ),
            target=1.0,
        )
        resolver = ConstraintResolver(
            {
                "Household.Firm.Flow": ParameterLocation("Flow", (0, 1)),
                "Household.Bank.Flow": ParameterLocation("Flow", (0, 2)),
                "Household.Household.Flow": ParameterLocation("Flow", (0, 0)),
            }
        )
        flow = torch.zeros(3, 3)
        flow[0, 1] = 0.3
        flow[0, 2] = 0.2
        params = {"Flow": flow}
        c.apply(params, resolver)
        assert params["Flow"][0, 0].item() == pytest.approx(0.5)
        # Other rows untouched
        assert params["Flow"][1, :].sum().item() == 0.0

    def test_apply_mixed_scalar_and_indexed_free_params(self):
        """Arithmetic works even when free params have different location shapes.

        Note: higher-level verify_constraints is expected to reject
        this case because mixed-shape constraints are rarely what the
        user wants. But the low-level apply is agnostic and should not
        crash if fed one; this test documents that behavior.
        """
        c = LinearConstraint(("Global", "Local.X", "Local.Y"), target=2.0)
        resolver = ConstraintResolver(
            {
                "Global": ParameterLocation("Global", None),
                "Local.X": ParameterLocation("Local", (0,)),
                "Local.Y": ParameterLocation("Local", (1,)),
            }
        )
        params = {
            "Global": torch.tensor(1.0),
            "Local": torch.tensor([0.5, 0.0]),
        }
        c.apply(params, resolver)
        # Local[1] (derived) = 2.0 - 1.0 - 0.5 = 0.5
        assert params["Local"][1].item() == pytest.approx(0.5)

    def test_apply_unknown_name_raises(self):
        c = LinearConstraint(("a", "b", "c"), target=1.0)
        resolver = _scalar_resolver("a", "b")  # "c" missing
        params = {"a": torch.tensor(0.3), "b": torch.tensor(0.2)}
        with pytest.raises(ConstraintError, match="Unknown constraint parameter"):
            c.apply(params, resolver)

    def test_apply_returns_dict(self):
        c = LinearConstraint(("a", "b", "c"), target=1.0)
        resolver = _scalar_resolver("a", "b", "c")
        params = {
            "a": torch.tensor(0.3),
            "b": torch.tensor(0.2),
            "c": torch.tensor(0.0),
        }
        result = c.apply(params, resolver)
        assert result is params
