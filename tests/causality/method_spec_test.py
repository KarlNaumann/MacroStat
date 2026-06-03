# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Karl Naumann-Woleske
r"""Tests for ``macrostat.causality.method_spec``.

Synthetic Python classes only; no MacroStat model imports. Fixtures live at
module level so :func:`inspect.getsource` resolves them via the file path
(in-test class definitions fail with ``OSError: could not get source code``).
"""

from __future__ import annotations

import pytest

from macrostat.causality.method_spec import (
    _STATE_MUTATORS,
    DYNAMIC,
    DriftStatus,
    MethodSpec,
    _AccessVisitor,
    _init_attr_names,
    _method_aliases,
    _parse_method,
    _reachable_methods,
    check_non_buffer_attrs,
    counts,
    lint_class,
    register_mutator,
    requires,
    writes,
)

# ---------------------------------------------------------------------------
# Module-level fixtures (must live here so inspect.getsource works)
# ---------------------------------------------------------------------------


class _Green:
    @writes(state=("X",))
    @requires(prior=("Y",), params=("alpha",))
    def step(self, t, scenario, params=None):
        self.state["X"] = self.prior["Y"] + self.params["alpha"]


class _Drift:
    @writes(state=("X", "Y"))
    def step(self, t, scenario, params=None):
        self.state["X"] = 1
        self.state["Z"] = 3  # not declared; "Y" declared but never written


class _MissingDecorator:
    def step(self, t, scenario, params=None):
        self.state["X"] = 1


class _DynamicUnresolved:
    @writes(state=("X",))
    def step(self, t, scenario, params=None):
        self.state["X"] = 1
        key = self._lookup()
        self.state[key] = 2  # dynamic — observed gets DYNAMIC sentinel

    def _lookup(self):
        return "Y"


class _DynamicDeclared:
    @writes(state=("X", DYNAMIC))
    def step(self, t, scenario, params=None):
        self.state["X"] = 1
        key = "Y"
        self.state[key] = 2


class _AugAssignFixture:
    @writes(state=("X",))
    def step(self, t, scenario, params=None):
        self.state["X"] += 1


class _ScenarioKwarg:
    @requires(scenario=("Shock",))
    @writes(state=("X",))
    def step(self, t, scenario, params=None):
        self.state["X"] = scenario["Shock"]


class _ScenarioBuffer:
    @requires(scenario=("Shock",))
    @writes(state=("X",))
    def step(self, t, params=None):
        self.state["X"] = self.scenarios["Shock"]


class _ScenarioRenamed:
    @writes(state=("X",))
    def step(self, t, scen, params=None):
        # `scen[...]` is NOT scenario kwarg magic — wrong arg name. The walker
        # ignores it; observed picks up only the write.
        self.state["X"] = scen["Shock"] if scen else 0


class _ParamsKwarg:
    @requires(params=("alpha",))
    @writes(state=("X",))
    def step(self, t, scenario, params=None):
        self.state["X"] = params["alpha"]


class _ParamsBuffer:
    @requires(params=("alpha",))
    @writes(state=("X",))
    def step(self, t, scenario, params=None):
        self.state["X"] = self.params["alpha"]


class _HyperBuffer:
    @requires(hyper=("seed",))
    @writes(state=("X",))
    def step(self, t, scenario, params=None):
        self.state["X"] = self.hyper["seed"]


class _MutatorCaller:
    @writes(state=("X",))
    def step(self, t, scenario, params=None):
        self.mutate_x()

    def mutate_x(self):
        self.state["X"] = 5


# Inheritance fixtures
class _ParentBeh:
    @writes(state=("X",))
    def step(self, t, scenario, params=None):
        self.state["X"] = 1


class _ChildNoDecorator(_ParentBeh):
    def step(self, t, scenario, params=None):
        self.state["X"] = 2
        self.state["Y"] = 3  # extra write — child should redeclare


class _ChildDecorated(_ParentBeh):
    @writes(state=("X", "Y"))
    def step(self, t, scenario, params=None):
        self.state["X"] = 2
        self.state["Y"] = 3


# Aliases (Pichler-style)
class _MatchAliasing:
    def __init__(self, mode: str):
        match mode:
            case "a":
                self.dispatch = self.handle_a
            case "b":
                self.dispatch = self.handle_b

    def step(self, t, scenario, params=None):
        self.dispatch()

    def handle_a(self):
        self.state["A"] = 1

    def handle_b(self):
        self.state["B"] = 2


class _DirectAliasing:
    def __init__(self):
        self.do_it = self.do_real

    def step(self, t, scenario, params=None):
        self.do_it()

    def do_real(self):
        self.state["X"] = 1


# R6d fixtures
class _R6dGood:
    def __init__(self):
        self.state = {}
        self.prior = {}
        self.history = {}
        self.params = {}
        self.scenarios = {}
        self.hyper = {}
        self._cache = None
        self.lookup_table = {"a": 1}

    def step(self, t, scenario, params=None):
        self.state["X"] = self.prior["X"] + self._cache_get() + self.lookup_table["a"]

    def _cache_get(self):
        return 0


class _R6dBadRead:
    def __init__(self):
        self.state = {}

    def step(self, t, scenario, params=None):
        self.state["X"] = self.random_attr  # not bound in __init__, not _-private


class _R6dCanaryWrite:
    def __init__(self):
        self.state = {}
        self.scratch = 0  # bound in __init__

    def step(self, t, scenario, params=None):
        self.scratch = 1  # write outside __init__ — banned regardless of init


class _R6dPrivateScratch:
    def __init__(self):
        self.state = {}

    def step(self, t, scenario, params=None):
        self._scratch = 5  # underscore-private — OK
        self.state["X"] = self._scratch


# ---------------------------------------------------------------------------
# AST primitives
# ---------------------------------------------------------------------------


def test_parse_method_returns_module():
    tree = _parse_method(_Green.step)
    assert tree is not None
    assert tree.body
    fd = tree.body[0]
    assert fd.name == "step"


def test_method_aliases_direct():
    aliases = _method_aliases(_DirectAliasing)
    assert aliases == {"do_it": ["do_real"]}


def test_method_aliases_match_block():
    aliases = _method_aliases(_MatchAliasing)
    assert sorted(aliases["dispatch"]) == ["handle_a", "handle_b"]


def test_reachable_methods_follows_aliases():
    seen = _reachable_methods(_MatchAliasing, "step")
    # Both branches of the match alias should be reached.
    assert "handle_a" in seen
    assert "handle_b" in seen


def test_reachable_methods_child_override_resolves():
    seen = _reachable_methods(_ChildDecorated, "step")
    assert seen == ["step"]
    # The reached step must be the child's, not the parent's — verified by
    # checking the spec on the resolved attr.
    spec = _ChildDecorated.step.__method_spec__
    assert spec.writes_state == frozenset({"X", "Y"})


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def test_writes_accepts_list_tuple_frozenset_set():
    for value in (["X"], ("X",), frozenset({"X"}), {"X"}):

        @writes(state=value)
        def f():
            pass

        assert f.__method_spec__.writes_state == frozenset({"X"})


def test_writes_rejects_unknown_buffer():
    with pytest.raises(TypeError, match="unknown"):

        @writes(scenario=("X",))
        def f():
            pass


def test_requires_accepts_all_buffers():
    @requires(
        state=("a",),
        prior=("b",),
        history=("c",),
        params=("d",),
        scenario=("e",),
        hyper=("f",),
    )
    def f():
        pass

    spec = f.__method_spec__
    assert spec.requires_state == frozenset({"a"})
    assert spec.requires_prior == frozenset({"b"})
    assert spec.requires_history == frozenset({"c"})
    assert spec.requires_params == frozenset({"d"})
    assert spec.requires_scenario == frozenset({"e"})
    assert spec.requires_hyper == frozenset({"f"})


def test_decorator_rejects_non_str_entries():
    with pytest.raises(TypeError, match="must be str"):

        @writes(state=("X", 42))
        def f():
            pass


def test_decorator_rejects_non_iterable():
    with pytest.raises(TypeError, match="list/tuple/frozenset"):

        @writes(state="not_iterable")
        def f():
            pass


def test_stacking_merges_via_replace():
    @writes(state=("X",))
    @writes(state=("Y",))
    @requires(prior=("Z",))
    def f():
        pass

    spec = f.__method_spec__
    assert spec.writes_state == frozenset({"X", "Y"})
    assert spec.requires_prior == frozenset({"Z"})


def test_dynamic_sentinel_in_frozenset():
    @writes(state=("X", DYNAMIC))
    def f():
        pass

    assert DYNAMIC in f.__method_spec__.writes_state


def test_zero_per_call_cost_no_wrapping():
    @writes(state=("X",))
    def f(a, b):
        return a + b

    assert f(1, 2) == 3
    # Decorator must return the function unwrapped.
    assert hasattr(f, "__method_spec__")
    assert f.__name__ == "f"


# ---------------------------------------------------------------------------
# AST walker (_AccessVisitor)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture,expected_writes,expected_requires",
    [
        (_Green, {"X"}, {"prior:Y", "params:alpha"}),
        (_ScenarioKwarg, {"X"}, {"scenario:Shock"}),
        (_ScenarioBuffer, {"X"}, {"scenario:Shock"}),
        (_ParamsKwarg, {"X"}, {"params:alpha"}),
        (_ParamsBuffer, {"X"}, {"params:alpha"}),
        (_HyperBuffer, {"X"}, {"hyper:seed"}),
    ],
)
def test_visitor_buffer_recognition(fixture, expected_writes, expected_requires):
    tree = _parse_method(fixture.step)
    visitor = _AccessVisitor(tree)
    visitor.visit(tree)
    obs = visitor.observed
    assert obs.writes_state == frozenset(expected_writes)
    actual = set()
    for f in (
        "requires_state",
        "requires_prior",
        "requires_history",
        "requires_params",
        "requires_scenario",
        "requires_hyper",
    ):
        buf = f.split("_", 1)[1]
        for k in getattr(obs, f):
            actual.add(f"{buf}:{k}")
    assert actual == expected_requires


def test_visitor_renamed_scenario_arg_ignored():
    # `scen` is not the magic kwarg name; subscripts on it are not credited.
    tree = _parse_method(_ScenarioRenamed.step)
    visitor = _AccessVisitor(tree)
    visitor.visit(tree)
    assert visitor.observed.requires_scenario == frozenset()


def test_visitor_dynamic_key_inserts_sentinel():
    tree = _parse_method(_DynamicUnresolved.step)
    visitor = _AccessVisitor(tree)
    visitor.visit(tree)
    assert DYNAMIC in visitor.observed.writes_state
    assert "X" in visitor.observed.writes_state


def test_visitor_augassign_counts_as_read_and_write():
    tree = _parse_method(_AugAssignFixture.step)
    visitor = _AccessVisitor(tree)
    visitor.visit(tree)
    obs = visitor.observed
    assert "X" in obs.writes_state
    # observed strips reads of self-written keys, so requires_state is empty.
    assert obs.requires_state == frozenset()


# ---------------------------------------------------------------------------
# lint_class outcomes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture,expected_status",
    [
        (_Green, DriftStatus.OK),
        (_Drift, DriftStatus.DRIFT),
        (_MissingDecorator, DriftStatus.MISSING_DECORATOR),
        (_DynamicUnresolved, DriftStatus.DYNAMIC_UNRESOLVED),
        (_DynamicDeclared, DriftStatus.OK),
        (_ScenarioKwarg, DriftStatus.OK),
        (_HyperBuffer, DriftStatus.OK),
    ],
)
def test_lint_class_status(fixture, expected_status):
    entries = lint_class(fixture, root="step")
    statuses = [e.status for e in entries if e.method == "step"]
    assert statuses == [expected_status]


def test_lint_class_drift_missing_and_spurious():
    entries = lint_class(_Drift, root="step")
    [step_entry] = [e for e in entries if e.method == "step"]
    assert step_entry.status is DriftStatus.DRIFT
    assert step_entry.missing == frozenset({"Z"})
    assert step_entry.spurious == frozenset({"Y"})


def test_counts_returns_full_keyspace():
    entries = lint_class(_Green, root="step") + lint_class(_Drift, root="step")
    c = counts(entries)
    assert set(c.keys()) == set(DriftStatus)
    assert c[DriftStatus.OK] == 1
    assert c[DriftStatus.DRIFT] == 1
    assert c[DriftStatus.MISSING_DECORATOR] == 0


def test_truthy_check_pattern():
    entries = lint_class(_Drift, root="step")
    bad = [e for e in entries if e.status is not DriftStatus.OK]
    assert bad  # at least one non-OK


# ---------------------------------------------------------------------------
# Inheritance regression
# ---------------------------------------------------------------------------


def test_inheritance_child_without_decorator_is_missing():
    entries = lint_class(_ChildNoDecorator, root="step")
    [e] = [entry for entry in entries if entry.method == "step"]
    assert e.status is DriftStatus.MISSING_DECORATOR


def test_inheritance_child_decorated_is_ok():
    entries = lint_class(_ChildDecorated, root="step")
    [e] = [entry for entry in entries if entry.method == "step"]
    assert e.status is DriftStatus.OK


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_mutators():
    _STATE_MUTATORS.clear()
    yield
    _STATE_MUTATORS.clear()


def test_register_mutator_idempotent():
    spec = MethodSpec(writes_state=frozenset({"X"}))
    register_mutator("mutate_x", spec)
    register_mutator("mutate_x", spec)  # no raise
    assert _STATE_MUTATORS["mutate_x"] == spec


def test_register_mutator_conflict_raises():
    register_mutator("mutate_x", MethodSpec(writes_state=frozenset({"X"})))
    with pytest.raises(ValueError, match="conflicting spec"):
        register_mutator("mutate_x", MethodSpec(writes_state=frozenset({"Y"})))


def test_indirect_write_flag_set_after_registration():
    register_mutator("mutate_x", MethodSpec(writes_state=frozenset({"X"})))
    entries = lint_class(_MutatorCaller, root="step")
    [step_entry] = [e for e in entries if e.method == "step"]
    assert step_entry.indirect is True
    # `step` declares writes_state=("X",) and is credited with "X" indirectly.
    # No drift since declared matches observed; the indirect flag triggers
    # INDIRECT_WRITE classification.
    assert step_entry.status is DriftStatus.INDIRECT_WRITE


def test_unregistered_mutator_call_yields_drift():
    # Without registry, the caller's self.mutate_x() is opaque; declared
    # writes_state=("X",) becomes spurious -> DRIFT.
    entries = lint_class(_MutatorCaller, root="step")
    [step_entry] = [e for e in entries if e.method == "step"]
    assert step_entry.status is DriftStatus.DRIFT
    assert step_entry.spurious == frozenset({"X"})


# ---------------------------------------------------------------------------
# R6d
# ---------------------------------------------------------------------------


def test_r6d_skips_behavior_base():
    from macrostat.core.behavior import Behavior

    assert check_non_buffer_attrs(Behavior) == []


def test_r6d_auto_allowlist_from_init():
    names = _init_attr_names(_R6dGood)
    assert "lookup_table" in names
    assert "_cache" in names


def test_r6d_good_class_no_violations():
    assert check_non_buffer_attrs(_R6dGood) == []


def test_r6d_bad_read_violation():
    violations = check_non_buffer_attrs(_R6dBadRead)
    assert violations
    method, attr, _line, reason = violations[0]
    assert method == "step"
    assert attr == "random_attr"
    assert "read" in reason


def test_r6d_canary_write_banned():
    # `scratch` is bound in __init__ but writing to it outside __init__ is
    # still a violation under variant (b).
    violations = check_non_buffer_attrs(_R6dCanaryWrite)
    methods = {(m, a) for m, a, _line, _r in violations}
    assert ("step", "scratch") in methods


def test_r6d_underscore_private_ok():
    assert check_non_buffer_attrs(_R6dPrivateScratch) == []
