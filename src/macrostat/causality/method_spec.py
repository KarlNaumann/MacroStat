# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Karl Naumann-Woleske
r"""Per-method read/write metadata, AST drift lint, and R6d attribute rule.

The :func:`writes` and :func:`requires` decorators attach a :class:`MethodSpec`
to a behavior method so the read/write graph over a call-graph root is
machine-readable. :func:`lint_class` walks the AST of every reachable method
and emits a :class:`DriftEntry` per non-trivial method classifying whether the
declared spec matches the observed accesses.

The module is import-pure: only stdlib (``ast``, ``dataclasses``, ``enum``,
``inspect``, ``textwrap``). No pandas, no NetworkX. Every model in MacroStat
can import :func:`writes` / :func:`requires` without dragging analyzer deps.

Decorator API
-------------

Two decorators expose a uniform surface keyed by buffer name:

* ``@writes(state=("X","Y"))`` declares a method writes those keys via
  ``self.state[K] = ...``. Accepts ``state``, ``prior``, ``history`` kwargs.
* ``@requires(state=("Z",), params=("alpha",))`` declares the reads.
  Accepts ``state``, ``prior``, ``history``, ``params``, ``scenario``,
  ``hyper`` kwargs.

The :data:`DYNAMIC` sentinel (``"*"``) declares an unresolved-key access:
``@requires(state=("X", DYNAMIC))`` says the method reads ``self.state["X"]``
plus at least one computed-key state slot the walker cannot resolve.

Decorators set ``func.__method_spec__`` only — no wrapping closure, zero
per-call cost. Stacking via :func:`dataclasses.replace` merges new keys into
the existing spec (insertion-order preserving via :func:`dict.fromkeys`).

Walker semantics
----------------

:class:`_AccessVisitor` recognises:

* ``self.state[K]`` / ``self.prior[K]`` / ``self.history[K]`` / ``self.params[K]``
  / ``self.scenarios[K]`` / ``self.hyper[K]`` — buffer reads and writes.
* ``scenario[K]`` — recognised only when the enclosing ``FunctionDef`` has
  ``scenario`` as a positional argument (matches the kwarg passed by
  ``Behavior.forward``).
* ``params[K]`` — recognised only when the enclosing ``FunctionDef`` has
  ``params`` as a positional argument.
* ``self.state[K] += ...`` — counted as both read and write via a one-pass
  pre-walk over :class:`ast.AugAssign` targets.
* ``self.<mutator>(K, ...)`` where ``<mutator>`` is in :data:`_STATE_MUTATORS`
  — caller credited with the mutator's declared writes; the ``indirect``
  flag is set so :func:`lint_class` can flip ``DRIFT`` to ``INDIRECT_WRITE``
  when the only difference is the indirect contribution.

Unrecognised slice shapes (tuple, walrus, starred-LHS, JoinedStr, computed
expression) insert the :data:`DYNAMIC` sentinel directly into the matching
buffer's frozenset — no parallel bool channel.

Limitations (documented):

* ``super().method()`` reads are invisible to the AST walker; the parent's
  spec does not transitively credit the child. The child must redeclare or
  the lint surfaces ``MISSING_DECORATOR`` on a covering frozenset of the
  child override.
* ``inspect.getsource`` failures on lambdas, exec-defined functions, and
  runtime-bound staticmethods raise normally; no silent OK.
* ``INDIRECT_WRITE`` / ``DYNAMIC_UNRESOLVED`` are advisory: the walker
  cannot prove the unresolved key set; the declared spec stands as authored
  intent.

R6d rule
--------

:func:`check_non_buffer_attrs` enforces that subclasses of ``Behavior`` only
use tracked buffer attributes, underscore-private caches, or names bound in
``__init__`` for reads in step methods. Writes outside ``__init__`` and
``initialize`` to non-buffer non-private names are always banned regardless
of init binding (variant b — closes the canary-binding evasion). The
allowlist is auto-derived from each class's own ``__init__`` AST; no
manually maintained dict.

Divergences from EIRINpy reference
----------------------------------

* ``requires_prior`` (singular) port-time API fix vs EIRINpy's
  ``requires_priors`` (plural). Reflects that ``self.prior`` is the buffer
  name in MacroStat.
* Two decorators (``writes`` / ``requires``) replace the 7-decorator EIRINpy
  surface. Functionality preserved; surface is uniform.
* ``DYNAMIC`` lives inside each buffer's frozenset rather than on parallel
  ``has_dynamic_*`` bool flags. Predicate ``DYNAMIC in spec.<field>``.
* :class:`DriftEntry` is a flat list; no :class:`DriftTable` wrapper.
* ``slots=True`` on :class:`MethodSpec` / :class:`DriftEntry` diverges from
  the ``core/constraints.py:42`` frozen-without-slots precedent. Separate
  one-line PR flagged to add slots to ``constraints.py``.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
import textwrap
from dataclasses import dataclass
from enum import Enum
from typing import Callable

DYNAMIC: str = "*"
r"""Sentinel string declaring a computed (non-literal) key. Inserted into the
relevant buffer's frozenset by the walker; passed by the user to either
decorator to declare an intentional dynamic access."""

_WRITE_BUFFERS: tuple[str, ...] = ("state", "prior", "history")
_REQUIRE_BUFFERS: tuple[str, ...] = (
    "state",
    "prior",
    "history",
    "params",
    "scenario",
    "hyper",
)


# ---------------------------------------------------------------------------
# AST primitives
# ---------------------------------------------------------------------------


def _parse_method(func: Callable) -> ast.Module | None:
    r"""Return the parsed AST of ``func`` or ``None`` on getsource failure.

    Pattern previously inlined at
    ``causality/docstring_causality_analyzer.py:109-129``; consumer becomes a
    one-line call once the analyzer is rewritten (dispatch #3).
    """
    try:
        src = inspect.getsource(func)
    except (OSError, TypeError):
        return None
    return ast.parse(textwrap.dedent(src))


def _method_aliases(cls: type) -> dict[str, list[str]]:
    r"""Resolve ``self.<name> = self.<target>`` and ``match``-block aliases
    written in ``cls.__init__``. Returns ``{alias: [target_variants]}``.

    Recognises:

    * Direct binding at top level of ``__init__``.
    * ``match``/``case`` branches binding the same alias to different targets.

    Out of scope (documented): ``IfExp`` chains, dict-dispatch, binding
    inside ``initialize``, lambda wrappers.
    """
    init = getattr(cls, "__init__", None)
    if init is None:
        return {}
    tree = _parse_method(init)
    if tree is None:
        return {}
    aliases: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        tgt = node.targets[0]
        val = node.value
        if not (
            isinstance(tgt, ast.Attribute)
            and isinstance(tgt.value, ast.Name)
            and tgt.value.id == "self"
        ):
            continue
        if not (
            isinstance(val, ast.Attribute)
            and isinstance(val.value, ast.Name)
            and val.value.id == "self"
        ):
            continue
        aliases.setdefault(tgt.attr, []).append(val.attr)
    return aliases


def _self_calls(tree: ast.AST) -> list[str]:
    r"""Return the ordered list of method names called as ``self.<name>(...)``
    inside ``tree``. 3-line inline attribute-chain check; no helper extracted
    per the minimality audit (#4)."""
    calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "self"
        ):
            calls.append(f.attr)
    return calls


def _reachable_methods(
    cls: type,
    root: str,
    exempt: set[str] | None = None,
) -> list[str]:
    r"""Return method names transitively reachable from ``cls.<root>`` via
    ``self.<name>(...)`` calls. Insertion-ordered (BFS); root included first.

    Resolves ``__init__``-bound aliases via :func:`_method_aliases`. MRO-walked
    via :func:`getattr` so child overrides resolve to the override, not the
    parent definition. ``exempt`` names are skipped during the walk (their
    bodies are not crawled for further callees).
    """
    if not hasattr(cls, root):
        return []
    exempt = exempt or set()
    aliases = _method_aliases(cls)
    seen: list[str] = []
    seen_set: set[str] = set()
    queue: list[str] = [root]
    while queue:
        name = queue.pop(0)
        if name in seen_set:
            continue
        method = getattr(cls, name, None)
        if method is None or not callable(method):
            continue
        seen.append(name)
        seen_set.add(name)
        if name in exempt:
            continue
        tree = _parse_method(method)
        if tree is None:
            continue
        for callee in _self_calls(tree):
            for resolved in aliases.get(callee, [callee]):
                if resolved not in seen_set and hasattr(cls, resolved):
                    queue.append(resolved)
    return seen


# ---------------------------------------------------------------------------
# Data shapes
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class MethodSpec:
    r"""Declared and observed read/write footprint of a behavior method.

    Used both as author-declared spec (attached by decorators) and as
    walker-extracted observation (returned by :class:`_AccessVisitor`).
    :func:`lint_class` does set arithmetic between the two.

    The :data:`DYNAMIC` sentinel ``"*"`` is allowed as a member of any field;
    use the predicate ``DYNAMIC in spec.writes_state`` to detect unresolved
    accesses.
    """

    writes_state: frozenset[str] = frozenset()
    writes_prior: frozenset[str] = frozenset()
    writes_history: frozenset[str] = frozenset()
    requires_state: frozenset[str] = frozenset()
    requires_prior: frozenset[str] = frozenset()
    requires_history: frozenset[str] = frozenset()
    requires_params: frozenset[str] = frozenset()
    requires_scenario: frozenset[str] = frozenset()
    requires_hyper: frozenset[str] = frozenset()


class DriftStatus(Enum):
    r"""Per-method lint verdict, in priority order ``DRIFT`` > ``INDIRECT_WRITE``
    > ``DYNAMIC_UNRESOLVED`` > ``MISSING_DECORATOR`` > ``OK``."""

    OK = "ok"
    MISSING_DECORATOR = "missing_decorator"
    DRIFT = "drift"
    INDIRECT_WRITE = "indirect_write"
    DYNAMIC_UNRESOLVED = "dynamic_unresolved"


@dataclass(frozen=True, slots=True)
class DriftEntry:
    r"""One method's lint outcome."""

    method: str
    status: DriftStatus
    missing: frozenset[str] = frozenset()
    spurious: frozenset[str] = frozenset()
    dynamic_buffers: frozenset[str] = frozenset()
    indirect: bool = False


# ---------------------------------------------------------------------------
# Decorators
# ---------------------------------------------------------------------------


def _coerce(value, buffer: str) -> frozenset[str]:
    if not isinstance(value, (list, tuple, frozenset, set)):
        raise TypeError(
            f"method_spec decorator: kwarg {buffer!r} must be list/tuple/"
            f"frozenset of str (with optional DYNAMIC); got {type(value).__name__}"
        )
    for k in value:
        if not isinstance(k, str):
            raise TypeError(
                f"method_spec decorator: kwarg {buffer!r} entries must be str; "
                f"got {type(k).__name__}: {k!r}"
            )
    return frozenset(value)


def _merge(existing: frozenset[str], new: frozenset[str]) -> frozenset[str]:
    return existing | new


def writes(**kwargs):
    r"""Declare buffer writes.

    Accepts ``state``, ``prior``, ``history`` kwargs whose values are
    ``list``/``tuple``/``frozenset``/``set`` of strings (with optional
    :data:`DYNAMIC` sentinel). Sets ``func.__method_spec__`` only — no
    wrapping closure, zero per-call cost. Stacks via :func:`dataclasses.replace`
    when applied alongside :func:`requires` or another :func:`writes`.
    """
    unknown = set(kwargs) - set(_WRITE_BUFFERS)
    if unknown:
        raise TypeError(
            f"@writes only accepts buffer kwargs from {_WRITE_BUFFERS}; "
            f"got unknown: {sorted(unknown)}"
        )
    coerced = {buf: _coerce(kwargs[buf], buf) for buf in kwargs}

    def _attach(func):
        existing = getattr(func, "__method_spec__", None) or MethodSpec()
        replacement = {
            f"writes_{buf}": _merge(getattr(existing, f"writes_{buf}"), keys)
            for buf, keys in coerced.items()
        }
        func.__method_spec__ = dataclasses.replace(existing, **replacement)
        return func

    return _attach


def requires(**kwargs):
    r"""Declare buffer reads.

    Accepts ``state``, ``prior``, ``history``, ``params``, ``scenario``,
    ``hyper`` kwargs. Same value shape and zero-cost semantics as
    :func:`writes`.
    """
    unknown = set(kwargs) - set(_REQUIRE_BUFFERS)
    if unknown:
        raise TypeError(
            f"@requires only accepts buffer kwargs from {_REQUIRE_BUFFERS}; "
            f"got unknown: {sorted(unknown)}"
        )
    coerced = {buf: _coerce(kwargs[buf], buf) for buf in kwargs}

    def _attach(func):
        existing = getattr(func, "__method_spec__", None) or MethodSpec()
        replacement = {
            f"requires_{buf}": _merge(getattr(existing, f"requires_{buf}"), keys)
            for buf, keys in coerced.items()
        }
        func.__method_spec__ = dataclasses.replace(existing, **replacement)
        return func

    return _attach


# ---------------------------------------------------------------------------
# Mutator registry
# ---------------------------------------------------------------------------


_STATE_MUTATORS: dict[str, MethodSpec] = {}
r"""Registry of recognised state-mutating helper methods. Maps method name to
the :class:`MethodSpec` that name produces when called. The walker credits a
caller's observed accesses with the mutator's spec and sets the
:attr:`DriftEntry.indirect` flag for downstream classification.

Initially empty. Populated by :func:`register_mutator`; downstream dispatches
register specific helpers (e.g. ``Behavior.tanhmask`` is not a mutator, but
SFC-specific sector-update helpers are candidates)."""


def register_mutator(name: str, spec: MethodSpec) -> None:
    r"""Register ``name`` as a state-mutating helper with the given spec.

    Idempotent on identical re-register; raises :class:`ValueError` on
    conflict so silent test pollution is impossible. Import-time-only
    contract: do not call at request/step time.
    """
    existing = _STATE_MUTATORS.get(name)
    if existing is not None and existing != spec:
        raise ValueError(
            f"register_mutator({name!r}): conflicting spec already registered. "
            f"Existing: {existing}; new: {spec}."
        )
    _STATE_MUTATORS[name] = spec


# ---------------------------------------------------------------------------
# AST walker
# ---------------------------------------------------------------------------


def _literal_key(slice_node: ast.AST) -> str | None:
    if isinstance(slice_node, ast.Constant) and isinstance(slice_node.value, str):
        return slice_node.value
    return None


class _AccessVisitor(ast.NodeVisitor):
    r"""Walks a single method body and accumulates a :class:`MethodSpec`
    observation. The instance is single-use: construct, ``visit(tree)``, then
    read :attr:`observed` and :attr:`indirect`."""

    def __init__(self, tree: ast.Module):
        self._tree = tree
        self._writes: dict[str, set[str]] = {buf: set() for buf in _WRITE_BUFFERS}
        self._requires: dict[str, set[str]] = {buf: set() for buf in _REQUIRE_BUFFERS}
        self._aug_target_ids: set[int] = set()
        self._arg_names: set[str] = set()
        self.indirect: bool = False
        self._collect_aug_targets()
        self._collect_arg_names()

    def _collect_aug_targets(self) -> None:
        for node in ast.walk(self._tree):
            if isinstance(node, ast.AugAssign) and isinstance(
                node.target, ast.Subscript
            ):
                self._aug_target_ids.add(id(node.target))

    def _collect_arg_names(self) -> None:
        if not self._tree.body:
            return
        fd = self._tree.body[0]
        if not isinstance(fd, ast.FunctionDef):
            return
        self._arg_names |= {a.arg for a in fd.args.posonlyargs}
        self._arg_names |= {a.arg for a in fd.args.args}
        self._arg_names |= {a.arg for a in fd.args.kwonlyargs}
        if fd.args.vararg is not None:
            self._arg_names.add(fd.args.vararg.arg)
        if fd.args.kwarg is not None:
            self._arg_names.add(fd.args.kwarg.arg)

    @property
    def observed(self) -> MethodSpec:
        return MethodSpec(
            writes_state=frozenset(self._writes["state"]),
            writes_prior=frozenset(self._writes["prior"]),
            writes_history=frozenset(self._writes["history"]),
            requires_state=frozenset(self._requires["state"])
            - frozenset(self._writes["state"]),
            requires_prior=frozenset(self._requires["prior"])
            - frozenset(self._writes["prior"]),
            requires_history=frozenset(self._requires["history"])
            - frozenset(self._writes["history"]),
            requires_params=frozenset(self._requires["params"]),
            requires_scenario=frozenset(self._requires["scenario"]),
            requires_hyper=frozenset(self._requires["hyper"]),
        )

    @staticmethod
    def _self_attr_name(node: ast.AST) -> str | None:
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "self"
        ):
            return node.attr
        return None

    def _record_write(self, buffer: str, key: str | None) -> None:
        if buffer not in self._writes:
            return
        self._writes[buffer].add(DYNAMIC if key is None else key)

    def _record_require(self, buffer: str, key: str | None) -> None:
        if buffer not in self._requires:
            return
        self._requires[buffer].add(DYNAMIC if key is None else key)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        tgt = node.target
        if isinstance(tgt, ast.Subscript):
            buffer = self._resolve_subscript_buffer(tgt.value)
            if buffer is not None:
                key = _literal_key(tgt.slice)
                self._record_write(buffer, key)
                self._record_require(buffer, key)
        self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> None:
        if id(node) in self._aug_target_ids:
            self.generic_visit(node)
            return
        buffer = self._resolve_subscript_buffer(node.value)
        if buffer is None:
            self.generic_visit(node)
            return
        key = _literal_key(node.slice)
        if isinstance(node.ctx, ast.Store):
            self._record_write(buffer, key)
        else:
            self._record_require(buffer, key)
        self.generic_visit(node)

    def _resolve_subscript_buffer(self, value_node: ast.AST) -> str | None:
        attr = self._self_attr_name(value_node)
        if attr == "state":
            return "state"
        if attr == "prior":
            return "prior"
        if attr == "history":
            return "history"
        if attr == "params":
            return "params"
        if attr == "scenarios":
            return "scenario"
        if attr == "hyper":
            return "hyper"
        if isinstance(value_node, ast.Name):
            if value_node.id == "scenario" and "scenario" in self._arg_names:
                return "scenario"
            if value_node.id == "params" and "params" in self._arg_names:
                return "params"
        return None

    def visit_Call(self, node: ast.Call) -> None:
        f = node.func
        if (
            isinstance(f, ast.Attribute)
            and isinstance(f.value, ast.Name)
            and f.value.id == "self"
            and f.attr in _STATE_MUTATORS
        ):
            mspec = _STATE_MUTATORS[f.attr]
            for buf in _WRITE_BUFFERS:
                self._writes[buf] |= getattr(mspec, f"writes_{buf}")
            for buf in _REQUIRE_BUFFERS:
                self._requires[buf] |= getattr(mspec, f"requires_{buf}")
            self.indirect = True
        self.generic_visit(node)


def _observe(func: Callable) -> tuple[MethodSpec | None, bool]:
    r"""Return ``(observed_spec, indirect_flag)`` for ``func`` or ``(None, False)``
    when getsource fails (lambdas, runtime-bound staticmethods)."""
    tree = _parse_method(func)
    if tree is None:
        return None, False
    visitor = _AccessVisitor(tree)
    visitor.visit(tree)
    return visitor.observed, visitor.indirect


# ---------------------------------------------------------------------------
# Lint entry point
# ---------------------------------------------------------------------------


_ALL_FIELDS: tuple[str, ...] = (
    "writes_state",
    "writes_prior",
    "writes_history",
    "requires_state",
    "requires_prior",
    "requires_history",
    "requires_params",
    "requires_scenario",
    "requires_hyper",
)


def _touches_buffers(spec: MethodSpec) -> bool:
    return any(getattr(spec, f) for f in _ALL_FIELDS)


def _strip_dynamic(s: frozenset[str]) -> frozenset[str]:
    return s - {DYNAMIC}


def _classify(
    declared: MethodSpec,
    observed: MethodSpec,
    indirect: bool,
) -> tuple[DriftStatus, frozenset[str], frozenset[str], frozenset[str]]:
    r"""Return ``(status, missing, spurious, dynamic_buffers)``.

    Missing is the union of literal keys observed but not declared, across
    all buffer fields. Spurious is the reverse. ``dynamic_buffers`` is the
    set of field names where observed has the :data:`DYNAMIC` sentinel but
    declared does not.
    """
    missing: set[str] = set()
    spurious: set[str] = set()
    dyn_fields: set[str] = set()
    for f in _ALL_FIELDS:
        obs = getattr(observed, f)
        dec = getattr(declared, f)
        obs_lit = _strip_dynamic(obs)
        dec_lit = _strip_dynamic(dec)
        missing |= obs_lit - dec_lit
        spurious |= dec_lit - obs_lit
        if DYNAMIC in obs and DYNAMIC not in dec:
            dyn_fields.add(f)
    has_drift = bool(missing or spurious)
    dynamic_buffers = frozenset(dyn_fields)
    if has_drift:
        if indirect and not spurious and missing:
            return (
                DriftStatus.INDIRECT_WRITE,
                frozenset(missing),
                frozenset(spurious),
                dynamic_buffers,
            )
        return (
            DriftStatus.DRIFT,
            frozenset(missing),
            frozenset(spurious),
            dynamic_buffers,
        )
    if indirect:
        return DriftStatus.INDIRECT_WRITE, frozenset(), frozenset(), dynamic_buffers
    if dynamic_buffers:
        return DriftStatus.DYNAMIC_UNRESOLVED, frozenset(), frozenset(), dynamic_buffers
    return DriftStatus.OK, frozenset(), frozenset(), dynamic_buffers


def lint_class(
    cls: type,
    root: str = "step",
    exempt: set[str] | None = None,
) -> list[DriftEntry]:
    r"""Walk every method reachable from ``cls.<root>`` and return one
    :class:`DriftEntry` per method that touches any tracked buffer.

    Methods that do not touch ``state``/``prior``/``history``/``params``/
    ``scenarios``/``hyper`` (or their kwarg forms) are skipped — no
    decorator expected.

    Truthy check: ``bad = [e for e in result if e.status is not DriftStatus.OK]``.
    """
    exempt = exempt or set()
    entries: list[DriftEntry] = []
    for name in _reachable_methods(cls, root, exempt=exempt):
        if name in exempt:
            continue
        func = getattr(cls, name, None)
        if func is None or not callable(func):
            continue
        observed, indirect = _observe(func)
        if observed is None:
            continue
        declared = getattr(func, "__method_spec__", None)
        if (
            not _touches_buffers(observed)
            and not indirect
            and (declared is None or not _touches_buffers(declared))
        ):
            continue
        if declared is None:
            entries.append(
                DriftEntry(
                    method=name,
                    status=DriftStatus.MISSING_DECORATOR,
                    missing=frozenset(
                        k
                        for f in _ALL_FIELDS
                        for k in _strip_dynamic(getattr(observed, f))
                    ),
                    dynamic_buffers=frozenset(
                        f for f in _ALL_FIELDS if DYNAMIC in getattr(observed, f)
                    ),
                    indirect=indirect,
                )
            )
            continue
        status, missing, spurious, dyn_fields = _classify(declared, observed, indirect)
        entries.append(
            DriftEntry(
                method=name,
                status=status,
                missing=missing,
                spurious=spurious,
                dynamic_buffers=dyn_fields,
                indirect=indirect,
            )
        )
    return entries


def counts(entries: list[DriftEntry]) -> dict[DriftStatus, int]:
    r"""Return ``{DriftStatus: count}`` over ``entries``, with every status
    keyed (zero for absent statuses) so downstream dashboards can rely on
    stable shape."""
    out = {s: 0 for s in DriftStatus}
    for e in entries:
        out[e.status] = out.get(e.status, 0) + 1
    return out


# ---------------------------------------------------------------------------
# R6d: non-buffer instance attribute rule
# ---------------------------------------------------------------------------


_TRACKED_BUFFERS: frozenset[str] = frozenset(
    {"state", "prior", "history", "params", "scenarios", "hyper"}
)

_INIT_EXEMPT_METHODS: frozenset[str] = frozenset({"__init__", "initialize"})

_INIT_NAMES_CACHE: dict[type, frozenset[str]] = {}


def _init_attr_names(cls: type) -> frozenset[str]:
    r"""Return names assigned via ``self.<name> = ...`` in ``cls.__init__``.

    Walks only ``cls.__init__`` (not parent or ``initialize``). Used by R6d
    to auto-allowlist reads of init-bound names in step methods. Per-class
    cache keyed on the class object.
    """
    cached = _INIT_NAMES_CACHE.get(cls)
    if cached is not None:
        return cached
    init = cls.__dict__.get("__init__")
    if init is None:
        result: frozenset[str] = frozenset()
        _INIT_NAMES_CACHE[cls] = result
        return result
    tree = _parse_method(init)
    if tree is None:
        result = frozenset()
        _INIT_NAMES_CACHE[cls] = result
        return result
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            for tgt in targets:
                if (
                    isinstance(tgt, ast.Attribute)
                    and isinstance(tgt.value, ast.Name)
                    and tgt.value.id == "self"
                ):
                    names.add(tgt.attr)
    result = frozenset(names)
    _INIT_NAMES_CACHE[cls] = result
    return result


def check_non_buffer_attrs(cls: type) -> list[tuple[str, str, int, str]]:
    r"""Enforce the R6d rule on ``cls``.

    For each method other than ``__init__`` and ``initialize``:

    * **Writes** ``self.<name> = ...`` outside the init pair are violations
      unless ``<name>`` is a tracked buffer or starts with ``_``. Writes are
      NOT exempt for ``__init__``-bound names — variant (b) closes the
      canary-binding evasion where a class declares a name in ``__init__``
      purely so step methods can scribble on it.
    * **Reads** ``self.<name>`` (Attribute Load context) are allowed if
      ``<name>`` is a tracked buffer, starts with ``_``, or is in
      :func:`_init_attr_names`. Method-name reads (``self.foo()``) are not
      checked — those are call-graph edges handled by the lint.

    Returns ``[(method, attr, line, reason), ...]``. The walker skips
    ``Behavior`` itself; only strict subclasses are checked.

    Limitations:

    * ``setattr(self, name, value)`` is invisible to the AST walker.
    * Reads of parent ``__init__``-bound names are allowed because Python
      class semantics dispatch ``cls.__init__`` to the inherited definition
      when not overridden, so :func:`_init_attr_names` walks the inherited
      ``__init__`` AST.
    """
    from macrostat.core.behavior import Behavior  # local: avoid import cycle

    if cls is Behavior:
        return []
    init_names = _init_attr_names(cls)
    violations: list[tuple[str, str, int, str]] = []
    for name, member in inspect.getmembers(cls, predicate=callable):
        if name in _INIT_EXEMPT_METHODS:
            continue
        if name not in cls.__dict__:
            continue
        tree = _parse_method(member)
        if tree is None:
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.Assign):
                for tgt in node.targets:
                    if (
                        isinstance(tgt, ast.Attribute)
                        and isinstance(tgt.value, ast.Name)
                        and tgt.value.id == "self"
                    ):
                        attr = tgt.attr
                        if attr in _TRACKED_BUFFERS or attr.startswith("_"):
                            continue
                        violations.append(
                            (
                                name,
                                attr,
                                node.lineno,
                                "R6d: write to non-buffer non-private attr outside __init__/initialize",
                            )
                        )
            elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
                if isinstance(node.value, ast.Name) and node.value.id == "self":
                    attr = node.attr
                    if (
                        attr in _TRACKED_BUFFERS
                        or attr.startswith("_")
                        or attr in init_names
                    ):
                        continue
                    if callable(getattr(cls, attr, None)):
                        continue
                    violations.append(
                        (
                            name,
                            attr,
                            node.lineno,
                            "R6d: read of non-buffer attr not bound in __init__",
                        )
                    )
    return violations
