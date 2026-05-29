# CLAUDE.md — MacroStat

Project-specific conventions for MacroStat. Inherits from workspace-level CLAUDE.md.

## Model Conventions

- Variable/parameter names use **descriptive CamelCase** (`InterestRateBills` not `rb`). Store textbook notation in the `notation` field.
- All computation in `behavior.py` must use **torch operations** (no numpy) for differentiability. Use `torch.where` for safe division.
- Model variants use **inheritance**: override only methods that change.
- Workflow: **implement → test → document → commit**.

## 6-File Model Structure

Models live in `src/macrostat/models/<MODEL_NAME>/` with a standard 6-file structure:

| File | Purpose |
|------|---------|
| `parameters.py` | Parameter definitions with names, values, notation |
| `variables.py` | Variable definitions with initial values, notation |
| `scenarios.py` | Scenario configurations (shocks, policy changes) |
| `behavior.py` | All model equations — torch-only computation |
| `model.py` | Model class assembling parameters, variables, behavior |
| `__init__.py` | Public API exports |

## Test Categories

Standard test suite for every model:
- **Smoke tests**: Model initializes and runs for N periods without error
- **Positivity checks**: Variables that must be positive stay positive
- **Redundant equation check**: No duplicate/unused equations
- **Steady-state properties**: Model converges (where applicable)
- **Shock responses**: Directional correctness of impulse responses
- **Accounting identities**: Balance sheet and flow consistency holds

## Testing

- **Never run the full test suite blindly.** The full suite (500+ tests) runs all model simulations and takes 10+ minutes. Always target specific tests or modules.
- **Targeted test runs:** `uv run pytest tests/models/gl06lp2_test.py -v --override-ini="addopts="` — filter to the relevant model.
- **Single test:** `uv run pytest tests/models/gl06lp2_test.py::test_wealth_accounting_identity -s --override-ini="addopts="`
- **Docs build:** The Sphinx config uses `nb_execution_mode = "off"` — notebooks are NOT re-executed during builds. If a build hangs, check whether a different `make` target or config was used.
- **`--override-ini="addopts="`** is required when running from a worktree (avoids pytest-cov errors from setup.cfg).

## Migration Notes

Kirman's Ants (SDE, 2 parameters) landed in S11. Mark0 COVID (heterogeneous-agent ABM, 5000 firms, 23 parameters) and Poledna are next.

The two prior `Variables`-class friction points are resolved (dispatch `MacroStat_HetAgentInfra`, branch `feat/hetagent-infra`):

- **Non-SFC mode**: variables without an `sfc` key are silently skipped by the balance-sheet / transaction-matrix builder and by `verify_sfc_info`. No opt-in flag — absence of `sfc` is the signal.
- **Variable-shape tensor state**: `info[k]["sectors"]` is now a shape-axis list. Each entry is either an int literal or a string resolved via `Parameters.__getitem__` (which falls through to `self.hyper`). All entries resolving to positive ints → that shape tuple. Any failure → legacy `len(sectors)` fallback for SFC-style label lists. Mark-0 / Poledna declare `sectors=["N_firms"]` etc.; SFC models with `sectors=["Households", "Firms", "Banks"]` are unchanged.

The 6-file convention is still heavyweight for simple models but acceptable for the planned ABM ports.

`Behavior.forward()` is the only caller that materializes the wide `self.timeseries` dict (single `gather_timeseries()` at end of loop). `record_state` only appends to `timeseries_list`. Consumers must not read `self.timeseries` mid-loop.

## Continuous-Time / SDE Models

`KirmansAnts` is the first continuous-time SDE model in MacroStat. Provisional pattern:

- **Sub-loop in `step()`**: outer `forward()` loops `timesteps` time units; inner numpy loop runs `int(1/dt)` Euler-Maruyama micro-steps per outer step. `forward()` is NOT overridden, preserving Scenario, parameter-shock, and `record_state` machinery.
- **`supports_differentiable = False`**: numpy SDE step + rejection sampling are not differentiable. `Behavior.__init__` raises `RuntimeError` if `differentiable=True` is passed for such a model.
- **Numpy in `behavior.py`**: the "torch-only computation" rule is relaxed for non-differentiable models. Use `self.numpy_rng` (the per-instance `np.random.Generator` seeded from `self.hyper["seed"]`) — never the global `np.random` state.
- **Inner-step recording**: model-local side-buffer (`self._micro_trajectory`) for the full micro-step trajectory; `record_state` continues to record outer values. Treat the side-buffer as model-private.

When a second SDE model lands (Heston, Mark0-stochastic, OU): refactor both into a `ContinuousTimeBehavior` subclass and a frequency-aware `Variables.record_state` that retires the side-buffer.

The `torch.Generator` / per-instance-RNG policy above also applies to heterogeneous-agent ABMs (Mark-0, Poledna): use `self.numpy_rng` / `self.torch_rng`, seed from `self.hyper["seed"]`, never touch global state. Same pattern, same justification (reproducibility across `forward()` calls and across processes).
