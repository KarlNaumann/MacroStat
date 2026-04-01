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

Kirman's Ants (SDE, 2 parameters) and Mark0 COVID (heterogeneous-agent ABM, 5000 firms, 23 parameters) are planned for migration from `packages/abmstat`. Key friction points: Variables class assumes SFC structure (needs to be optional for non-SFC models), fixed-shape tensor assumption blocks variable-size agent populations, 6-file convention is heavyweight for simple models.
