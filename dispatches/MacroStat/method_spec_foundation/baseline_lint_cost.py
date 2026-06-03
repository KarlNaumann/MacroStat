# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Karl Naumann-Woleske
r"""Wall-clock baseline for :func:`lint_class` on the synthetic foundation
fixtures (no real model imports). Establishes the empty ``MODELS_LINTED``
budget for downstream dispatches: a pilot retrofit must keep per-class lint
under ~100 ms.

Run: ``uv run python baseline_lint_cost.py``. Writes
``baseline_lint_cost.csv``.
"""

from __future__ import annotations

import csv
import statistics
import time
from pathlib import Path

from macrostat.causality.method_spec import lint_class, requires, writes


class _Empty:
    def step(self, t, scenario, params=None):
        pass


class _Small:
    @writes(state=("X",))
    @requires(prior=("Y",), params=("alpha",))
    def step(self, t, scenario, params=None):
        self.state["X"] = self.prior["Y"] + self.params["alpha"]


class _Wide:
    @writes(state=tuple(f"S{i}" for i in range(20)))
    @requires(
        prior=tuple(f"P{i}" for i in range(20)),
        params=tuple(f"K{i}" for i in range(20)),
    )
    def step(self, t, scenario, params=None):
        for i in range(20):
            self.state[f"S{i}"] = self.prior[f"P{i}"] + self.params[f"K{i}"]


def _measure_ms(cls, root, reps=21) -> float:
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        lint_class(cls, root=root)
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    rows = [
        {
            "fixture": "_Empty (no buffer access)",
            "median_ms": _measure_ms(_Empty, "step"),
        },
        {"fixture": "_Small (3 keys, OK)", "median_ms": _measure_ms(_Small, "step")},
        {"fixture": "_Wide (60 keys, loop)", "median_ms": _measure_ms(_Wide, "step")},
    ]
    rows.append({"fixture": "BUDGET (empty MODELS_LINTED)", "median_ms": 100.0})
    out = Path(__file__).with_suffix(".csv")
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["fixture", "median_ms"])
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {"fixture": row["fixture"], "median_ms": f"{row['median_ms']:.4f}"}
            )
    for row in rows:
        print(f"{row['fixture']:<48s}  {row['median_ms']:>8.4f} ms")
    margin = 100.0 / max(rows[2]["median_ms"], 1e-6)
    print(f"\nWide-fixture margin vs 100 ms budget: {margin:.1f}x")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
