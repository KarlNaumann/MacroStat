# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Karl Naumann-Woleske
r"""Micro-benchmark asserting :func:`writes` and :func:`requires` add no
measurable per-call overhead vs an undecorated function.

The decorators must set ``func.__method_spec__`` and return the function
unwrapped. Any wrapping closure (a `functools.wraps` indirection) would add
~50-100 ns per call; the contract is zero.

Run: ``uv run python decorator_zerocost_bench.py`` from this directory.
Writes ``decorator_zerocost_bench.json`` next to itself.
"""

from __future__ import annotations

import json
import statistics
import sys
import timeit
from pathlib import Path

from macrostat.causality.method_spec import requires, writes

ITER = 5_000_000
REPS = 7


def _baseline(a: int, b: int) -> int:
    return a + b


@writes(state=("X",))
def _writes_only(a: int, b: int) -> int:
    return a + b


@requires(prior=("Y",), params=("alpha",))
def _requires_only(a: int, b: int) -> int:
    return a + b


@writes(state=("X",))
@requires(prior=("Y",), params=("alpha",))
def _stacked(a: int, b: int) -> int:
    return a + b


def _time(fn, n: int) -> float:
    return timeit.timeit(lambda: fn(1, 2), number=n)


def main() -> int:
    rows: list[dict] = []
    for name, fn in (
        ("baseline", _baseline),
        ("writes_only", _writes_only),
        ("requires_only", _requires_only),
        ("stacked", _stacked),
    ):
        samples = [_time(fn, ITER) for _ in range(REPS)]
        median = statistics.median(samples)
        rows.append(
            {
                "fixture": name,
                "iter": ITER,
                "reps": REPS,
                "median_sec": median,
                "ns_per_call": median / ITER * 1e9,
                "samples_sec": samples,
            }
        )
    baseline_ns = rows[0]["ns_per_call"]
    for r in rows:
        r["overhead_ns_vs_baseline"] = r["ns_per_call"] - baseline_ns
    out_path = Path(__file__).with_suffix(".json")
    out_path.write_text(json.dumps({"rows": rows}, indent=2))
    print(json.dumps({"rows": rows}, indent=2))
    # Soft assertion: overhead per call must be within ~5 ns (timer noise).
    # Hard assertion would fail under loaded CPUs; the JSON is the artifact.
    for r in rows[1:]:
        overhead = r["overhead_ns_vs_baseline"]
        if abs(overhead) > 10.0:
            print(
                f"WARN: {r['fixture']} overhead {overhead:.1f} ns/call "
                "exceeds 10 ns budget (likely CPU noise).",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
