"""Shared utilities for the mink-warp benchmark scripts.

``summarize`` is ported verbatim from mink's benchmark harness so the two
suites report identical statistics. The batched helpers (``throughput``,
``sync``) are the mink-warp additions: mink measures single-world per-step
latency, mink-warp additionally measures how many IK solves per second the
batched GPU pipeline sustains as the world count grows.
"""

from __future__ import annotations

import statistics
from typing import Sequence

import warp as wp


def summarize(times_us: Sequence[float]) -> dict:
    """Summary statistics (in microseconds) for a list of per-step times."""
    s = sorted(times_us)
    n = len(s)

    def pct(p: float) -> float:
        return s[min(int(p * n), n - 1)]

    return {
        "n": n,
        "mean": statistics.mean(s),
        "median": statistics.median(s),
        "p95": pct(0.95),
        "p99": pct(0.99),
        "std": statistics.stdev(s) if n > 1 else 0.0,
        "min": s[0],
        "max": s[-1],
    }


def sync(device: str | None = None) -> None:
    """Block until all device work has finished (no-op-ish on CPU).

    Required before/after a timed region so wall-clock captures the actual
    kernel execution rather than just async launch overhead on CUDA.
    """
    if device is not None:
        wp.synchronize_device(device)
    else:
        wp.synchronize()


def throughput(per_step_s: float, nworld: int) -> float:
    """IK solves per second sustained by a batched step of ``nworld`` worlds."""
    return nworld / per_step_s if per_step_s > 0 else float("inf")


_FIELDS = ("mean", "median", "p95", "p99", "std", "min", "max")


def print_stats(label: str, stats: dict) -> None:
    print(f"\n  {label}  (n={stats['n']})")
    for f in _FIELDS:
        print(f"    {f:>7s}: {stats[f]:12.2f} us")
