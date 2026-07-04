"""Batched IK throughput sweep for mink-warp.

Where mink's bench_ik measures single-world per-step latency, this measures how
many IK solves per second the batched pipeline sustains as the world count
grows. Each step does a host->device target upload + one
``solver.solve_and_integrate`` (eager, or a captured CUDA graph on GPU), with a
device sync around the timed region so wall-clock reflects real kernel time.

Usage:
    uv run python benchmarks/bench_ik.py                       # panda, default sweep
    uv run python benchmarks/bench_ik.py g1 --batches 1 64 1024
    uv run python benchmarks/bench_ik.py --graph --save gpu.json
    uv run python benchmarks/bench_ik.py --compare cpu.json gpu.json
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import warp as wp
from common import summarize, sync, throughput
from scenes import DT, SCENES

import mink_warp as mw

_ns = time.perf_counter_ns

# L-BFGS does per-candidate line-search step sizes -> no CUDA graph capture.
_GRAPH_CAPABLE = {"dls", "lm"}
_DEFAULT_ITERS = {"dls": 1, "lm": 5, "lbfgs": 5}


def run_batch(scene_key: str, nworld: int, steps: int, warmup: int,
              use_graph: bool, device: str | None,
              solver_kind: str = "dls", iters: int | None = None) -> dict:
    scene = SCENES[scene_key]
    s = scene.setup_mw(nworld, device=device)
    tasks, damping = s["tasks"], s["damping"]
    if iters is None:
        iters = _DEFAULT_ITERS[solver_kind]

    kw = {"damping": damping} if solver_kind == "dls" else {}
    solver = mw.make_solver(s["configuration"], solver_kind, **kw)

    graph = (use_graph and wp.get_device(device).is_cuda
             and solver_kind in _GRAPH_CAPABLE
             and (solver_kind != "dls" or iters == 1))
    times_us: list[float] = []
    t = 0.0
    for i in range(warmup + steps):
        t0 = _ns()
        scene.update_mw(s, t)
        solver.solve_and_integrate(tasks, DT, iterations=iters, use_graph=graph)
        sync(device)
        if i >= warmup:
            times_us.append((_ns() - t0) * 1e-3)
        t += DT

    stats = summarize(times_us)
    stats["nworld"] = nworld
    stats["per_solve_us"] = stats["mean"] / nworld
    stats["solves_per_s"] = throughput(stats["mean"] * 1e-6, nworld)
    stats["graph"] = graph
    stats["solver"] = solver_kind
    stats["iters"] = iters
    return stats


def print_row(st: dict) -> None:
    print(f"  {st['nworld']:>6d}  {st['mean']:>11.1f}  {st['median']:>11.1f}  "
          f"{st['per_solve_us']:>11.2f}  {st['solves_per_s']:>13.0f}")


def print_header(scene_key: str, device: str, graph: bool,
                 solver: str, iters: int) -> None:
    print(f"\n  scene={scene_key}  device={device}  solver={solver}  "
          f"iters={iters}  graph={graph}  dt={DT * 1000:.1f} ms")
    print(f"  {'nworld':>6s}  {'step_us':>11s}  {'median_us':>11s}  "
          f"{'us/solve':>11s}  {'solves/s':>13s}")
    print(f"  {'-' * 6}  {'-' * 11}  {'-' * 11}  {'-' * 11}  {'-' * 13}")


def print_comparison(a: dict, b: dict, la: str, lb: str) -> None:
    print(f"\n  {'nworld':>6s} {la + ' solves/s':>18s} {lb + ' solves/s':>18s} {'speedup':>9s}")
    keys = sorted(set(a) & set(b), key=int)
    for k in keys:
        x, y = a[k]["solves_per_s"], b[k]["solves_per_s"]
        sp = y / x if x else float("inf")
        print(f"  {k:>6s} {x:>18.0f} {y:>18.0f} {sp:>8.2f}x")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scene", nargs="?", default="panda", choices=list(SCENES))
    ap.add_argument("--batches", type=int, nargs="+",
                    default=[1, 16, 64, 256, 1024, 4096])
    ap.add_argument("--steps", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--solver", default="dls", choices=["dls", "lm", "lbfgs"],
                    help="IK backend (default: dls).")
    ap.add_argument("--iters", type=int, default=None,
                    help="inner iterations/call (default: dls=1, lm/lbfgs=5).")
    ap.add_argument("--graph", action="store_true", help="Capture a CUDA graph (GPU, dls/lm only).")
    ap.add_argument("--device", default=None, help="warp device, e.g. 'cuda:0' or 'cpu'.")
    ap.add_argument("--save", type=str)
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"))
    args = ap.parse_args()

    if args.compare:
        a = json.loads(Path(args.compare[0]).read_text())
        b = json.loads(Path(args.compare[1]).read_text())
        print_comparison(a, b, Path(args.compare[0]).stem, Path(args.compare[1]).stem)
        print()
        return

    device = args.device or str(wp.get_device())
    iters = args.iters if args.iters is not None else _DEFAULT_ITERS[args.solver]
    print_header(args.scene, device, args.graph, args.solver, iters)
    results: dict[str, dict] = {}
    for b in args.batches:
        st = run_batch(args.scene, b, args.steps, args.warmup, args.graph,
                       args.device, args.solver, args.iters)
        results[str(b)] = st
        print_row(st)
    print()

    if args.save:
        Path(args.save).write_text(json.dumps(results, indent=2))
        print(f"  saved -> {args.save}\n")


if __name__ == "__main__":
    main()
