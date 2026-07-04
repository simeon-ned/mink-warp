"""Compare the IK solver backends on any scene: throughput vs tracking accuracy.

Each backend (``dls`` / ``lm`` / ``lbfgs``) replays the same moving-target
trajectory for the chosen scene (``panda`` or ``g1``). DLS takes one
Gauss-Newton step per control tick; the optimizer backends take several inner
iterations, so they track the moving target more tightly at a higher per-call
cost.

``--motion`` scales the target trajectory. ``gentle`` (default) barely moves the
target per tick, so one GN step already converges and every backend tracks
equally; ``aggressive`` moves it fast, so a single linear step lags and the
multi-iteration backends (LM / L-BFGS) track visibly tighter.

Reports, per backend:
* ``solves/s`` — batched throughput (worlds x steps / wall time)
* ``|dpos| mean/max`` — world-0 tracked-frame distance to its target [m]
alongside ``tgt/tick`` — mean world-0 target displacement per control tick [m].

Usage:
    uv run python benchmarks/bench_solvers.py                          # panda, gentle
    uv run python benchmarks/bench_solvers.py g1 --motion aggressive
    uv run python benchmarks/bench_solvers.py panda --motion aggressive --nworld 4096 --graph --device cuda:0
"""

from __future__ import annotations

import argparse
import time

import numpy as np
import warp as wp
from common import summarize, sync, throughput
from scenes import DT, SCENES

import mink_warp as mw

_ns = time.perf_counter_ns
_DEFAULT_ITERS = {"dls": 1, "lm": 2, "lbfgs": 5}
_GRAPH_CAPABLE = {"dls", "lm"}
#: (amp_scale, freq_scale) presets for the target trajectory.
_MOTION = {"gentle": (1.0, 1.0), "aggressive": (2.0, 6.0)}


def run(scene_key: str, solver_kind: str, nworld: int, steps: int, warmup: int,
        iters: int, use_graph: bool, device: str | None,
        amp_scale: float = 1.0, freq_scale: float = 1.0) -> dict:
    scene = SCENES[scene_key]
    s = scene.setup_mw(nworld, device=device)
    s["amp_scale"], s["freq_scale"] = amp_scale, freq_scale
    cfg, tasks = s["configuration"], s["tasks"]
    frame = s["frame"]  # the scene's tracked FrameTask (name/type generic)
    kw = {"damping": s["damping"]} if solver_kind == "dls" else {}
    solver = mw.make_solver(cfg, solver_kind, **kw)
    graph = (use_graph and wp.get_device(device).is_cuda
             and solver_kind in _GRAPH_CAPABLE
             and (solver_kind != "dls" or iters == 1))

    times_us: list[float] = []
    dpos: list[float] = []
    tgt_speed: list[float] = []
    prev_tgt = None
    t = 0.0
    for i in range(warmup + steps):
        t0 = _ns()
        scene.update_mw(s, t)
        solver.solve_and_integrate(tasks, DT, iterations=iters, use_graph=graph)
        sync(device)
        dt_us = (_ns() - t0) * 1e-3
        tgt = s["targets"][0, 4:7].copy()
        if i >= warmup:
            times_us.append(dt_us)
            ee = cfg.get_transform_frame_to_world(
                frame.frame_name, frame.frame_type).numpy()[0, 4:7]
            dpos.append(float(np.linalg.norm(ee - tgt)))
            if prev_tgt is not None:
                tgt_speed.append(float(np.linalg.norm(tgt - prev_tgt)))
        prev_tgt = tgt
        t += DT

    st = summarize(times_us)
    return dict(
        scene=scene_key, solver=solver_kind, iters=iters, nworld=nworld, graph=graph,
        solves_per_s=throughput(st["mean"] * 1e-6, nworld),
        us_per_solve=st["mean"] / nworld,
        dpos_mean=float(np.mean(dpos)), dpos_max=float(np.max(dpos)),
        tgt_per_tick=float(np.mean(tgt_speed)) if tgt_speed else 0.0,
    )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scene", nargs="?", default="panda", choices=list(SCENES))
    ap.add_argument("--solvers", nargs="+", default=["dls", "lm", "lbfgs"],
                    choices=["dls", "lm", "lbfgs"])
    ap.add_argument("--nworld", type=int, default=1)
    ap.add_argument("--steps", type=int, default=150)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--iters", type=int, default=None,
                    help="inner iterations (default: dls=1, lm=2, lbfgs=5).")
    ap.add_argument("--motion", default="gentle", choices=list(_MOTION),
                    help="target trajectory speed preset (default: gentle).")
    ap.add_argument("--amp-scale", type=float, default=None,
                    help="override target amplitude scale.")
    ap.add_argument("--freq-scale", type=float, default=None,
                    help="override target frequency scale.")
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    amp0, freq0 = _MOTION[args.motion]
    amp = args.amp_scale if args.amp_scale is not None else amp0
    freq = args.freq_scale if args.freq_scale is not None else freq0

    device = args.device or str(wp.get_device())
    print(f"\n  scene={args.scene}  device={device}  nworld={args.nworld}  "
          f"motion={args.motion} (amp x{amp:g}, freq x{freq:g})  "
          f"graph={args.graph}  dt={DT * 1000:.1f} ms")
    print(f"  {'solver':>6s}  {'iters':>5s}  {'solves/s':>12s}  {'us/solve':>10s}  "
          f"{'tgt/tick':>10s}  {'|dpos| mean':>12s}  {'|dpos| max':>11s}")
    print(f"  {'-' * 6}  {'-' * 5}  {'-' * 12}  {'-' * 10}  {'-' * 10}  "
          f"{'-' * 12}  {'-' * 11}")
    for kind in args.solvers:
        iters = args.iters if args.iters is not None else _DEFAULT_ITERS[kind]
        r = run(args.scene, kind, args.nworld, args.steps, args.warmup, iters,
                args.graph, args.device, amp, freq)
        print(f"  {r['solver']:>6s}  {r['iters']:>5d}  {r['solves_per_s']:>12.0f}  "
              f"{r['us_per_solve']:>10.2f}  {r['tgt_per_tick']:>10.2e}  "
              f"{r['dpos_mean']:>12.2e}  {r['dpos_max']:>11.2e}")
    print()


if __name__ == "__main__":
    main()
