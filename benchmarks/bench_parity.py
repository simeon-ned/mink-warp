"""CPU mink baseline + mink-warp accuracy parity harness.

mink is the oracle: this replays a deterministic scene trajectory through both
mink (CPU, single world, real qpsolvers QP with ``limits=[]``) and mink-warp
(batched normal-equations + Cholesky, world 0) on the same soft task stack. It
reports (a) the per-step tangent-velocity error — "does mink-warp agree with
mink?" — and (b) a single-environment throughput baseline for each library, so
mink's own solves/sec can be compared against the batched numbers in bench_ik.

Usage:
    uv run python benchmarks/bench_parity.py                 # panda, 300 steps
    uv run python benchmarks/bench_parity.py --steps 500 --tol 1e-3
    uv run python benchmarks/bench_parity.py --solver quadprog
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import warp as wp
from scenes import DT, SCENES

_ns = time.perf_counter_ns


def _pick_solver(requested: str | None) -> str:
    import qpsolvers
    avail = qpsolvers.available_solvers
    if not avail:
        raise SystemExit("No qpsolvers backend installed (pip install quadprog).")
    if requested:
        if requested not in avail:
            raise SystemExit(f"solver '{requested}' unavailable; have {avail}")
        return requested
    for pref in ("quadprog", "daqp", "proxqp", "osqp", "cvxopt", "scs"):
        if pref in avail:
            return pref
    return avail[0]


def run(scene_key: str, steps: int, seed: int, solver: str, warmup: int = 30) -> dict:
    import mink

    scene = SCENES[scene_key]
    if not scene.parity:
        raise SystemExit(f"scene '{scene_key}' is not parity-enabled")

    s_mw = scene.setup_mw(1, seed=seed, perturb=True)
    s_mk = scene.setup_mink(seed=seed, perturb=True)
    cfg_mw, solver_mw = s_mw["configuration"], s_mw["solver"]
    cfg_mk = s_mk["configuration"]
    damping = s_mw["damping"]

    v_err: list[float] = []
    q_err: list[float] = []
    mw_ns = 0  # mink-warp per-step wall (1 world)
    mk_ns = 0  # mink per-step wall
    t = 0.0
    for i in range(warmup + steps):
        # mink-warp control step (target upload + solve + integrate), timed.
        a = _ns()
        scene.update_mw(s_mw, t)
        v_wp = solver_mw.solve(s_mw["tasks"], DT, damping=damping)
        cfg_mw.integrate_inplace(v_wp, DT)
        wp.synchronize()
        b = _ns()

        # mink control step (same target), timed.
        target0 = s_mw["targets"][0]
        scene.update_mink(s_mk, target0)
        v_mk = mink.solve_ik(
            cfg_mk, s_mk["tasks"], DT, solver=solver, damping=damping, limits=[]
        )
        cfg_mk.integrate_inplace(v_mk, DT)
        c = _ns()

        if i >= warmup:
            mw_ns += b - a
            mk_ns += c - b
            v_mw = v_wp.numpy()[0]
            v_err.append(float(np.max(np.abs(v_mw - v_mk))))
            q_err.append(float(np.max(np.abs(cfg_mw.q.numpy()[0] - cfg_mk.q))))
        t += DT

    return {
        "v_raw": v_err,
        "q_raw": q_err,
        "mink_us": mk_ns * 1e-3 / steps,
        "mink_warp_us": mw_ns * 1e-3 / steps,
        "mink_solves_s": steps / (mk_ns * 1e-9) if mk_ns else float("inf"),
        "mink_warp_solves_s": steps / (mw_ns * 1e-9) if mw_ns else float("inf"),
    }


def main() -> None:
    import mink  # noqa: F401

    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("scene", nargs="?", default="panda",
                    choices=[k for k, s in SCENES.items() if s.parity])
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--solver", default=None)
    ap.add_argument("--tol", type=float, default=5e-3,
                    help="Max tangent-velocity abs error to PASS. Default is sized "
                         "for mink-warp's float32 solve vs mink's float64 (~1e-3 "
                         "typical peak); tighten with --tol for a float64 build.")
    args = ap.parse_args()

    solver = _pick_solver(args.solver)
    r = run(args.scene, args.steps, args.seed, solver, warmup=args.warmup)
    v_arr = np.asarray(r["v_raw"])
    q_arr = np.asarray(r["q_raw"])
    vmax, vmean = float(v_arr.max()), float(v_arr.mean())
    vp99 = float(np.percentile(v_arr, 99))
    qmax = float(q_arr.max())

    dev = str(wp.get_device())
    print(f"\n  parity: scene={args.scene} mink[{solver}] vs mink-warp[{dev}]  "
          f"steps={args.steps} dt={DT * 1000:.1f} ms")
    print(f"    tangent-velocity |Δv|  max={vmax:.3e}  mean={vmean:.3e}  p99={vp99:.3e}  [rad/s or m/s]")
    print(f"    configuration    |Δq|  max={qmax:.3e}")

    print("\n  single-env baseline (1 world):")
    print(f"    mink       (CPU, {solver}) : {r['mink_us']:9.1f} us/step   {r['mink_solves_s']:12.0f} solves/s")
    print(f"    mink-warp  ({dev})         : {r['mink_warp_us']:9.1f} us/step   {r['mink_warp_solves_s']:12.0f} solves/s")

    ok = vmax <= args.tol
    print(f"\n  {'PASS' if ok else 'FAIL'} (tol={args.tol:.1e})\n")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
