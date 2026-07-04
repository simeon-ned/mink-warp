"""Benchmark + validate the constrained (box-ADMM) IK solver.

Two things this measures that the unconstrained benchmarks cannot:

* **Safety** — the maximum joint-limit *violation* over a whole trajectory,
  ``max(0, q - upper, lower - q)`` across every step / world / limited dof. For
  the constrained solver this must stay ~0 even when the target drives the arm
  hard into its limits (``--amp-scale`` cranks the motion); the unconstrained
  ``dls`` backend is shown alongside to make the contrast explicit.
* **Overhead** — throughput (solves/s, us/solve) of ``constrained`` vs ``dls``.

``--check`` sweeps ``--iters`` (ADMM count) against mink's ``daqp`` +
``ConfigurationLimit`` oracle, in lockstep (both solve from the same q each
step), reporting per-step ``|dv|`` so the default ``admm_iters`` can be picked.

Examples::

    uv run python benchmarks/bench_constrained.py panda --amp-scale 4 --device cuda:0 --graph
    uv run python benchmarks/bench_constrained.py panda --check --iters 5 10 20 40 80 --device cuda:0
    uv run python benchmarks/bench_constrained.py panda --solvers dls constrained --nworld 4096 --device cuda:0 --graph
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import common  # noqa: E402
from scenes import DT, SCENES  # noqa: E402

import mink_warp as mw  # noqa: E402

_SCALAR = None  # lazy mujoco enums


def _limited_dofs(model):
    """Return (qposadr, lower, upper) arrays for limited hinge/slide joints."""
    import mujoco

    scalar = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)
    qa, lo, hi = [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] in scalar and model.jnt_limited[j]:
            qa.append(int(model.jnt_qposadr[j]))
            lo.append(float(model.jnt_range[j][0]))
            hi.append(float(model.jnt_range[j][1]))
    return np.array(qa, dtype=int), np.array(lo), np.array(hi)


def _max_violation(q_np, qa, lo, hi):
    """Largest amount any limited dof is outside its range, over all worlds."""
    if qa.size == 0:
        return 0.0
    qv = q_np[:, qa]  # (nworld, n_limited)
    over = np.maximum(0.0, qv - hi[None, :])
    under = np.maximum(0.0, lo[None, :] - qv)
    return float(np.max(np.maximum(over, under)))


def _build_solver(kind, cfg, model, *, admm_iters, rho_scale, damping):
    if kind == "dls":
        return mw.DLSSolver(cfg, damping=damping)
    if kind == "constrained":
        return mw.ConstrainedSolver(
            cfg,
            limits=[mw.ConfigurationLimit(model)],
            admm_iters=admm_iters,
            rho_scale=rho_scale,
            damping=damping,
        )
    raise ValueError(f"unknown backend {kind!r}")


def run(scene_key, kind, *, nworld, steps, warmup, admm_iters, rho_scale,
        use_graph, device, amp_scale, freq_scale):
    scene = SCENES[scene_key]
    s = scene.setup_mw(nworld, device=device)
    s["amp_scale"] = amp_scale
    s["freq_scale"] = freq_scale
    cfg = s["configuration"]
    model = cfg.model
    tasks = s["tasks"]
    damping = s.get("damping", 1e-3)
    solver = _build_solver(kind, cfg, model, admm_iters=admm_iters,
                           rho_scale=rho_scale, damping=damping)
    qa, lo, hi = _limited_dofs(model)

    graph_ok = use_graph and kind in ("dls", "constrained")
    times_us, max_viol = [], 0.0
    for i in range(warmup + steps):
        scene.update_mw(s, i * DT)
        t0 = perf_counter_ns()
        solver.solve_and_integrate(tasks, DT, iterations=1, use_graph=graph_ok)
        common.sync(device)
        dt_us = (perf_counter_ns() - t0) * 1e-3
        if i >= warmup:
            times_us.append(dt_us)
            max_viol = max(max_viol, _max_violation(cfg.q.numpy(), qa, lo, hi))

    st = common.summarize(times_us)
    return dict(
        solver=kind, nworld=nworld, mean=st["mean"],
        solves_per_s=common.throughput(st["mean"] * 1e-6, nworld),
        us_per_solve=st["mean"] / nworld, max_violation=max_viol,
    )


def check(scene_key, iters_list, *, steps, rho_scale, device, amp_scale, freq_scale):
    """Lockstep accuracy vs mink daqp + ConfigurationLimit, per admm_iters."""
    import mink

    scene = SCENES[scene_key]
    if not scene.parity or scene.setup_mink is None:
        raise SystemExit(f"--check needs a parity scene; {scene_key!r} has no mink oracle")
    model = scene.setup_mw(1, device=device)["configuration"].model
    qa, lo, hi = _limited_dofs(model)

    print(f"# check {scene_key}: constrained(K) vs mink daqp+ConfigurationLimit, "
          f"amp_scale={amp_scale}, steps={steps}")
    print(f"{'K':>5} {'|dv| mean':>12} {'|dv| max':>12} {'max_viol':>12}")
    for K in iters_list:
        s = scene.setup_mw(1, device=device)
        s["amp_scale"], s["freq_scale"] = amp_scale, freq_scale
        cfg = s["configuration"]
        tasks = s["tasks"]
        damping = s.get("damping", 1e-3)
        solver = mw.ConstrainedSolver(
            cfg, limits=[mw.ConfigurationLimit(model)],
            admm_iters=K, rho_scale=rho_scale, damping=damping)

        mk = scene.setup_mink()
        mk_cfg, mk_tasks = mk["configuration"], mk["tasks"]
        mk_limits = [mink.ConfigurationLimit(model)]

        dvs, max_viol = [], 0.0
        for i in range(steps):
            t = i * DT
            scene.update_mw(s, t)
            scene.update_mink(mk, s["targets"][0])
            # Both solve from the SAME q (world 0 mirrors mink).
            v_wp = solver.solve(tasks, DT, damping=damping).numpy()[0]
            v_mk = mink.solve_ik(mk_cfg, mk_tasks, DT, solver="daqp",
                                 damping=damping, limits=mk_limits)
            dvs.append(float(np.linalg.norm(v_wp - v_mk)))
            # Advance both by the oracle step to stay in lockstep.
            mk_cfg.integrate_inplace(v_mk, DT)
            cfg.update(q=mk_cfg.q.astype(np.float32))
            max_viol = max(max_viol, _max_violation(cfg.q.numpy(), qa, lo, hi))
        print(f"{K:>5} {np.mean(dvs):>12.3e} {np.max(dvs):>12.3e} {max_viol:>12.3e}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", default="panda", choices=list(SCENES))
    ap.add_argument("--solvers", nargs="+", default=["dls", "constrained"])
    ap.add_argument("--nworld", type=int, default=256)
    ap.add_argument("--steps", type=int, default=120)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--iters", nargs="+", type=int, default=[30],
                    help="admm_iters (single value for run; list for --check sweep)")
    ap.add_argument("--rho-scale", type=float, default=1.0)
    ap.add_argument("--amp-scale", type=float, default=1.0)
    ap.add_argument("--freq-scale", type=float, default=1.0)
    ap.add_argument("--device", default=None)
    ap.add_argument("--graph", action="store_true")
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()

    if args.check:
        check(args.scene, args.iters, steps=args.steps, rho_scale=args.rho_scale,
              device=args.device, amp_scale=args.amp_scale, freq_scale=args.freq_scale)
        return

    K = args.iters[0]
    print(f"# {args.scene}  nworld={args.nworld}  admm_iters={K}  "
          f"rho_scale={args.rho_scale}  amp_scale={args.amp_scale}  graph={args.graph}")
    print(f"{'solver':>12} {'solves/s':>12} {'us/solve':>10} {'max_viol':>12}")
    for kind in args.solvers:
        r = run(args.scene, kind, nworld=args.nworld, steps=args.steps,
                warmup=args.warmup, admm_iters=K, rho_scale=args.rho_scale,
                use_graph=args.graph, device=args.device,
                amp_scale=args.amp_scale, freq_scale=args.freq_scale)
        print(f"{r['solver']:>12} {r['solves_per_s']:>12.0f} "
              f"{r['us_per_solve']:>10.2f} {r['max_violation']:>12.3e}")


if __name__ == "__main__":
    main()
