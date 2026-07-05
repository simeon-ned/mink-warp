"""Benchmark + validate the closed-kinematics and collision-avoidance tasks.

Two feature families land on ``feat/tasks`` whose hot path is *host MuJoCo*
(not the GPU solve), so they need their own harness:

* **Closed kinematics** — :class:`EqualityConstraintTask` (Cassie four-bar
  loops). Per world it runs host ``mj_*`` + reads ``efc_pos`` / ``efc_J``.
* **Collision avoidance** — :class:`CollisionAvoidanceLimit` (dual iiwa
  self-collision). Per world / per geom pair it runs host ``mj_geomDistance`` +
  ``mj_jac`` to build the ``G dq <= h`` rows.

What this measures that the soft-task benchmarks cannot:

* **Throughput** — solves/s, us/solve vs world count.
* **Component split** — how much of a step is spent in the host task/limit
  assembly (the part we optimize) vs the GPU solve. This is the number to push
  down without moving the accuracy.
* **Accuracy** — ``--check`` replays the trajectory through mink (world 0) and
  reports the error/Jacobian (equality) or active-row ``G/h`` (collision) parity,
  so an optimization that changes the numbers is caught immediately.

Examples::

    uv run python benchmarks/bench_tasks.py                       # both, sweep
    uv run python benchmarks/bench_tasks.py cassie --profile
    uv run python benchmarks/bench_tasks.py dual_iiwa --nworld 256 --profile
    uv run python benchmarks/bench_tasks.py cassie --check
    uv run python benchmarks/bench_tasks.py dual_iiwa --check
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path
from time import perf_counter_ns

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import common  # noqa: E402

import mink_warp as mw  # noqa: E402

_HERE = Path(__file__).parent
_EXAMPLES = _HERE.parent / "examples"
DT = 0.01

_ARM_HOME_Q = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.0], dtype=np.float64)


# ---------------------------------------------------------------------------
# Cassie: closed four-bar chains regulated by EqualityConstraintTask.
# ---------------------------------------------------------------------------


def _cassie_model() -> mujoco.MjModel:
    xml = _EXAMPLES / "agility_cassie" / "scene.xml"
    return mujoco.MjModel.from_xml_path(xml.as_posix())


def build_cassie(nworld: int, device: str | None):
    model = _cassie_model()
    cfg = mw.Configuration(model, nworld=nworld, device=device)
    cfg.update_from_keyframe("home")

    pelvis = mw.FrameTask("cassie-pelvis", "body", position_cost=0.0,
                          orientation_cost=10.0)
    posture = mw.PostureTask(model, cost=1.0)
    com = mw.ComTask(cost=200.0)
    equality = mw.EqualityConstraintTask(model=model, cost=500.0, gain=0.5,
                                         lm_damping=1e-3)
    feet = ["left-foot", "right-foot"]
    feet_tasks = [mw.FrameTask(f, "body", position_cost=200.0,
                               orientation_cost=10.0, lm_damping=1.0) for f in feet]
    tasks = [pelvis, posture, com, equality, *feet_tasks]

    posture.set_target_from_configuration(cfg)
    pelvis.set_target_from_configuration(cfg)
    for ft in feet_tasks:
        ft.set_target_from_configuration(cfg)
    base_com = cfg.wp_data.subtree_com.numpy()[:, 1, :].copy()
    com.set_target(base_com, configuration=cfg)
    phase = np.arange(nworld) * (2.0 * math.pi / max(nworld, 1))

    solver = mw.IKSolver(cfg)

    def update(t: float) -> None:
        tg = base_com.copy()
        tg[:, 2] += 0.15 * np.sin(2.0 * math.pi * 0.35 * t + phase)
        com.set_target(tg, configuration=cfg)

    return dict(cfg=cfg, model=model, tasks=tasks, solver=solver, update=update,
                hot_task=equality, damping=1e-1)


# ---------------------------------------------------------------------------
# Dual iiwa: inter-arm self-collision via CollisionAvoidanceLimit.
# ---------------------------------------------------------------------------

_MIN_DIST = 0.04
_DETECT_DIST = 0.08
_COLLISION_GAIN = 0.45


def _subtree_geom_ids(model: mujoco.MjModel, root_body_name: str) -> list[int]:
    root_id = model.body(root_body_name).id
    ids: list[int] = []
    for gid in range(model.ngeom):
        if not (model.geom_contype[gid] and model.geom_conaffinity[gid]):
            continue
        body_id = model.geom_bodyid[gid]
        while True:
            if body_id == root_id:
                ids.append(gid)
                break
            if body_id <= 0:
                break
            body_id = model.body_parentid[body_id]
    return ids


def _dual_iiwa_model() -> mujoco.MjModel:
    iiwa_xml = _EXAMPLES / "kuka_iiwa_14" / "iiwa14.xml"
    root = mujoco.MjSpec()
    root.stat.meansize = 0.08
    root.stat.extent = 1.0
    root.stat.center[:] = (0, 0, 0.5)
    root.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE,
                            size=[1, 1, 0.01], contype=0, conaffinity=0)
    left_site = root.worldbody.add_site(name="l_attachment_site", pos=[0, 0.2, 0], group=5)
    right_site = root.worldbody.add_site(name="r_attachment_site", pos=[0, -0.2, 0], group=5)
    left = mujoco.MjSpec.from_file(iiwa_xml.as_posix())
    left.modelname = "l_iiwa"
    left.delete(left.key("home"))
    root.attach(left, site=left_site, prefix="l_iiwa/")
    right = mujoco.MjSpec.from_file(iiwa_xml.as_posix())
    right.modelname = "r_iiwa"
    right.delete(right.key("home"))
    root.attach(right, site=right_site, prefix="r_iiwa/")
    return root.compile()


def build_dual_iiwa(nworld: int, device: str | None, motion: str = "dense"):
    """``motion='dense'`` sweeps the arms through each other (most worlds have a
    nearby pair every step — prefilter can skip little); ``'sparse'`` keeps each
    arm on its own side (few worlds ever collide — prefilter skips most)."""
    model = _dual_iiwa_model()
    cfg = mw.Configuration(model, nworld=nworld, device=device)
    q_home = np.tile(np.concatenate([_ARM_HOME_Q, _ARM_HOME_Q]), (nworld, 1))
    cfg.update(q_home)

    left_ee = mw.FrameTask("l_iiwa/attachment_site", "site", position_cost=2.0,
                           orientation_cost=1.0, lm_damping=1e-2)
    right_ee = mw.FrameTask("r_iiwa/attachment_site", "site", position_cost=2.0,
                            orientation_cost=1.0, lm_damping=1e-2)
    posture = mw.PostureTask(model, cost=5e-3)
    posture.set_target_from_configuration(cfg)
    tasks = [left_ee, right_ee, posture]

    collision = mw.CollisionAvoidanceLimit(
        model,
        geom_pairs=[(_subtree_geom_ids(model, "l_iiwa/link5"),
                     _subtree_geom_ids(model, "r_iiwa/link5"))],
        gain=_COLLISION_GAIN,
        minimum_distance_from_collisions=_MIN_DIST,
        collision_detection_distance=_DETECT_DIST,
    )
    limits = [mw.ConfigurationLimit(model), collision]
    solver = mw.ConstrainedSolver(cfg, limits=limits, admm_iters=40, damping=1e-2)

    base_l = cfg.get_transform_frame_to_world("l_iiwa/attachment_site", "site").numpy().copy()
    base_r = cfg.get_transform_frame_to_world("r_iiwa/attachment_site", "site").numpy().copy()
    left_ee.set_target(base_l, configuration=cfg)
    right_ee.set_target(base_r, configuration=cfg)
    # dense: left arm reaches to +y, right to -y -> they cross in the middle.
    # sparse: each arm stays on its own side, so pairs are rarely near.
    if motion == "sparse":
        pos_a = np.array([0.392, -0.55, 0.6])   # left stays left
        pos_b = np.array([0.392, 0.55, 0.6])    # right stays right
    else:
        pos_a = np.array([0.392, -0.392, 0.6])
        pos_b = np.array([0.392, 0.392, 0.6])
    phase = np.arange(nworld) * (2.0 * math.pi / max(nworld, 1))
    lt = base_l.copy()
    rt = base_r.copy()

    def update(t: float) -> None:
        mu = 0.5 * (1.0 + np.cos(t + phase))
        bump = 0.2 * np.sin(mu * math.pi) ** 2
        lt[:] = base_l
        rt[:] = base_r
        if motion == "sparse":
            # Small local bob on each side; the arms never approach each other.
            lt[:, 6] = pos_a[2] + 0.15 * mu
            rt[:, 6] = pos_b[2] + 0.15 * (1.0 - mu)
            lt[:, 5] = pos_a[1]
            rt[:, 5] = pos_b[1]
        else:
            lt[:, 4] = pos_a[0] + (pos_b[0] - pos_a[0]) * mu
            lt[:, 5] = pos_a[1] + (pos_b[1] - pos_a[1]) * mu
            lt[:, 6] = pos_a[2] + (pos_b[2] - pos_a[2] + bump) * mu
            rt[:, 4] = pos_b[0] + (pos_a[0] - pos_b[0]) * mu
            rt[:, 5] = pos_b[1] + (pos_a[1] - pos_b[1]) * mu
            rt[:, 6] = pos_b[2] + (pos_a[2] - pos_b[2] - bump) * mu
        left_ee.set_target(lt, configuration=cfg)
        right_ee.set_target(rt, configuration=cfg)

    return dict(cfg=cfg, model=model, tasks=tasks, solver=solver, update=update,
                limits=limits, collision=collision, hot_limit=collision, damping=1e-2)


SCENES = {"cassie": build_cassie, "dual_iiwa": build_dual_iiwa}


# ---------------------------------------------------------------------------
# Throughput + component split.
# ---------------------------------------------------------------------------


def _time_hot(s, cfg, dt: float) -> float:
    """Time the host-heavy assembly component in isolation (us)."""
    import warp as wp

    if "hot_task" in s:  # equality: one _eval fills both error + jacobian
        task = s["hot_task"]
        t0 = perf_counter_ns()
        task.compute_jacobian(cfg)
        return (perf_counter_ns() - t0) * 1e-3
    # collision: time scatter into fresh G/h buffers
    limit = s["hot_limit"]
    m = limit.n_inequalities
    with wp.ScopedDevice(cfg.device):
        G = wp.zeros((cfg.nworld, m, cfg.nv), dtype=float)
        h = wp.zeros((cfg.nworld, m), dtype=float)
    t0 = perf_counter_ns()
    limit.scatter_inequalities(cfg, dt, 0, G, h)
    return (perf_counter_ns() - t0) * 1e-3


def run(scene_key, *, nworld, steps, warmup, device, iterations, profile, motion="dense"):
    builder = SCENES[scene_key]
    s = builder(nworld, device, motion) if scene_key == "dual_iiwa" else builder(nworld, device)
    cfg, tasks, solver, update = s["cfg"], s["tasks"], s["solver"], s["update"]
    collision = s.get("collision")
    if collision is not None and not hasattr(collision, "_prefilter_worlds"):
        collision = None  # baseline limit has no device prefilter to report

    step_us, hot_us, skip = [], [], []
    for i in range(warmup + steps):
        update(i * DT)
        t0 = perf_counter_ns()
        solver.solve_and_integrate(tasks, DT, iterations=iterations, use_graph=False)
        common.sync(device)
        dt_us = (perf_counter_ns() - t0) * 1e-3
        if profile:
            common.sync(device)
            hot_us.append(_time_hot(s, cfg, DT))
            if collision is not None and i >= warmup:
                surv = collision._prefilter_worlds(cfg).size
                skip.append(1.0 - surv / nworld)
        if i >= warmup:
            step_us.append(dt_us)

    st = common.summarize(step_us)
    out = dict(scene=scene_key, nworld=nworld, mean_us=st["mean"],
               solves_per_s=common.throughput(st["mean"] * 1e-6, nworld),
               us_per_solve=st["mean"] / nworld)
    if profile:
        hot = common.summarize(hot_us[warmup:] if len(hot_us) > warmup else hot_us)
        out["hot_us"] = hot["mean"]
        out["hot_frac"] = hot["mean"] / st["mean"] if st["mean"] > 0 else 0.0
        out["skip_pct"] = 100.0 * (sum(skip) / len(skip)) if skip else float("nan")
    return out


# ---------------------------------------------------------------------------
# Accuracy parity vs mink (world 0).
# ---------------------------------------------------------------------------


def check_cassie(*, steps, device):
    import mink

    model = _cassie_model()
    s = build_cassie(1, device)
    cfg = s["cfg"]
    task_mw = s["hot_task"]

    cfg_mk = mink.Configuration(model)
    cfg_mk.update_from_keyframe("home")
    task_mk = mink.EqualityConstraintTask(model=model, cost=500.0, gain=0.5,
                                          lm_damping=1e-3)

    e_errs, j_errs = [], []
    rng = np.random.default_rng(0)
    q0 = cfg.q.numpy()[0].copy()
    for _ in range(steps):
        q = q0 + rng.uniform(-0.1, 0.1, size=q0.shape).astype(np.float32)
        cfg.update(q=q)
        cfg_mk.update(q.astype(np.float64))
        e_mw = task_mw.compute_error(cfg).numpy()[0]
        j_mw = task_mw.compute_jacobian(cfg).numpy()[0]
        e_mk = task_mk.compute_error(cfg_mk)
        j_mk = task_mk.compute_jacobian(cfg_mk)
        e_errs.append(float(np.max(np.abs(e_mw - e_mk))))
        j_errs.append(float(np.max(np.abs(j_mw - j_mk))))
    print(f"# cassie equality parity vs mink over {steps} random configs")
    print(f"  max |e_mw - e_mk| : {max(e_errs):.3e}")
    print(f"  max |J_mw - J_mk| : {max(j_errs):.3e}")


def check_dual_iiwa(*, steps, device):
    import mink
    import warp as wp

    model = _dual_iiwa_model()
    s = build_dual_iiwa(1, device)
    cfg = s["cfg"]
    limit_mw = s["collision"]
    geom_pairs = [(_subtree_geom_ids(model, "l_iiwa/link5"),
                   _subtree_geom_ids(model, "r_iiwa/link5"))]
    limit_mk = mink.CollisionAvoidanceLimit(
        model=model, geom_pairs=geom_pairs, gain=_COLLISION_GAIN,
        minimum_distance_from_collisions=_MIN_DIST,
        collision_detection_distance=_DETECT_DIST)
    cfg_mk = mink.Configuration(model)

    m = limit_mw.n_inequalities
    g_errs, h_errs, nactive = [], [], []
    rng = np.random.default_rng(0)
    q0 = np.concatenate([_ARM_HOME_Q, _ARM_HOME_Q])
    for _ in range(steps):
        q = q0 + rng.uniform(-0.3, 0.3, size=q0.shape)
        cfg.update(q=q.astype(np.float32))
        cfg_mk.update(q)
        cn = limit_mk.compute_qp_inequalities(cfg_mk, DT)
        with wp.ScopedDevice(cfg.device):
            G = wp.zeros((1, m, model.nv), dtype=float)
            h = wp.zeros((1, m), dtype=float)
        limit_mw.scatter_inequalities(cfg, DT, 0, G, h)
        g_mw, h_mw = G.numpy()[0], h.numpy()[0]
        active = np.isfinite(cn.h) & (np.abs(cn.h) < 1e20)
        nactive.append(int(active.sum()))
        for i in np.where(active)[0]:
            g_errs.append(float(np.max(np.abs(g_mw[i] - cn.G[i]))))
            h_errs.append(float(np.abs(h_mw[i] - cn.h[i])))
    print(f"# dual_iiwa collision parity vs mink over {steps} random configs")
    print(f"  active rows/step  : {np.mean(nactive):.1f} (of {m})")
    print(f"  max |G_mw - G_mk| : {max(g_errs) if g_errs else 0.0:.3e}")
    print(f"  max |h_mw - h_mk| : {max(h_errs) if h_errs else 0.0:.3e}")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("scene", nargs="?", default="both",
                    choices=["both", *SCENES])
    ap.add_argument("--nworld", nargs="+", type=int, default=[1, 16, 64, 256])
    ap.add_argument("--steps", type=int, default=60)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iterations", type=int, default=1)
    ap.add_argument("--device", default=None)
    ap.add_argument("--profile", action="store_true",
                    help="also report host-assembly component time + fraction")
    ap.add_argument("--collision-motion", choices=["dense", "sparse"], default="dense",
                    help="dual_iiwa trajectory: dense crosses the arms (prefilter "
                         "skips little), sparse keeps them apart (prefilter skips most)")
    ap.add_argument("--check", action="store_true",
                    help="accuracy parity vs mink (world 0), no timing")
    args = ap.parse_args()

    scenes = list(SCENES) if args.scene == "both" else [args.scene]

    if args.check:
        for sk in scenes:
            (check_cassie if sk == "cassie" else check_dual_iiwa)(
                steps=args.steps, device=args.device)
        return

    hdr = f"{'scene':>10} {'nworld':>7} {'solves/s':>12} {'us/solve':>10} {'step us':>10}"
    if args.profile:
        hdr += f" {'hot us':>10} {'hot %':>7} {'bp-skip':>8}"
    print(f"# steps={args.steps} warmup={args.warmup} iters={args.iterations} "
          f"device={args.device} collision-motion={args.collision_motion}")
    print(hdr)
    for sk in scenes:
        for nw in args.nworld:
            r = run(sk, nworld=nw, steps=args.steps, warmup=args.warmup,
                    device=args.device, iterations=args.iterations,
                    profile=args.profile, motion=args.collision_motion)
            line = (f"{r['scene']:>10} {r['nworld']:>7} {r['solves_per_s']:>12.0f} "
                    f"{r['us_per_solve']:>10.2f} {r['mean_us']:>10.1f}")
            if args.profile:
                skip = r.get("skip_pct", float("nan"))
                skip_s = "     -  " if skip != skip else f"{skip:>6.1f}% "
                line += f" {r['hot_us']:>10.1f} {100 * r['hot_frac']:>6.1f}% {skip_s:>8}"
            print(line)


if __name__ == "__main__":
    main()
