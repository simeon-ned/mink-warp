"""Dual Kuka iiwa14 with inter-arm self-collision avoidance.

Port of Mink ``arm_dual_iiwa.py``: two iiwa arms sweep through each other's
workspace while ``CollisionAvoidanceLimit`` keeps link5-and-below subtrees apart.

Run:
  uv sync --extra dev --extra examples
  uv run examples/04_self_collision_dual_iiwa.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import mink_warp as mw

from _viser_utils import (
    grid_origins,
    subtree_collision_geom_ids,
    sync_scene,
    warmup_solver,
)

_IIWA_XML = _EXAMPLES / "kuka_iiwa_14" / "iiwa14.xml"
_ARM_HOME_Q = np.array([0.0, 0.0, 0.0, -1.5708, 0.0, 1.5708, 0.0], dtype=np.float64)

NUM_WORLDS = 128
DT = 1.0 / 60.0
FREQUENCY = 60.0
ENV_SPACING = 2.0
Z_BUMP = 0.2

# Collision: keep detection_dist ~2× min_dist so constraints ramp in smoothly.
MIN_COLLISION_DIST = 0.04
COLLISION_DETECT_DIST = 0.08
COLLISION_GAIN = 0.45
IK_ITERS = 2


def construct_model() -> mujoco.MjModel:
    """Dual iiwa scene (same layout as Mink ``arm_dual_iiwa.py``)."""
    root = mujoco.MjSpec()
    root.stat.meansize = 0.08
    root.stat.extent = 1.0
    root.stat.center[:] = (0, 0, 0.5)

    root.worldbody.add_light(pos=(0, 0, 1.5), type=mujoco.mjtLightType.mjLIGHT_SPOT)
    root.worldbody.add_geom(
        name="floor",
        type=mujoco.mjtGeom.mjGEOM_PLANE,
        size=[1, 1, 0.01],
        contype=0,
        conaffinity=0,
    )

    left_site = root.worldbody.add_site(name="l_attachment_site", pos=[0, 0.2, 0], group=5)
    right_site = root.worldbody.add_site(name="r_attachment_site", pos=[0, -0.2, 0], group=5)

    left_iiwa = mujoco.MjSpec.from_file(_IIWA_XML.as_posix())
    left_iiwa.modelname = "l_iiwa"
    left_iiwa.delete(left_iiwa.key("home"))
    root.attach(left_iiwa, site=left_site, prefix="l_iiwa/")

    right_iiwa = mujoco.MjSpec.from_file(_IIWA_XML.as_posix())
    right_iiwa.modelname = "r_iiwa"
    right_iiwa.delete(right_iiwa.key("home"))
    root.attach(right_iiwa, site=right_site, prefix="r_iiwa/")

    return root.compile()


def main() -> None:
    model = construct_model()
    cfg = mw.Configuration(model, nworld=NUM_WORLDS)
    q_home = np.tile(np.concatenate([_ARM_HOME_Q, _ARM_HOME_Q]), (NUM_WORLDS, 1))
    cfg.update(q_home)

    left_ee = mw.FrameTask(
        "l_iiwa/attachment_site", "site",
        position_cost=2.0, orientation_cost=1.0, lm_damping=1e-2,
    )
    right_ee = mw.FrameTask(
        "r_iiwa/attachment_site", "site",
        position_cost=2.0, orientation_cost=1.0, lm_damping=1e-2,
    )
    posture = mw.PostureTask(model, cost=5e-3)
    posture.set_target_from_configuration(cfg)

    limits = [
        mw.ConfigurationLimit(model),
        mw.CollisionAvoidanceLimit(
            model,
            geom_pairs=[
                (
                    subtree_collision_geom_ids(model, "l_iiwa/link5"),
                    subtree_collision_geom_ids(model, "r_iiwa/link5"),
                )
            ],
            gain=COLLISION_GAIN,
            minimum_distance_from_collisions=MIN_COLLISION_DIST,
            collision_detection_distance=COLLISION_DETECT_DIST,
        ),
    ]
    tasks = [left_ee, right_ee, posture]
    solver = mw.ConstrainedSolver(cfg, limits=limits, admm_iters=40, damping=1e-2)

    base_l = cfg.get_transform_frame_to_world("l_iiwa/attachment_site", "site").numpy().copy()
    base_r = cfg.get_transform_frame_to_world("r_iiwa/attachment_site", "site").numpy().copy()
    left_ee.set_target(base_l, configuration=cfg)
    right_ee.set_target(base_r, configuration=cfg)

    # Mink arm_dual_iiwa.py rest positions for the swap trajectory.
    pos_a = np.array([0.392, -0.392, 0.6], dtype=np.float64)
    pos_b = np.array([0.392, 0.392, 0.6], dtype=np.float64)

    origins = grid_origins(NUM_WORLDS, ENV_SPACING)
    warmup_solver(solver, tasks, DT, label="ConstrainedSolver")

    server = viser.ViserServer(label="mink-warp dual Kuka self-collision")
    scene = ViserMujocoScene(server, model, num_envs=NUM_WORLDS)
    sync_scene(scene, cfg, origins)
    extent = float(np.max(np.linalg.norm(origins[:, :2], axis=1)) + ENV_SPACING)
    if hasattr(scene, "create_scene_gui"):
        scene.create_scene_gui(
            camera_distance=max(2.8, 1.2 * extent),
            camera_azimuth=-180.0,
            camera_elevation=-20.0,
        )

    print(f"Dual iiwa model built from {_IIWA_XML}")
    print(f"Open the viser URL. {NUM_WORLDS} dual-arm cells @ {FREQUENCY} Hz.")
    print("Ctrl+C to stop.")

    left_targets = np.empty_like(base_l)
    right_targets = np.empty_like(base_r)
    t_start = time.time()
    try:
        while True:
            t0 = time.time()
            t = t0 - t_start
            for i in range(NUM_WORLDS):
                phase = i * (2.0 * math.pi / NUM_WORLDS)
                mu = 0.5 * (1.0 + math.cos(t + phase))
                bump = np.array([0.0, 0.0, Z_BUMP * math.sin(mu * math.pi) ** 2])
                l_xyz = pos_a + (pos_b - pos_a + bump) * mu
                r_xyz = pos_b + (pos_a - pos_b - bump) * mu
                left_targets[i] = base_l[i]
                right_targets[i] = base_r[i]
                left_targets[i, 4:7] = l_xyz
                right_targets[i, 4:7] = r_xyz

            left_ee.set_target(left_targets, configuration=cfg)
            right_ee.set_target(right_targets, configuration=cfg)
            solver.solve_and_integrate(tasks, DT, iterations=IK_ITERS, use_graph=False)

            sync_scene(scene, cfg, origins)

            dt_loop = time.time() - t0
            if dt_loop < 1.0 / FREQUENCY:
                time.sleep(1.0 / FREQUENCY - dt_loop)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
