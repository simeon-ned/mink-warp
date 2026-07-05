"""Batched dual Panda with inter-arm self-collision avoidance.

Arms sweep through each other's workspace (Mink ``arm_dual_iiwa.py`` pattern).
Collision pairs are the link5-and-below subtrees on left vs right.

Run:
  uv sync --extra dev --extra examples
  uv run examples/self_collision_dual_panda.py
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

import mink_warp as mw

NUM_WORLDS = 64
DT = 0.01
FREQUENCY = 50.0
ENV_SPACING = 1.8
SWAP_FREQ = 0.35
Z_BUMP = 0.12


def _grid_origins(n: int, spacing: float) -> np.ndarray:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    origins = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, cols)
        origins[i, 0] = (c - 0.5 * (cols - 1)) * spacing
        origins[i, 1] = (r - 0.5 * (rows - 1)) * spacing
    return origins


def _child_body_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    return [
        i
        for i in range(model.nbody)
        if model.body_parentid[i] == body_id and i != body_id
    ]


def _subtree_body_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    stack = [body_id]
    out: list[int] = []
    while stack:
        bid = stack.pop()
        out.append(bid)
        stack.extend(_child_body_ids(model, bid))
    return out


def _body_geom_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    start = model.body_geomadr[body_id]
    return list(range(start, start + model.body_geomnum[body_id]))


def _subtree_geom_ids(model: mujoco.MjModel, body_name: str) -> list[int]:
    body_id = model.body(body_name).id
    geoms: list[int] = []
    for bid in _subtree_body_ids(model, body_id):
        geoms.extend(_body_geom_ids(model, bid))
    return geoms


def main() -> None:
    xml = Path(__file__).resolve().parent / "franka_emika_panda" / "dual_panda_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    cfg = mw.Configuration(model, nworld=NUM_WORLDS)
    cfg.update_from_keyframe("home1")

    left_ee = mw.FrameTask(
        "attachment_site_left", "site",
        position_cost=2.0, orientation_cost=1.0, lm_damping=1e-3,
    )
    right_ee = mw.FrameTask(
        "attachment_site_right", "site",
        position_cost=2.0, orientation_cost=1.0, lm_damping=1e-3,
    )
    posture = mw.PostureTask(model, cost=1e-2)
    posture.set_target_from_configuration(cfg)

    l_geoms = _subtree_geom_ids(model, "link5_l")
    r_geoms = _subtree_geom_ids(model, "link5_r")
    limits = [
        mw.ConfigurationLimit(model),
        mw.CollisionAvoidanceLimit(
            model,
            geom_pairs=[(l_geoms, r_geoms)],
            minimum_distance_from_collisions=0.08,
            collision_detection_distance=0.15,
        ),
    ]
    tasks = [left_ee, right_ee, posture]
    solver = mw.ConstrainedSolver(cfg, limits=limits, admm_iters=40, damping=1e-2)

    base_l = cfg.get_transform_frame_to_world("attachment_site_left", "site").numpy().copy()
    base_r = cfg.get_transform_frame_to_world("attachment_site_right", "site").numpy().copy()
    left_ee.set_target(base_l, configuration=cfg)
    right_ee.set_target(base_r, configuration=cfg)

    pos_l = base_l[:, 4:7].copy()
    pos_r = base_r[:, 4:7].copy()

    origins = _grid_origins(NUM_WORLDS, ENV_SPACING)
    server = viser.ViserServer(label="mink-warp dual Panda self-collision")
    scene = ViserMujocoScene(server, model, num_envs=NUM_WORLDS)
    extent = float(np.max(np.linalg.norm(origins[:, :2], axis=1)) + ENV_SPACING)
    if hasattr(scene, "create_scene_gui"):
        scene.create_scene_gui(
            camera_distance=max(2.8, 1.2 * extent),
            camera_azimuth=120.0,
            camera_elevation=-20.0,
        )

    print(f"Dual Panda XML: {xml}")
    print(
        f"Open the viser URL. {NUM_WORLDS} dual-arm cells, "
        f"{len(l_geoms)}×{len(r_geoms)} geom pairs @ {FREQUENCY} Hz."
    )
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
                mu = 0.5 * (1.0 + math.cos(2.0 * math.pi * SWAP_FREQ * t + phase))
                bump = np.array([0.0, 0.0, Z_BUMP * math.sin(mu * math.pi) ** 2])
                l_xyz = pos_l[i] + (pos_r[i] - pos_l[i] + bump) * mu
                r_xyz = pos_r[i] + (pos_l[i] - pos_r[i] - bump) * mu
                left_targets[i] = base_l[i]
                right_targets[i] = base_r[i]
                left_targets[i, 4:7] = l_xyz
                right_targets[i, 4:7] = r_xyz

            left_ee.set_target(left_targets, configuration=cfg)
            right_ee.set_target(right_targets, configuration=cfg)
            solver.solve_and_integrate(tasks, DT, iterations=1, use_graph=False)

            xpos = cfg.wp_data.xpos.numpy().copy()
            xpos += origins[:, None, :]
            scene.update_from_arrays(xpos, cfg.wp_data.xmat.numpy(), qpos=cfg.q.numpy())

            dt_loop = time.time() - t0
            if dt_loop < 1.0 / FREQUENCY:
                time.sleep(1.0 / FREQUENCY - dt_loop)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
