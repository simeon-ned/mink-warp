"""G1: RelativeFrameTask + collision limits, mjviser grid.

Right palm traces a world-space circle; left palm tracks a torso-relative wave.
Per-env phase offsets drive squat and hand motion.

Run:
  uv sync --extra dev --extra examples
  uv run examples/05_relative_frame_g1.py
"""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

_EXAMPLES = Path(__file__).resolve().parent
if str(_EXAMPLES) not in sys.path:
    sys.path.insert(0, str(_EXAMPLES))

import mujoco
import numpy as np
import viser
from mjviser import ViserMujocoScene

import mink_warp as mw
from mink_warp.lie import SE3

from _viser_utils import grid_origins, sync_scene, warmup_solver

NUM_WORLDS = 512
DT = 0.01
FREQUENCY = 50.0
ENV_SPACING = 2.0

SQUAT_DEPTH = 0.20
SQUAT_FREQ = 0.5
CIRCLE_RADIUS = 0.04
CIRCLE_FREQ = 0.8
WAVE_RADIUS = 0.04
WAVE_FREQ = 0.8
WAVE_Y_OFFSET = 0.05


def main() -> None:
    xml = Path(__file__).resolve().parent / "unitree_g1" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    cfg = mw.Configuration(model, nworld=NUM_WORLDS)
    cfg.update_from_keyframe("stand")

    feet = ["right_foot", "left_foot"]

    def orient(name: str, cost: float) -> mw.FrameTask:
        return mw.FrameTask(
            name, "body", position_cost=0.0, orientation_cost=cost, lm_damping=1e-3,
        )

    tasks = [
        orient("pelvis", 10.0),
        orient("torso_link", 5.0),
        posture := mw.PostureTask(model, cost=1e-1),
        com := mw.ComTask(cost=50.0),
    ]
    feet_tasks = [
        mw.FrameTask(
            foot, "site",
            position_cost=100.0, orientation_cost=10.0, lm_damping=0.0,
        )
        for foot in feet
    ]
    r_hand = mw.FrameTask(
        "right_palm", "site",
        position_cost=5.0, orientation_cost=0.5, gain=0.5, lm_damping=1e-3,
    )
    relative = mw.RelativeFrameTask(
        "left_palm", "site",
        "torso_link", "body",
        position_cost=5.0, orientation_cost=0.5, gain=0.5, lm_damping=1e-3,
    )
    tasks.extend([*feet_tasks, r_hand, relative])

    limits = [
        mw.ConfigurationLimit(model),
        mw.CollisionAvoidanceLimit(
            model,
            geom_pairs=[
                (["left_hand_collision"], ["left_thigh_collision"]),
                (["right_hand_collision"], ["right_thigh_collision"]),
            ],
            minimum_distance_from_collisions=0.005,
            collision_detection_distance=0.15,
        ),
    ]
    solver = mw.ConstrainedSolver(cfg, limits=limits, admm_iters=40, damping=1e-2)

    for t in tasks:
        if hasattr(t, "set_target_from_configuration"):
            t.set_target_from_configuration(cfg)

    base_com = cfg.wp_data.subtree_com.numpy()[:, 1, :].copy()
    base_r_hand = cfg.get_transform_frame_to_world("right_palm", "site").numpy().copy()
    base_relative = cfg.get_transform(
        "left_palm", "site", "torso_link", "body",
    ).numpy().copy()

    origins = grid_origins(NUM_WORLDS, ENV_SPACING)
    warmup_solver(solver, tasks, DT, label="ConstrainedSolver")

    server = viser.ViserServer(label="mink-warp G1 (RelativeFrameTask)")
    scene = ViserMujocoScene(server, model, num_envs=NUM_WORLDS)
    sync_scene(scene, cfg, origins)
    extent = float(np.max(np.linalg.norm(origins[:, :2], axis=1)) + ENV_SPACING)
    if hasattr(scene, "create_scene_gui"):
        scene.create_scene_gui(
            camera_distance=max(3.0, 1.3 * extent),
            camera_azimuth=180.0,
            camera_elevation=-20.0,
        )

    print(f"G1 XML: {xml}")
    print(f"Open the viser URL. {NUM_WORLDS} G1s, relative hand + collision @ {FREQUENCY} Hz.")
    print("Ctrl+C to stop.")

    rel_targets = np.empty((NUM_WORLDS, 7), dtype=np.float32)
    t_start = time.time()
    try:
        while True:
            t0 = time.time()
            t = t0 - t_start

            com_targets = base_com.copy()
            r_hand_targets = base_r_hand.copy()
            for i in range(NUM_WORLDS):
                phase = i * (2.0 * math.pi / NUM_WORLDS)

                squat = 0.5 * (1.0 - math.cos(2.0 * math.pi * SQUAT_FREQ * t + phase))
                com_targets[i, 2] -= SQUAT_DEPTH * squat

                angle = 2.0 * math.pi * CIRCLE_FREQ * t + phase
                r_hand_targets[i, 4] += CIRCLE_RADIUS * math.cos(angle)
                r_hand_targets[i, 6] += CIRCLE_RADIUS * math.sin(angle)

                wave_t = 2.0 * math.pi * WAVE_FREQ * t + phase
                dy = WAVE_Y_OFFSET + WAVE_RADIUS * math.sin(wave_t)
                dz = WAVE_RADIUS * math.cos(wave_t)
                T = SE3(wxyz_xyz=base_relative[i].astype(np.float64))
                rel_targets[i] = (
                    T @ SE3.from_translation(np.array([0.0, dy, dz]))
                ).wxyz_xyz.astype(np.float32)

            com.set_target(com_targets, configuration=cfg)
            r_hand.set_target(r_hand_targets, configuration=cfg)
            relative.set_target(rel_targets, configuration=cfg)

            solver.solve_and_integrate(tasks, DT, iterations=1, use_graph=False)

            sync_scene(scene, cfg, origins)

            dt_loop = time.time() - t0
            if dt_loop < 1.0 / FREQUENCY:
                time.sleep(1.0 / FREQUENCY - dt_loop)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
