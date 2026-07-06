"""UR5e IK with hard limits: config, collision avoidance, velocity cap.

Phase-offset EE circles; each arm is spaced on a grid and visualized with mjviser.

Run:
  uv sync --extra dev --extra examples
  uv run examples/02_constrained_ur5e.py
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

from _viser_utils import grid_origins, sync_scene, warmup_solver

NUM_WORLDS = 256
DT = 0.01
FREQUENCY = 50.0
ENV_SPACING = 1.4
AMP_XY = 0.08
AMP_Z = 0.04
FREQ_XY = 0.25
FREQ_Z = 0.4


def main() -> None:
    xml = Path(__file__).resolve().parent / "universal_robots_ur5e" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    cfg = mw.Configuration(model, nworld=NUM_WORLDS)
    cfg.update_from_keyframe("home")

    q0 = cfg.q.numpy().copy()
    for i in range(NUM_WORLDS):
        q0[i, 0] += 0.04 * math.sin(i * (2.0 * math.pi / NUM_WORLDS))
    cfg.update(q=q0)

    ee = mw.FrameTask(
        "attachment_site", "site",
        position_cost=1.0, orientation_cost=1.0,
        gain=0.8, lm_damping=1.0,
    )
    posture = mw.PostureTask(model, cost=1e-2)
    posture.set_target_from_configuration(cfg)

    limits = [
        mw.ConfigurationLimit(model),
        mw.CollisionAvoidanceLimit(
            model,
            geom_pairs=[(["wrist_3_link"], ["floor", "wall"])],
        ),
        mw.VelocityLimit(
            model,
            {
                "shoulder_pan": np.pi,
                "shoulder_lift": np.pi,
                "elbow": np.pi,
                "wrist_1": np.pi,
                "wrist_2": np.pi,
                "wrist_3": np.pi,
            },
        ),
    ]
    tasks = [ee, posture]
    solver = mw.ConstrainedSolver(cfg, limits=limits, admm_iters=40)

    base_ee = cfg.get_transform_frame_to_world("attachment_site", "site").numpy().copy()
    ee.set_target(base_ee, configuration=cfg)

    origins = grid_origins(NUM_WORLDS, ENV_SPACING)
    warmup_solver(solver, tasks, DT, label="ConstrainedSolver")

    server = viser.ViserServer(label="mink-warp UR5e (constrained)")
    scene = ViserMujocoScene(server, model, num_envs=NUM_WORLDS)
    sync_scene(scene, cfg, origins)
    extent = float(np.max(np.linalg.norm(origins[:, :2], axis=1)) + ENV_SPACING)
    if hasattr(scene, "create_scene_gui"):
        scene.create_scene_gui(
            camera_distance=max(2.5, 1.2 * extent),
            camera_azimuth=120.0,
            camera_elevation=-20.0,
        )

    print(f"UR5e XML: {xml}")
    print(f"Open the viser URL. {NUM_WORLDS} arms with hard limits @ {FREQUENCY} Hz.")
    print("Ctrl+C to stop.")

    t_start = time.time()
    try:
        while True:
            t0 = time.time()
            t = t0 - t_start
            targets = base_ee.copy()
            for i in range(NUM_WORLDS):
                phase = i * (2.0 * math.pi / NUM_WORLDS)
                targets[i, 4] += AMP_XY * math.cos(2.0 * math.pi * FREQ_XY * t + phase)
                targets[i, 5] += AMP_XY * math.sin(2.0 * math.pi * FREQ_XY * t + phase)
                targets[i, 6] += AMP_Z * math.sin(2.0 * math.pi * FREQ_Z * t + 1.5 * phase)
            ee.set_target(targets, configuration=cfg)

            solver.solve_and_integrate(tasks, DT, iterations=1, use_graph=False)

            sync_scene(scene, cfg, origins)

            dt_loop = time.time() - t0
            if dt_loop < 1.0 / FREQUENCY:
                time.sleep(1.0 / FREQUENCY - dt_loop)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
