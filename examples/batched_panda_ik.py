"""Batched differential IK on Franka Emika Panda, visualized with mjviser.

Phase-offset circular EE trajectories with a small z bob.

Run:
  uv sync --extra dev --extra examples
  uv run examples/batched_panda_ik.py
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

NUM_WORLDS = 512
DT = 0.01
FREQUENCY = 50.0
ENV_SPACING = 1.2
AMP_XY = 0.10
AMP_Z = 0.03
FREQ_XY = 0.2
FREQ_Z = 0.35


def main() -> None:
    xml = Path(__file__).resolve().parent / "franka_emika_panda" / "mjx_scene.xml"
    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    cfg = mw.Configuration(model, nworld=NUM_WORLDS)
    cfg.update_from_keyframe("home")

    q0 = cfg.q.numpy().copy()
    for i in range(NUM_WORLDS):
        q0[i, 0] += 0.03 * math.sin(i * (2.0 * math.pi / NUM_WORLDS))
    cfg.update(q=q0)

    frame = mw.FrameTask(
        "attachment_site", "site",
        position_cost=1.0, orientation_cost=1.0, gain=0.8, lm_damping=1.0,
    )
    posture = mw.PostureTask(model, cost=1e-2)
    limits = mw.JointLimitTask(model, cost=1.0)
    frame.set_target_from_configuration(cfg)
    posture.set_target_from_configuration(cfg)
    base = cfg.get_transform_frame_to_world("attachment_site", "site").numpy().copy()

    tasks = [frame, posture, limits]
    solver = mw.IKSolver(cfg)

    cols = math.ceil(math.sqrt(NUM_WORLDS))
    rows = math.ceil(NUM_WORLDS / cols)
    origins = np.zeros((NUM_WORLDS, 3), dtype=np.float32)
    for i in range(NUM_WORLDS):
        r, c = divmod(i, cols)
        origins[i, 0] = (c - 0.5 * (cols - 1)) * ENV_SPACING
        origins[i, 1] = (r - 0.5 * (rows - 1)) * ENV_SPACING

    server = viser.ViserServer(label="mink-warp batched Panda IK")
    scene = ViserMujocoScene(server, model, num_envs=NUM_WORLDS)
    extent = float(np.max(np.linalg.norm(origins[:, :2], axis=1)) + ENV_SPACING)
    if hasattr(scene, "create_scene_gui"):
        scene.create_scene_gui(
            camera_distance=max(2.5, 1.2 * extent),
            camera_azimuth=120.0,
            camera_elevation=-20.0,
        )

    print(f"Panda XML: {xml}")
    print(f"Open the viser URL. Tracking {NUM_WORLDS} Pandas @ {FREQUENCY} Hz.")
    print("Ctrl+C to stop.")

    t_start = time.time()
    try:
        while True:
            t0 = time.time()
            t = t0 - t_start
            targets = base.copy()
            for i in range(NUM_WORLDS):
                phase = i * (2.0 * math.pi / NUM_WORLDS)
                targets[i, 4] += AMP_XY * math.cos(2.0 * math.pi * FREQ_XY * t + phase)
                targets[i, 5] += AMP_XY * math.sin(2.0 * math.pi * FREQ_XY * t + phase)
                targets[i, 6] += AMP_Z * math.sin(2.0 * math.pi * FREQ_Z * t + 1.5 * phase)
            frame.set_target(targets, configuration=cfg)

            solver.solve_and_integrate(
                tasks, DT, damping=1e-3, iterations=1, use_graph=True
            )

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
