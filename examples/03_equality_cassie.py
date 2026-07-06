"""Cassie IK with equality constraint task (closed-chain four-bars).

Each biped gets a phase-offset COM bob while feet stay pinned; mjviser grid view.

Run:
  uv sync --extra dev --extra examples
  uv run examples/03_equality_cassie.py
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

from _viser_utils import grid_origins

NUM_WORLDS = 1024
DT = 0.01
FREQUENCY = 50.0
ENV_SPACING = 2.5
AMP_COM_Z = 0.15
FREQ_COM = 0.35


def _grid_origins(n: int, spacing: float) -> np.ndarray:
    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    origins = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, cols)
        origins[i, 0] = (c - 0.5 * (cols - 1)) * spacing
        origins[i, 1] = (r - 0.5 * (rows - 1)) * spacing
    return origins


def main() -> None:
    xml = Path(__file__).resolve().parent / "agility_cassie" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    cfg = mw.Configuration(model, nworld=NUM_WORLDS)
    cfg.update_from_keyframe("home")

    pelvis = mw.FrameTask(
        "cassie-pelvis", "body",
        position_cost=0.0, orientation_cost=10.0,
    )
    posture = mw.PostureTask(model, cost=1.0)
    com = mw.ComTask(cost=200.0)
    equality = mw.EqualityConstraintTask(
        model=model,
        cost=500.0,
        gain=0.5,
        lm_damping=1e-3,
    )

    feet = ["left-foot", "right-foot"]
    feet_tasks = [
        mw.FrameTask(
            foot, "body",
            position_cost=200.0, orientation_cost=10.0, lm_damping=1.0,
        )
        for foot in feet
    ]
    tasks = [pelvis, posture, com, equality, *feet_tasks]
    solver = mw.IKSolver(cfg)

    posture.set_target_from_configuration(cfg)
    pelvis.set_target_from_configuration(cfg)
    for ft in feet_tasks:
        ft.set_target_from_configuration(cfg)

    base_com = cfg.wp_data.subtree_com.numpy()[:, 1, :].copy()
    com.set_target(base_com, configuration=cfg)

    origins = _grid_origins(NUM_WORLDS, ENV_SPACING)
    server = viser.ViserServer(label="mink-warp Cassie (equality)")
    scene = ViserMujocoScene(server, model, num_envs=NUM_WORLDS)
    extent = float(np.max(np.linalg.norm(origins[:, :2], axis=1)) + ENV_SPACING)
    if hasattr(scene, "create_scene_gui"):
        scene.create_scene_gui(
            camera_distance=max(3.5, 1.3 * extent),
            camera_azimuth=150.0,
            camera_elevation=-20.0,
        )

    print(f"Cassie XML: {xml}")
    print(f"Open the viser URL. {NUM_WORLDS} Cassies with equality task @ {FREQUENCY} Hz.")
    print("Ctrl+C to stop.")

    t_start = time.time()
    try:
        while True:
            t0 = time.time()
            t = t0 - t_start
            com_targets = base_com.copy()
            for i in range(NUM_WORLDS):
                phase = i * (2.0 * math.pi / NUM_WORLDS)
                com_targets[i, 2] += AMP_COM_Z * math.sin(
                    2.0 * math.pi * FREQ_COM * t + phase
                )
            com.set_target(com_targets, configuration=cfg)

            solver.solve_and_integrate(
                tasks, DT, damping=1e-1, iterations=1, use_graph=False
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
