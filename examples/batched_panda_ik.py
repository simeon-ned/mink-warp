"""Batched differential IK on Franka Emika Panda, visualized with mjviser.

Uses the same Panda MJCF as Mink's ``examples/arm_panda.py``, with several
worlds tracking phase-offset circular end-effector trajectories in parallel.

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

import mink_warp as mw

NUM_WORLDS = 512
DT = 0.01
IK_ITERS = 10
FREQUENCY = 50.0
# Display layout (IK stays in a shared local frame; we offset xpos for viz only).
ENV_SPACING = 1.2
# Match Mink's arm_panda.py trajectory in xy; add a small z bob.
AMP_XY = 0.10
AMP_Z = 0.03
FREQ_XY = 0.2
FREQ_Z = 0.35


def _panda_xml() -> Path:
    """Local Franka Emika Panda scene (vendored from Mink examples)."""
    path = Path(__file__).resolve().parent / "franka_emika_panda" / "mjx_scene.xml"
    if not path.is_file():
        raise FileNotFoundError(f"Missing Panda assets at {path}")
    return path


def _env_origins(nworld: int, spacing: float = ENV_SPACING) -> np.ndarray:
    """Even grid of env origins in xy, centered on the origin."""
    cols = math.ceil(math.sqrt(nworld))
    rows = math.ceil(nworld / cols)
    origins = np.zeros((nworld, 3), dtype=np.float32)
    for i in range(nworld):
        row, col = divmod(i, cols)
        origins[i, 0] = (col - 0.5 * (cols - 1)) * spacing
        origins[i, 1] = (row - 0.5 * (rows - 1)) * spacing
    return origins


def _target_poses(
    t: float,
    nworld: int,
    base_poses: np.ndarray,
    amp_xy: float = AMP_XY,
    amp_z: float = AMP_Z,
    freq_xy: float = FREQ_XY,
    freq_z: float = FREQ_Z,
) -> np.ndarray:
    """Per-world EE targets: circular xy + slight z oscillation (local frame)."""
    poses = base_poses.copy()
    for i in range(nworld):
        phase_i = i * (2.0 * math.pi / nworld)
        phase_xy = 2.0 * math.pi * freq_xy * t + phase_i
        phase_z = 2.0 * math.pi * freq_z * t + 1.5 * phase_i
        poses[i, 4] += amp_xy * math.cos(phase_xy)
        poses[i, 5] += amp_xy * math.sin(phase_xy)
        poses[i, 6] += amp_z * math.sin(phase_z)
    return poses


def main() -> None:
    import viser
    from mjviser import ViserMujocoScene

    xml_path = _panda_xml()
    model = mujoco.MjModel.from_xml_path(xml_path.as_posix())

    cfg = mw.Configuration(model, nworld=NUM_WORLDS)
    cfg.update_from_keyframe("home")

    # Slight posture diversity so worlds are not identical at t=0.
    q0 = cfg.q.numpy().copy()
    for i in range(NUM_WORLDS):
        q0[i, 0] += 0.03 * math.sin(i * (2.0 * math.pi / NUM_WORLDS))
    cfg.update(q=q0)

    frame = mw.FrameTask(
        frame_name="attachment_site",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=1.0,
        gain=0.8,
        lm_damping=1.0,
    )
    posture = mw.PostureTask(model, cost=1e-2)
    posture.set_target_from_configuration(cfg)

    # Home EE poses (orientation held; position orbits like Mink's mocap target).
    base_poses = cfg.get_transform_frame_to_world(
        "attachment_site", "site"
    ).numpy().copy()
    frame.set_target(base_poses, configuration=cfg)

    tasks = [frame, posture]
    solver = mw.IKSolver(cfg)

    # Even grid for visualization only (IK remains in a shared local frame).
    origins = _env_origins(NUM_WORLDS)
    grid_extent = float(np.max(np.linalg.norm(origins[:, :2], axis=1)) + ENV_SPACING)

    server = viser.ViserServer(label="mink-warp batched Panda IK")
    scene = ViserMujocoScene(server, model, num_envs=NUM_WORLDS)
    if hasattr(scene, "create_scene_gui"):
        scene.create_scene_gui(
            camera_distance=max(2.5, 1.2 * grid_extent),
            camera_azimuth=120.0,
            camera_elevation=-25.0,
        )
    elif hasattr(scene, "create_visualization_gui"):
        scene.create_visualization_gui()

    print(f"Panda XML: {xml_path}")
    print(
        f"Open the viser URL. Tracking {NUM_WORLDS} Pandas on a "
        f"{ENV_SPACING}m grid @ {FREQUENCY} Hz."
    )
    print("Ctrl+C to stop.")

    t0 = time.time()
    period = 1.0 / FREQUENCY

    try:
        while True:
            loop_start = time.time()
            t = loop_start - t0
            targets = _target_poses(t, NUM_WORLDS, base_poses)
            frame.set_target(targets, configuration=cfg)

            for _ in range(IK_ITERS):
                v = solver.solve(tasks, DT, damping=1e-3)
                cfg.integrate_inplace(v, DT)

            # Offset body positions for display after IK (do not affect solve).
            xpos = cfg.wp_data.xpos.numpy().copy()
            xpos += origins[:, None, :]

            scene.update_from_arrays(
                xpos,
                cfg.wp_data.xmat.numpy(),
                qpos=cfg.q.numpy(),
            )

            elapsed = time.time() - loop_start
            if elapsed < period:
                time.sleep(period - elapsed)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
