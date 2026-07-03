"""Batched differential IK on Unitree G1, visualized with mjviser.

Random hands / feet (stance|swing) / squat / torso lean are re-sampled every
``RESAMPLE_ITERS`` steps. Soft joint limits enabled. CUDA graph when available.

Run:
  uv sync --extra dev --extra examples
  uv run examples/batched_g1_ik.py
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
ENV_SPACING = 2.0
RESAMPLE_ITERS = 100

HAND_XY_RANGE = 0.12
HAND_Z_RANGE = 0.18
AMP_SQUAT = 0.12
AMP_LEAN_PITCH = 0.35
AMP_LEAN_ROLL = 0.20
SWING_Z_MIN, SWING_Z_MAX = 0.10, 0.20
SWING_XY_RANGE = 0.06


def main() -> None:
    xml = Path(__file__).resolve().parent / "unitree_g1" / "scene.xml"
    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    cfg = mw.Configuration(model, nworld=NUM_WORLDS)
    cfg.update_from_keyframe("stand")

    def frame(name, ftype, pos_cost, ori_cost):
        return mw.FrameTask(
            name, ftype, position_cost=pos_cost, orientation_cost=ori_cost,
            gain=0.7, lm_damping=1.0,
        )

    pelvis = frame("pelvis", "body", [3.0, 3.0, 5.0], 1.0)
    torso = frame("torso_link", "body", 0.0, 2.0)
    left_foot = frame("left_foot", "site", [15.0, 15.0, 30.0], 3.0)
    right_foot = frame("right_foot", "site", [15.0, 15.0, 30.0], 3.0)
    left_hand = frame("left_palm", "site", 2.0, 0.5)
    right_hand = frame("right_palm", "site", 2.0, 0.5)
    posture = mw.PostureTask(model, cost=5e-2)
    com = mw.ComTask(cost=[10.0, 10.0, 10.0])
    limits = mw.JointLimitTask(model, cost=1.0)

    for t in (pelvis, torso, left_foot, right_foot, left_hand, right_hand):
        t.set_target_from_configuration(cfg)
    posture.set_target_from_configuration(cfg)
    com.set_target_from_configuration(cfg)

    # Stand references (local frame).
    base_hand_l = cfg.get_transform_frame_to_world("left_palm", "site").numpy().copy()
    base_hand_r = cfg.get_transform_frame_to_world("right_palm", "site").numpy().copy()
    base_pelvis = cfg.get_transform_frame_to_world("pelvis", "body").numpy().copy()
    base_torso = cfg.get_transform_frame_to_world("torso_link", "body").numpy().copy()
    base_foot_l = cfg.get_transform_frame_to_world("left_foot", "site").numpy().copy()
    base_foot_r = cfg.get_transform_frame_to_world("right_foot", "site").numpy().copy()
    ground_z = 0.5 * (base_foot_l[:, 6] + base_foot_r[:, 6])
    base_foot_l[:, 6] = ground_z
    base_foot_r[:, 6] = ground_z
    com_height = cfg.wp_data.subtree_com.numpy()[:, 1, 2] - ground_z

    tasks = [
        pelvis, torso, left_foot, right_foot, left_hand, right_hand,
        posture, com, limits,
    ]
    solver = mw.IKSolver(cfg)

    # Even grid for display only.
    cols = math.ceil(math.sqrt(NUM_WORLDS))
    rows = math.ceil(NUM_WORLDS / cols)
    origins = np.zeros((NUM_WORLDS, 3), dtype=np.float32)
    for i in range(NUM_WORLDS):
        r, c = divmod(i, cols)
        origins[i, 0] = (c - 0.5 * (cols - 1)) * ENV_SPACING
        origins[i, 1] = (r - 0.5 * (rows - 1)) * ENV_SPACING

    server = viser.ViserServer(label="mink-warp batched G1 IK")
    scene = ViserMujocoScene(server, model, num_envs=NUM_WORLDS)
    extent = float(np.max(np.linalg.norm(origins[:, :2], axis=1)) + ENV_SPACING)
    if hasattr(scene, "create_scene_gui"):
        scene.create_scene_gui(
            camera_distance=max(3.0, 1.3 * extent),
            camera_azimuth=180.0,
            camera_elevation=-20.0,
        )

    print(f"G1 XML: {xml}")
    print(f"Open the viser URL. {NUM_WORLDS} G1s, resample every {RESAMPLE_ITERS}.")
    print("Ctrl+C to stop.")

    rng = np.random.default_rng(0)
    n = NUM_WORLDS
    iteration = 0
    period = 1.0 / FREQUENCY

    try:
        while True:
            t0 = time.time()

            if iteration % RESAMPLE_ITERS == 0:
                # Hands.
                hand_l, hand_r = base_hand_l.copy(), base_hand_r.copy()
                hand_l[:, 4:6] += rng.uniform(-HAND_XY_RANGE, HAND_XY_RANGE, (n, 2))
                hand_l[:, 6] += rng.uniform(-HAND_Z_RANGE, HAND_Z_RANGE, n)
                hand_r[:, 4:6] += rng.uniform(-HAND_XY_RANGE, HAND_XY_RANGE, (n, 2))
                hand_r[:, 6] += rng.uniform(-HAND_Z_RANGE, HAND_Z_RANGE, n)

                # Squat.
                squat = rng.uniform(0.0, AMP_SQUAT, n).astype(np.float32)
                hand_l[:, 6] -= squat
                hand_r[:, 6] -= squat

                # Feet: 0=both stance, 1=left swing, 2=right swing.
                mode = rng.choice(3, size=n, p=(0.4, 0.3, 0.3))
                left_stance = mode != 1
                right_stance = mode != 2
                foot_l, foot_r = base_foot_l.copy(), base_foot_r.copy()
                swing_z = rng.uniform(SWING_Z_MIN, SWING_Z_MAX, n).astype(np.float32)
                swing_xy = rng.uniform(-SWING_XY_RANGE, SWING_XY_RANGE, (n, 2)).astype(
                    np.float32
                )
                swing_l, swing_r = ~left_stance, ~right_stance
                foot_l[swing_l, 4:6] += swing_xy[swing_l]
                foot_l[swing_l, 6] = ground_z[swing_l] + swing_z[swing_l]
                foot_r[swing_r, 4:6] += swing_xy[swing_r]
                foot_r[swing_r, 6] = ground_z[swing_r] + swing_z[swing_r]

                # Torso lean.
                torso_poses = np.empty_like(base_torso)
                pitch = rng.uniform(-AMP_LEAN_PITCH, AMP_LEAN_PITCH, n)
                roll = rng.uniform(-AMP_LEAN_ROLL, AMP_LEAN_ROLL, n)
                for i in range(n):
                    R = mw.SO3.exp(np.array([roll[i], pitch[i], 0.0]))
                    T = mw.SE3(wxyz_xyz=base_torso[i].astype(float)) @ mw.SE3.from_rotation(R)
                    torso_poses[i] = T.wxyz_xyz.astype(np.float32)

                # CoM / pelvis over stance feet.
                w_l = left_stance.astype(np.float32)
                w_r = right_stance.astype(np.float32)
                both = (w_l + w_r) < 0.5
                w_l = np.where(both, 1.0, w_l)
                w_r = np.where(both, 1.0, w_r)
                w = w_l + w_r
                stance_xy = (
                    foot_l[:, 4:6] * w_l[:, None] + foot_r[:, 4:6] * w_r[:, None]
                ) / w[:, None]
                support_z = np.where(
                    left_stance & right_stance,
                    0.5 * (foot_l[:, 6] + foot_r[:, 6]),
                    np.where(left_stance, foot_l[:, 6], foot_r[:, 6]),
                )

                pelvis_t = base_pelvis.copy()
                pelvis_t[:, 4:6] = stance_xy
                pelvis_t[:, 6] -= squat

                com_t = np.empty((n, 3), dtype=np.float32)
                com_t[:, 0:2] = stance_xy
                com_t[:, 2] = support_z + com_height - squat

                # Upload once per resample.
                left_foot.set_target(foot_l, configuration=cfg)
                right_foot.set_target(foot_r, configuration=cfg)
                pelvis.set_target(pelvis_t, configuration=cfg)
                torso.set_target(torso_poses, configuration=cfg)
                com.set_target(com_t, configuration=cfg)
                left_hand.set_target(hand_l, configuration=cfg)
                right_hand.set_target(hand_r, configuration=cfg)

            solver.solve_and_integrate(
                tasks, DT, damping=1e-1, iterations=1, use_graph=True
            )

            xpos = cfg.wp_data.xpos.numpy().copy()
            xpos += origins[:, None, :]
            scene.update_from_arrays(xpos, cfg.wp_data.xmat.numpy(), qpos=cfg.q.numpy())

            iteration += 1
            dt_loop = time.time() - t0
            if dt_loop < 1.0 / FREQUENCY:
                time.sleep(1.0 / FREQUENCY - dt_loop)

    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
