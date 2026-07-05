"""Single-world IK loop used in the Sphinx quickstart tutorial."""

from __future__ import annotations

import mujoco
import numpy as np

import mink_warp as mw
from mink_warp import SE3, SO3

model = mujoco.MjModel.from_xml_path("franka_emika_panda/mjx_scene.xml")
cfg = mw.Configuration(model, nworld=1)
cfg.update_from_keyframe("home")

ee_pose = cfg.get_transform_frame_to_world_se3("attachment_site", "site")
translation = np.array([0.0, -0.4, -0.2])
rotation = SO3.exp(np.array([0.0, -np.pi / 2, 0.0]))
target = SE3.from_translation(translation) @ ee_pose
target = target @ SE3.from_rotation(rotation)

duration, fps = 2.0, 60
n_frames = int(duration * fps)
gain = 1.0 - 0.01 ** (1.0 / n_frames)

frame = mw.FrameTask(
    frame_name="attachment_site",
    frame_type="site",
    position_cost=1.0,
    orientation_cost=1.0,
    gain=gain,
)
frame.set_target(target)
posture = mw.PostureTask(model, cost=1e-2)
posture.set_target_from_configuration(cfg)

dt = 1.0 / fps
for _ in range(n_frames):
    v = mw.solve_ik(cfg, [frame, posture], dt)
    cfg.integrate_inplace(v, dt)

final = cfg.get_transform_frame_to_world_se3("attachment_site", "site")
print(
    f"Position error: {np.linalg.norm(final.translation() - target.translation()):.2e} m"
)
