"""Smoke test for mink-warp package artifacts (wheel / sdist)."""

from __future__ import annotations

import importlib.metadata as md
import os
import sys
import traceback

# Before mujoco import on headless CI.
os.environ.setdefault("MUJOCO_GL", "disable")

_ARM_XML = """
<mujoco model="planar_arm">
  <compiler angle="radian"/>
  <worldbody>
    <body name="link1" pos="0 0 0.1">
      <joint name="joint1" type="hinge" axis="0 0 1" range="-3.14 3.14" limited="true"/>
      <geom type="capsule" fromto="0 0 0 0.3 0 0" size="0.03"/>
      <body name="link2" pos="0.3 0 0">
        <joint name="joint2" type="hinge" axis="0 0 1" range="-3.14 3.14" limited="true"/>
        <geom type="capsule" fromto="0 0 0 0.25 0 0" size="0.025"/>
        <site name="ee" pos="0.25 0 0" size="0.02"/>
      </body>
    </body>
  </worldbody>
  <keyframe>
    <key name="home" qpos="0.4 -0.7"/>
  </keyframe>
</mujoco>
"""


def test_package_metadata() -> None:
    """Installed artifact exposes mink_warp and version metadata."""
    import mink_warp as mw

    assert mw.__name__ == "mink_warp"
    assert mw.__version__

    dist = md.distribution("mink-warp")
    assert dist.version == mw.__version__
    assert "FrameTask" in dir(mw)
    assert "solve_ik" in dir(mw)


def test_one_ik_step_cpu() -> None:
    """Minimal CPU IK step on a tiny model."""
    import mujoco

    import mink_warp as mw

    model = mujoco.MjModel.from_xml_string(_ARM_XML)
    cfg = mw.Configuration(model, nworld=2, device="cpu")
    frame = mw.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
    frame.set_target_from_configuration(cfg)
    posture = mw.PostureTask(model, cost=1e-2)
    posture.set_target_from_configuration(cfg)

    v = mw.solve_ik(cfg, [frame, posture], dt=0.01)
    assert v.shape == (2, model.nv)


if __name__ == "__main__":
    try:
        test_package_metadata()
        test_one_ik_step_cpu()
        print("Smoke test passed!")
        sys.exit(0)
    except Exception:
        print("Smoke test failed:")
        traceback.print_exc()
        sys.exit(1)
