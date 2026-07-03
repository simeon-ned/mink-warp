"""Shared fixtures."""

from __future__ import annotations

import mujoco
import pytest

# Simple 2-DoF planar arm with a site on the end-effector.
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


@pytest.fixture(scope="module")
def arm_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_string(_ARM_XML)
