"""Configuration kinematics and Jacobians match Mink at B=1."""

from __future__ import annotations

import numpy as np
import pytest

mink = pytest.importorskip("mink")

import mink_warp as mw


@pytest.fixture
def configs(arm_model):
    q = arm_model.key_qpos[0].copy()
    mk = mink.Configuration(arm_model, q=q)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    return mk, cfg


@pytest.mark.parametrize("frame_name,frame_type", [("ee", "site"), ("link2", "body")])
def test_frame_pose_matches_mink(configs, frame_name, frame_type):
    mk, cfg = configs
    T_mk = mk.get_transform_frame_to_world(frame_name, frame_type)
    T_wp = cfg.get_transform_frame_to_world_se3(frame_name, frame_type)
    np.testing.assert_allclose(T_wp.wxyz_xyz, T_mk.wxyz_xyz, atol=1e-5)

    pose = cfg.get_transform_frame_to_world(frame_name, frame_type).numpy()
    np.testing.assert_allclose(pose[0], T_mk.wxyz_xyz, atol=1e-5)


@pytest.mark.parametrize("frame_name,frame_type", [("ee", "site"), ("link2", "body")])
def test_frame_jacobian_matches_mink(configs, frame_name, frame_type):
    mk, cfg = configs
    J_mk = mk.get_frame_jacobian(frame_name, frame_type)
    J_wp = cfg.get_frame_jacobian(frame_name, frame_type).numpy()
    assert J_wp.shape == (1, 6, cfg.nv)
    np.testing.assert_allclose(J_wp[0], J_mk, atol=1e-4)


def test_batched_update_independent_worlds(arm_model):
    q0 = arm_model.key_qpos[0].copy()
    q1 = q0 + np.array([0.2, -0.1])
    cfg = mw.Configuration(arm_model, q=np.stack([q0, q1]), nworld=2)

    mk0 = mink.Configuration(arm_model, q=q0)
    mk1 = mink.Configuration(arm_model, q=q1)

    J = cfg.get_frame_jacobian("ee", "site").numpy()
    np.testing.assert_allclose(
        J[0], mk0.get_frame_jacobian("ee", "site"), atol=1e-4
    )
    np.testing.assert_allclose(
        J[1], mk1.get_frame_jacobian("ee", "site"), atol=1e-4
    )


def test_update_from_device_q(arm_model):
    q0 = arm_model.key_qpos[0].copy()
    cfg = mw.Configuration(arm_model, q=q0, nworld=1)
    q1 = q0 + np.array([0.1, -0.05])
    import warp as wp

    with wp.ScopedDevice(cfg.device):
        q_wp = wp.array(q1.astype(np.float32), dtype=float)
    cfg.update(q=q_wp)
    np.testing.assert_allclose(cfg.q.numpy()[0], q1, atol=1e-6)
