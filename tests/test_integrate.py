"""Device integrate matches MuJoCo mj_integratePos."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest

import mink_warp as mw


def test_integrate_matches_mujoco(arm_model):
    q = arm_model.key_qpos[0].copy()
    v = np.array([0.3, -0.5], dtype=np.float64)
    dt = 0.02

    q_mj = q.copy()
    mujoco.mj_integratePos(arm_model, q_mj, v, dt)

    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    q_wp = cfg.integrate(v, dt).numpy()[0]
    np.testing.assert_allclose(q_wp, q_mj, atol=1e-5)


def test_integrate_inplace_updates_kinematics(arm_model):
    q = arm_model.key_qpos[0].copy()
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    pose0 = cfg.get_transform_frame_to_world("ee", "site").numpy()[0].copy()

    v = np.array([1.0, -0.5], dtype=np.float32)
    cfg.integrate_inplace(v, dt=0.05)
    pose1 = cfg.get_transform_frame_to_world("ee", "site").numpy()[0]
    assert not np.allclose(pose0[4:], pose1[4:], atol=1e-5)


def test_integrate_batched(arm_model):
    q0 = arm_model.key_qpos[0].copy()
    q1 = q0 + np.array([0.1, -0.2])
    v = np.stack(
        [np.array([0.2, -0.1]), np.array([-0.3, 0.4])],
        axis=0,
    )
    dt = 0.01

    expected = np.stack([q0.copy(), q1.copy()])
    for i in range(2):
        mujoco.mj_integratePos(arm_model, expected[i], v[i], dt)

    cfg = mw.Configuration(arm_model, q=np.stack([q0, q1]), nworld=2)
    q_wp = cfg.integrate(v, dt).numpy()
    np.testing.assert_allclose(q_wp, expected, atol=1e-5)
