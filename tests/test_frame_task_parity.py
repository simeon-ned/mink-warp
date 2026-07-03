"""FrameTask error and Jacobian match Mink at B=1."""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

mink = pytest.importorskip("mink")

import mink_warp as mw


def test_frame_task_error_and_jacobian_match_mink(arm_model):
    q = arm_model.key_qpos[0].copy()
    mk_cfg = mink.Configuration(arm_model, q=q)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)

    target = mink.SE3.from_translation(np.array([0.4, 0.1, 0.1]))
    target = target @ mink.SE3.from_rotation(mink.SO3.exp(np.array([0.0, 0.0, 0.3])))

    mk_task = mink.FrameTask(
        frame_name="ee",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.5,
        lm_damping=1.0,
    )
    mk_task.set_target(target)

    task = mw.FrameTask(
        frame_name="ee",
        frame_type="site",
        position_cost=1.0,
        orientation_cost=0.5,
        lm_damping=1.0,
    )
    # Optional host upload (SE3).
    task.set_target(mw.SE3(wxyz_xyz=target.wxyz_xyz.copy()))

    e_mk = mk_task.compute_error(mk_cfg)
    e_wp = task.compute_error(cfg).numpy()
    np.testing.assert_allclose(e_wp[0], e_mk, atol=1e-5)

    J_mk = mk_task.compute_jacobian(mk_cfg)
    J_wp = task.compute_jacobian(cfg).numpy()
    np.testing.assert_allclose(J_wp[0], J_mk, atol=1e-4)

    W_mk, eW_mk, mu_mk = mk_task.compute_qp_residual(mk_cfg)
    W_wp, eW_wp, mu_wp = task.compute_qp_residual(cfg)
    np.testing.assert_allclose(W_wp.numpy()[0], W_mk, atol=1e-4)
    np.testing.assert_allclose(eW_wp.numpy()[0], eW_mk, atol=1e-5)
    np.testing.assert_allclose(mu_wp.numpy()[0], mu_mk, atol=1e-8)


def test_frame_task_set_target_from_configuration(arm_model):
    q = arm_model.key_qpos[0].copy()
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task = mw.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
    task.set_target_from_configuration(cfg)
    err = task.compute_error(cfg).numpy()
    np.testing.assert_allclose(err, 0.0, atol=1e-6)


def test_frame_task_device_target(arm_model):
    q0 = arm_model.key_qpos[0].copy()
    q1 = q0 + np.array([0.15, -0.2])
    cfg = mw.Configuration(arm_model, q=np.stack([q0, q1]), nworld=2)

    task = mw.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
    task.set_target_from_configuration(cfg)
    err = task.compute_error(cfg).numpy()
    np.testing.assert_allclose(err, 0.0, atol=1e-6)

    # Device-native target update (no NumPy in the set_target path).
    poses = cfg.get_transform_frame_to_world("ee", "site")
    with wp.ScopedDevice(cfg.device):
        targets = wp.zeros((2, 7), dtype=float)
        wp.copy(targets, poses)
    # Perturb world 1 on device via host write to a fresh buffer, then upload once.
    t_np = targets.numpy()
    t_np[1] = (
        mw.SE3(wxyz_xyz=t_np[1])
        .plus(np.array([0.05, 0, 0, 0, 0, 0.1]))
        .wxyz_xyz.astype(np.float32)
    )
    targets_wp = mw.to_wp(t_np, device=cfg.device)
    task.set_target(targets_wp, configuration=cfg)

    err = task.compute_error(cfg).numpy()
    np.testing.assert_allclose(err[0], 0.0, atol=1e-6)
    assert np.linalg.norm(err[1]) > 1e-3
