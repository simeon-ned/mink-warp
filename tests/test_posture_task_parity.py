"""PostureTask and DampingTask match Mink at B=1."""

from __future__ import annotations

import numpy as np
import pytest

mink = pytest.importorskip("mink")

import mink_warp as mw


def test_posture_task_matches_mink(arm_model):
    q = arm_model.key_qpos[0].copy()
    target = q + np.array([0.3, -0.2])

    mk_cfg = mink.Configuration(arm_model, q=q)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)

    mk_task = mink.PostureTask(arm_model, cost=1e-2)
    mk_task.set_target(target)

    task = mw.PostureTask(arm_model, cost=1e-2)
    # Optional NumPy upload.
    task.set_target(target)

    np.testing.assert_allclose(
        task.compute_error(cfg).numpy()[0],
        mk_task.compute_error(mk_cfg),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        task.compute_jacobian(cfg).numpy()[0],
        mk_task.compute_jacobian(mk_cfg),
        atol=1e-12,
    )


def test_posture_task_device_target(arm_model):
    q = arm_model.key_qpos[0].copy()
    target = q + np.array([0.3, -0.2])
    cfg = mw.Configuration(arm_model, q=q, nworld=1)

    task = mw.PostureTask(arm_model, cost=1e-2)
    target_wp = mw.to_wp(target.astype(np.float32), device=cfg.device)
    task.set_target(target_wp, configuration=cfg)

    err = task.compute_error(cfg).numpy()[0]
    np.testing.assert_allclose(err, q - target, atol=1e-6)


def test_posture_set_target_from_configuration(arm_model):
    q = arm_model.key_qpos[0].copy()
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task = mw.PostureTask(arm_model, cost=1e-2)
    task.set_target_from_configuration(cfg)
    err = task.compute_error(cfg).numpy()
    np.testing.assert_allclose(err, 0.0, atol=1e-6)


def test_damping_task_matches_mink(arm_model):
    q = arm_model.key_qpos[0].copy()
    mk_cfg = mink.Configuration(arm_model, q=q)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)

    mk_task = mink.DampingTask(arm_model, cost=1.0)
    task = mw.DampingTask(arm_model, cost=1.0)

    np.testing.assert_allclose(
        task.compute_error(cfg).numpy()[0],
        mk_task.compute_error(mk_cfg),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        task.compute_jacobian(cfg).numpy()[0],
        mk_task.compute_jacobian(mk_cfg),
        atol=1e-12,
    )
