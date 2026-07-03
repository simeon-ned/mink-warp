"""Soft configuration limit task behavior."""

from __future__ import annotations

import numpy as np

import mink_warp as mw


def test_limit_task_zero_inside(arm_model):
    q = arm_model.key_qpos[0].copy()
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task = mw.ConfigurationLimitTask(arm_model, cost=1.0)
    err = task.compute_error(cfg).numpy()[0]
    np.testing.assert_allclose(err, 0.0, atol=1e-8)


def test_limit_task_penalizes_violation(arm_model):
    q = arm_model.key_qpos[0].copy()
    # Push joint 0 past its upper limit.
    upper = float(arm_model.jnt_range[0, 1])
    q[0] = upper + 0.2
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task = mw.ConfigurationLimitTask(arm_model, cost=1.0)
    err = task.compute_error(cfg).numpy()[0]
    assert err[0] > 0.15
    jac = task.compute_jacobian(cfg).numpy()[0]
    assert jac[0, 0] == 1.0
