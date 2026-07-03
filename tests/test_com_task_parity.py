"""ComTask matches Mink at B=1."""

from __future__ import annotations

import numpy as np
import pytest

mink = pytest.importorskip("mink")

import mink_warp as mw


def test_com_task_matches_mink(arm_model):
    q = arm_model.key_qpos[0].copy()
    mk_cfg = mink.Configuration(arm_model, q=q)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)

    target = mk_cfg.data.subtree_com[1] + np.array([0.02, -0.01, 0.03])

    mk_task = mink.ComTask(cost=1.0)
    mk_task.set_target(target)

    task = mw.ComTask(cost=1.0)
    task.set_target(target, configuration=cfg)

    np.testing.assert_allclose(
        task.compute_error(cfg).numpy()[0],
        mk_task.compute_error(mk_cfg),
        atol=1e-5,
    )
    np.testing.assert_allclose(
        task.compute_jacobian(cfg).numpy()[0],
        mk_task.compute_jacobian(mk_cfg),
        atol=1e-4,
    )


def test_com_set_target_from_configuration(arm_model):
    q = arm_model.key_qpos[0].copy()
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task = mw.ComTask(cost=1.0)
    task.set_target_from_configuration(cfg)
    err = task.compute_error(cfg).numpy()
    np.testing.assert_allclose(err, 0.0, atol=1e-6)
