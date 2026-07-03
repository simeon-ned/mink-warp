"""solve_ik matches Mink unconstrained DLS at B=1."""

from __future__ import annotations

import numpy as np
import pytest

mink = pytest.importorskip("mink")

import mink_warp as mw


def test_solve_ik_matches_mink_soft_tasks(arm_model):
    q = arm_model.key_qpos[0].copy()
    target_q = q + np.array([0.2, -0.15])

    mk_cfg = mink.Configuration(arm_model, q=q)
    cfg = mw.Configuration(arm_model, q=q, nworld=1)

    mk_frame = mink.FrameTask(
        "ee", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0
    )
    mk_frame.set_target_from_configuration(mk_cfg)
    # Move target away.
    T = mk_cfg.get_transform_frame_to_world("ee", "site")
    T = T @ mink.SE3.from_translation(np.array([0.05, 0.02, 0.0]))
    mk_frame.set_target(T)
    mk_posture = mink.PostureTask(arm_model, cost=1e-2)
    mk_posture.set_target(target_q)

    frame = mw.FrameTask(
        "ee", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=1.0
    )
    frame.set_target(mw.SE3(wxyz_xyz=T.wxyz_xyz.copy()))
    posture = mw.PostureTask(arm_model, cost=1e-2)
    posture.set_target(target_q)

    dt = 0.01
    damping = 1e-6
    v_mk = mink.solve_ik(
        mk_cfg,
        [mk_frame, mk_posture],
        dt,
        solver="daqp",
        damping=damping,
        limits=[],
    )
    v_wp = mw.solve_ik(cfg, [frame, posture], dt, damping=damping).numpy()[0]
    np.testing.assert_allclose(v_wp, v_mk, atol=1e-4)


def test_solve_ik_batched_independent(arm_model):
    q0 = arm_model.key_qpos[0].copy()
    q1 = q0 + np.array([0.1, -0.2])
    cfg = mw.Configuration(arm_model, q=np.stack([q0, q1]), nworld=2)

    frame = mw.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
    frame.set_target_from_configuration(cfg)
    # Perturb only world 1 target.
    T = cfg.get_transform_frame_to_world("ee", "site").numpy().copy()
    T[1, 4] += 0.05
    frame.set_target(T, configuration=cfg)

    posture = mw.PostureTask(arm_model, cost=1e-2)
    posture.set_target_from_configuration(cfg)

    solver = mw.IKSolver(cfg)
    v = solver.solve([frame, posture], dt=0.01, damping=1e-4).numpy()
    # World 0 at target => near-zero velocity (posture also at target).
    np.testing.assert_allclose(v[0], 0.0, atol=1e-4)
    assert np.linalg.norm(v[1]) > 1e-3


def test_solve_ik_iterations_reduces_error(arm_model):
    q = arm_model.key_qpos[0].copy()
    cfg = mw.Configuration(arm_model, q=q, nworld=1)

    # gain < 1 takes damped steps so the linearization stays valid.
    frame = mw.FrameTask(
        "ee",
        "site",
        position_cost=1.0,
        orientation_cost=0.0,
        gain=0.5,
        lm_damping=0.0,
    )
    T = cfg.get_transform_frame_to_world("ee", "site").numpy()[0].copy()
    T[4] += 0.05
    T[5] += 0.02
    frame.set_target(T, configuration=cfg)

    posture = mw.PostureTask(arm_model, cost=1e-4)
    posture.set_target_from_configuration(cfg)

    e0 = float(np.linalg.norm(frame.compute_error(cfg).numpy()[0, :3]))
    mw.solve_ik_iterations(
        cfg, [frame, posture], dt=0.01, iterations=40, damping=1e-4
    )
    e1 = float(np.linalg.norm(frame.compute_error(cfg).numpy()[0, :3]))
    assert e1 < e0 * 0.5
