"""Interchangeable solver backends: DLS, Levenberg-Marquardt, L-BFGS.

All backends minimise the same weighted task cost and share the
``solve_and_integrate`` API, so the same reachable target must drive every one
to the same configuration; and each must respect per-world independence.
"""

from __future__ import annotations

import numpy as np
import pytest

import mink_warp as mw
from mink_warp.solvers import DLSSolver, LBFGSSolver, LMSolver, Solver

OPTIMIZERS = ["lm", "lbfgs"]
ALL_KINDS = ["dls", "lm", "lbfgs"]


def _ee_pos(cfg) -> np.ndarray:
    return cfg.get_transform_frame_to_world("ee", "site").numpy()[:, 4:7]


def _reachable_target(cfg) -> np.ndarray:
    """A target inside the 2-DoF workspace (pull the EE inward)."""
    tgt = cfg.get_transform_frame_to_world("ee", "site").numpy().copy()
    tgt[:, 4] -= 0.03
    tgt[:, 5] += 0.02
    return tgt


# --- API surface -----------------------------------------------------------


def test_registry_maps_names_to_backends(arm_model):
    cfg = mw.Configuration(arm_model, nworld=1)
    assert isinstance(mw.make_solver(cfg, "dls"), DLSSolver)
    assert isinstance(mw.make_solver(cfg, "lm"), LMSolver)
    assert isinstance(mw.make_solver(cfg, "lbfgs"), LBFGSSolver)
    # Every backend is a Solver with the shared entry point.
    for kind in ALL_KINDS:
        s = mw.make_solver(cfg, kind)
        assert isinstance(s, Solver)
        assert hasattr(s, "solve_and_integrate")


def test_ik_solver_alias_is_dls():
    assert mw.IKSolver is DLSSolver


def test_unknown_solver_raises(arm_model):
    cfg = mw.Configuration(arm_model, nworld=1)
    with pytest.raises(ValueError, match="unknown solver"):
        mw.make_solver(cfg, "nope")


@pytest.mark.parametrize("kind", OPTIMIZERS)
def test_iterations_must_be_positive(arm_model, kind):
    cfg = mw.Configuration(arm_model, nworld=1)
    frame = mw.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
    frame.set_target_from_configuration(cfg)
    solver = mw.make_solver(cfg, kind)
    with pytest.raises(ValueError, match="iterations"):
        solver.solve_and_integrate([frame], 0.01, iterations=0)


def test_lbfgs_history_must_be_positive(arm_model):
    cfg = mw.Configuration(arm_model, nworld=1)
    with pytest.raises(ValueError, match="history"):
        LBFGSSolver(cfg, history=0)


# --- Convergence -----------------------------------------------------------


@pytest.mark.parametrize("kind", OPTIMIZERS)
def test_optimizer_reaches_reachable_target(arm_model, kind):
    q0 = arm_model.key_qpos[0].copy()
    cfg = mw.Configuration(arm_model, q=q0, nworld=1)
    frame = mw.FrameTask("ee", "site", position_cost=1.0, orientation_cost=0.0)
    tgt = _reachable_target(cfg)
    frame.set_target(tgt, configuration=cfg)
    posture = mw.PostureTask(arm_model, cost=1e-6)
    posture.set_target_from_configuration(cfg)

    solver = mw.make_solver(cfg, kind)
    solver.solve_and_integrate([frame, posture], 0.01, iterations=40)

    dpos = np.linalg.norm(_ee_pos(cfg)[0] - tgt[0, 4:7])
    assert dpos < 1e-4


def test_all_backends_agree_at_fixed_point(arm_model):
    """DLS (many small steps), LM, and L-BFGS reach the same EE pose."""
    q0 = arm_model.key_qpos[0].copy()
    ee = {}
    for kind in ALL_KINDS:
        cfg = mw.Configuration(arm_model, q=q0, nworld=1)
        frame = mw.FrameTask(
            "ee", "site", position_cost=1.0, orientation_cost=0.0, gain=0.5
        )
        tgt = _reachable_target(cfg)
        frame.set_target(tgt, configuration=cfg)
        posture = mw.PostureTask(arm_model, cost=1e-6)
        posture.set_target_from_configuration(cfg)
        solver = mw.make_solver(cfg, kind)
        iters = 400 if kind == "dls" else 60
        solver.solve_and_integrate([frame, posture], 0.02, iterations=iters)
        ee[kind] = _ee_pos(cfg)[0]

    np.testing.assert_allclose(ee["lm"], ee["dls"], atol=1e-3)
    np.testing.assert_allclose(ee["lbfgs"], ee["dls"], atol=1e-3)


# --- Batched independence --------------------------------------------------


@pytest.mark.parametrize("kind", OPTIMIZERS)
def test_batched_worlds_are_independent(arm_model, kind):
    q0 = arm_model.key_qpos[0].copy()
    cfg = mw.Configuration(arm_model, q=np.stack([q0, q0]), nworld=2)
    frame = mw.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
    frame.set_target_from_configuration(cfg)
    tgt = cfg.get_transform_frame_to_world("ee", "site").numpy().copy()
    tgt[1, 4] -= 0.03  # perturb only world 1
    frame.set_target(tgt, configuration=cfg)
    posture = mw.PostureTask(arm_model, cost=1e-4)
    posture.set_target_from_configuration(cfg)

    solver = mw.make_solver(cfg, kind)
    solver.solve_and_integrate([frame, posture], 0.01, iterations=20)

    q = cfg.q.numpy()
    assert np.linalg.norm(q[0] - q0) < 1e-5  # world 0 already at target
    assert np.linalg.norm(q[1] - q0) > 1e-2  # world 1 moved


# --- LM reduces to the Gauss-Newton step -----------------------------------


def test_lm_first_step_matches_dls_gauss_newton(arm_model):
    """With tiny damping and a small target, one LM step == one DLS step."""
    q0 = arm_model.key_qpos[0].copy()
    tgt = None
    dq = {}
    for kind in ("dls", "lm"):
        cfg = mw.Configuration(arm_model, q=q0, nworld=1)
        frame = mw.FrameTask(
            "ee", "site", position_cost=1.0, orientation_cost=1.0, lm_damping=0.0
        )
        if tgt is None:
            t = cfg.get_transform_frame_to_world("ee", "site").numpy()[0].copy()
            t[4] += 0.005
            tgt = t
        frame.set_target(tgt, configuration=cfg)
        if kind == "dls":
            solver = DLSSolver(cfg, damping=1e-9)
            v = solver.solve([frame], 0.01).numpy()[0]
        else:
            solver = LMSolver(cfg, lambda0=1e-9)
            v = solver.solve_and_integrate([frame], 0.01, iterations=1).numpy()[0]
        dq[kind] = v * 0.01

    np.testing.assert_allclose(dq["lm"], dq["dls"], atol=1e-4)
