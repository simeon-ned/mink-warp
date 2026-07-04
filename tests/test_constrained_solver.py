"""ConstrainedSolver: hard joint limits via box-ADMM.

Two layers:

* **CPU** — the QP *assembly* (H, b, box, rho) runs on any device, so we assemble
  it, solve the box QP with the NumPy ADMM oracle, and check feasibility and
  agreement with mink's daqp QP. This validates the entire pipeline except the
  GPU-only tile-Cholesky ADMM kernel.
* **kernel** — the real box-ADMM tile kernel (runs on CPU or GPU) is exercised:
  feasibility at any iteration count, the limit actually binding when pushed,
  agreement with mink, velocity limits, and batched independence.
"""

from __future__ import annotations

import numpy as np
import pytest

mink = pytest.importorskip("mink")

from helpers_admm import box_qp_admm  # noqa: E402

import mink_warp as mw  # noqa: E402

DT = 0.02
GAIN = 0.95


# --------------------------------------------------------------------------- #
# API / construction (device-agnostic)
# --------------------------------------------------------------------------- #
def test_registry_and_defaults(arm_model):
    cfg = mw.Configuration(arm_model, nworld=1)
    s = mw.make_solver(cfg, "constrained")
    assert isinstance(s, mw.ConstrainedSolver)
    assert isinstance(s, mw.Solver)
    assert [type(x).__name__ for x in s.limits] == ["ConfigurationLimit"]


def test_admm_iters_guard(arm_model):
    cfg = mw.Configuration(arm_model, nworld=1)
    with pytest.raises(ValueError):
        mw.ConstrainedSolver(cfg, admm_iters=0)


def test_empty_limits_allowed(arm_model):
    cfg = mw.Configuration(arm_model, nworld=1)
    s = mw.ConstrainedSolver(cfg, limits=[])
    assert s.limits == []


# --------------------------------------------------------------------------- #
# CPU: assembled QP matches mink; box QP feasible + agrees with daqp
# --------------------------------------------------------------------------- #
def _posture_pushing_past_limit(model, q, q_target, cost=1.0):
    """mw + mink posture tasks driving q toward a target beyond the joint range."""
    mw_task = mw.PostureTask(model, cost=cost)
    mw_task.set_target(q_target)
    mk_task = mink.PostureTask(model, cost=cost)
    mk_task.set_target(q_target)
    return mw_task, mk_task


def test_assembly_matches_mink_qp_active_limit(arm_model):
    # joint1 near its +limit; posture target well beyond it -> limit binds.
    q = np.array([2.9, 0.0])
    q_target = np.array([3.5, 0.3])  # joint1 target is outside range (+/-3)
    damping = 1e-6

    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    mw_task, mk_task = _posture_pushing_past_limit(arm_model, q, q_target)

    solver = mw.ConstrainedSolver(
        cfg,
        limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)],
        damping=damping,
    )
    solver._assemble([mw_task], DT, damping)
    H = solver.H.numpy()[0]
    b = solver.rhs.numpy()[0]
    lo = solver.lo.numpy()[0]
    hi = solver.hi.numpy()[0]
    rho = float(solver.rho.numpy()[0])

    # The limit must actually be binding: unconstrained step exceeds the box.
    dq_unc = np.linalg.solve(H, b)
    assert dq_unc[0] > hi[0] + 1e-4, "test did not create an active limit"

    dq = box_qp_admm(H, -b, lo, hi, rho=rho, iters=4000, alpha=1.6)
    # Feasible (exact) and clipped at the joint1 upper bound.
    assert np.all(dq <= hi + 1e-6) and np.all(dq >= lo - 1e-6)
    assert dq[0] == pytest.approx(hi[0], abs=1e-4)

    # Same QP in mink, solved by daqp.
    mk_cfg = mink.Configuration(arm_model, q=q)
    v_mk = mink.solve_ik(
        mk_cfg,
        [mk_task],
        DT,
        solver="daqp",
        damping=damping,
        limits=[mink.ConfigurationLimit(arm_model, gain=GAIN)],
    )
    np.testing.assert_allclose(dq, v_mk * DT, atol=2e-4)


def test_assembly_unconstrained_matches_mink(arm_model):
    # Limits far away -> constrained assembly reduces to the plain solve.
    q = np.array([0.4, -0.7])
    q_target = np.array([0.5, -0.6])
    damping = 1e-6
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    mw_task, mk_task = _posture_pushing_past_limit(arm_model, q, q_target)

    solver = mw.ConstrainedSolver(
        cfg, limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)], damping=damping
    )
    solver._assemble([mw_task], DT, damping)
    H = solver.H.numpy()[0]
    b = solver.rhs.numpy()[0]
    lo, hi = solver.lo.numpy()[0], solver.hi.numpy()[0]
    rho = float(solver.rho.numpy()[0])
    dq = box_qp_admm(H, -b, lo, hi, rho=rho, iters=2000, alpha=1.6)

    mk_cfg = mink.Configuration(arm_model, q=q)
    v_mk = mink.solve_ik(
        mk_cfg, [mk_task], DT, solver="daqp", damping=damping, limits=[]
    )
    np.testing.assert_allclose(dq, v_mk * DT, atol=2e-4)


# --------------------------------------------------------------------------- #
# The real box-ADMM tile kernel (CPU or GPU)
# --------------------------------------------------------------------------- #
def _limited_dofs(model):
    import mujoco

    dofs, los, his = [], [], []
    for j in range(model.njnt):
        if model.jnt_type[j] in (
            mujoco.mjtJoint.mjJNT_HINGE,
            mujoco.mjtJoint.mjJNT_SLIDE,
        ) and model.jnt_limited[j]:
            dofs.append(int(model.jnt_dofadr[j]))
            los.append(float(model.jnt_range[j][0]))
            his.append(float(model.jnt_range[j][1]))
    return np.array(dofs), np.array(los), np.array(his)


@pytest.mark.parametrize("admm_iters", [1, 5, 40])
def test_feasible_when_pushed_into_limit(arm_model, admm_iters):
    q = np.array([2.9, 0.0])
    q_target = np.array([3.5, 0.3])
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task, _ = _posture_pushing_past_limit(arm_model, q, q_target)

    solver = mw.ConstrainedSolver(
        cfg,
        limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)],
        admm_iters=admm_iters,
    )
    v = solver.solve([task], DT).numpy()[0]
    dq = v * DT
    lo, hi = solver.lo.numpy()[0], solver.hi.numpy()[0]

    # Box feasibility holds at ANY iteration count (even K=1).
    assert np.all(dq <= hi + 1e-5)
    assert np.all(dq >= lo - 1e-5)

    # The real safety property: q_next stays inside every joint range.
    q_next = q + dq  # hinge dofs: 1 qpos <-> 1 dof
    dofs, los, his = _limited_dofs(arm_model)
    assert np.all(q_next[dofs] <= his + 1e-5)
    assert np.all(q_next[dofs] >= los - 1e-5)


def test_constrained_prevents_unconstrained_violation(arm_model):
    q = np.array([2.9, 0.0])
    q_target = np.array([3.5, 0.3])
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task, _ = _posture_pushing_past_limit(arm_model, q, q_target)

    hi_lim = float(arm_model.jnt_range[0][1])  # joint1 upper limit (3.14)

    # Unconstrained DLS would drive joint1 past its +limit.
    dls = mw.DLSSolver(cfg, damping=1e-6)
    v_unc = dls.solve([task], DT).numpy()[0]
    assert (q + v_unc * DT)[0] > hi_lim + 1e-3

    cs = mw.ConstrainedSolver(
        cfg, limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)], admm_iters=40
    )
    v_con = cs.solve([task], DT).numpy()[0]
    # Constrained step keeps joint1 inside the limit (strictly, since gain < 1).
    assert (q + v_con * DT)[0] <= hi_lim + 1e-5
    assert (q + v_con * DT)[0] < (q + v_unc * DT)[0]


def test_matches_mink_daqp_active_limit(arm_model):
    q = np.array([2.9, 0.0])
    q_target = np.array([3.5, 0.3])
    damping = 1e-6
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    mw_task, mk_task = _posture_pushing_past_limit(arm_model, q, q_target)

    cs = mw.ConstrainedSolver(
        cfg,
        limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)],
        admm_iters=200,
        damping=damping,
    )
    v_wp = cs.solve([mw_task], DT).numpy()[0]

    mk_cfg = mink.Configuration(arm_model, q=q)
    v_mk = mink.solve_ik(
        mk_cfg,
        [mk_task],
        DT,
        solver="daqp",
        damping=damping,
        limits=[mink.ConfigurationLimit(arm_model, gain=GAIN)],
    )
    np.testing.assert_allclose(v_wp, v_mk, atol=1e-4)


def test_velocity_limit_respected(arm_model):
    q = np.array([0.4, -0.7])
    q_target = np.array([2.0, 1.5])  # large move -> velocity box binds
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task, _ = _posture_pushing_past_limit(arm_model, q, q_target)

    vmax = 1.0
    cs = mw.ConstrainedSolver(
        cfg,
        limits=[mw.VelocityLimit(arm_model, {"joint1": vmax, "joint2": vmax})],
        admm_iters=60,
    )
    v = cs.solve([task], DT).numpy()[0]
    dq = v * DT
    assert np.all(np.abs(dq) <= DT * vmax + 1e-6)


def test_batched_independent_feasibility(arm_model):
    q0 = np.array([2.95, 0.0])
    q1 = np.array([0.0, -2.95])
    cfg = mw.Configuration(arm_model, q=np.stack([q0, q1]), nworld=2)
    t0 = mw.PostureTask(arm_model, cost=1.0)
    t0.set_target(np.array([3.5, 0.5]))
    v = mw.ConstrainedSolver(
        cfg, limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)], admm_iters=40
    ).solve([t0], DT).numpy()

    dofs, los, his = _limited_dofs(arm_model)
    for w, q in enumerate([q0, q1]):
        q_next = q + v[w] * DT
        assert np.all(q_next[dofs] <= his + 1e-5)
        assert np.all(q_next[dofs] >= los - 1e-5)


def test_gpu_kernel_matches_numpy_reference(arm_model):
    q = np.array([2.9, 0.0])
    q_target = np.array([3.5, 0.3])
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    task, _ = _posture_pushing_past_limit(arm_model, q, q_target)
    cs = mw.ConstrainedSolver(
        cfg, limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)], admm_iters=200
    )
    v = cs.solve([task], DT).numpy()[0]
    dq_gpu = v * DT

    H = cs.H.numpy()[0]
    b = cs.rhs.numpy()[0]
    lo, hi = cs.lo.numpy()[0], cs.hi.numpy()[0]
    rho = float(cs.rho.numpy()[0])
    dq_ref = box_qp_admm(H, -b, lo, hi, rho=rho, iters=200, alpha=cs.alpha)
    np.testing.assert_allclose(dq_gpu, dq_ref, atol=1e-5)
