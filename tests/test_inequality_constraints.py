"""General dense inequality path ``G dq <= h`` for the constrained solver.

Layered like ``test_constrained_solver.py``:

* **numpy oracle** — the reduced OSQP-ADMM reference (``ineq_qp_admm``) is locked
  against ``daqp`` on random inequality QPs.
* **kernel** — the real ``admm_ineq`` tile kernel (CPU or GPU) matches the numpy
  oracle, and the whole solver path reproduces mink's QP with the *same* dense
  rows: the configuration limit re-expressed as ``G=[P;-P]`` inequalities, and an
  arbitrary half-space via :class:`LinearInequalityLimit`.
* **properties** — feasibility when a row is active, no-op when inactive, padded
  inert rows, batched independence, and (crucially) that constraints which never
  bind leave the tracked trajectory identical to the unconstrained solver.
"""

from __future__ import annotations

import numpy as np
import pytest

mink = pytest.importorskip("mink")
daqp = pytest.importorskip("daqp")

from helpers_admm import ineq_qp_admm  # noqa: E402

import mink_warp as mw  # noqa: E402

DT = 0.02
GAIN = 0.95
SIGMA = 1e-6


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _daqp_ineq(H, c, G, h):
    """Solve ``min 1/2 x'Hx + c'x s.t. Gx<=h`` with daqp. Returns (x, exitflag)."""
    m = G.shape[0]
    A = np.ascontiguousarray(G, dtype=np.float64)
    bupper = np.ascontiguousarray(h, dtype=np.float64)
    blower = np.full(m, -1e30)
    sense = np.zeros(m, dtype=np.int32)
    x, _fval, exitflag, _info = daqp.solve(
        np.ascontiguousarray(H, dtype=np.float64),
        np.ascontiguousarray(c, dtype=np.float64),
        A,
        bupper,
        blower,
        sense,
    )
    return x, exitflag


def _rand_ineq_qp(rng, n, m, margin=(0.05, 0.4)):
    """Random SPD ``H`` and rows placed so several are active at the optimum."""
    Wt = rng.standard_normal((n, n))
    H = (Wt @ Wt.T + 0.3 * np.eye(n)).astype(np.float32)
    c = rng.standard_normal(n).astype(np.float32)
    G = rng.standard_normal((m, n)).astype(np.float32)
    x_unc = np.linalg.solve(H, -c)
    h = (G @ x_unc - rng.uniform(*margin, m)).astype(np.float32)
    return H, c, G, h


def _rho(H):
    d = np.diag(H)
    return float(np.sqrt(d.min() * d.max()))


# --------------------------------------------------------------------------- #
# numpy oracle vs daqp (locks the math)
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("n,m", [(3, 4), (5, 6), (8, 10)])
def test_numpy_ineq_admm_matches_daqp(n, m):
    rng = np.random.default_rng(100 + n)
    H, c, G, h = _rand_ineq_qp(rng, n, m)
    x_ref, ef = _daqp_ineq(H, c, G, h)
    assert ef == 1, "daqp did not solve the reference QP"
    x = ineq_qp_admm(H, c, G, h, rho=_rho(H), sigma=SIGMA, iters=4000, alpha=1.6)
    np.testing.assert_allclose(x, x_ref, atol=1e-4)
    assert np.max(G @ x - h) <= 1e-5  # feasible


# --------------------------------------------------------------------------- #
# device kernel vs numpy oracle
# --------------------------------------------------------------------------- #
def test_kernel_matches_numpy_reference():
    import warp as wp

    from mink_warp.kernels.constrained import (
        get_admm_ineq_kernel,
        launch_admm_ineq_solve,
    )

    rng = np.random.default_rng(7)
    n, m, iters = 6, 8, 200
    H, c, G, h = _rand_ineq_qp(rng, n, m)
    rho = _rho(H)
    ref = ineq_qp_admm(H, c, G, h, rho=rho, sigma=SIGMA, iters=iters, alpha=1.6)

    k = get_admm_ineq_kernel(n, m, iters)
    dq = wp.zeros((1, n), dtype=float)
    launch_admm_ineq_solve(
        k,
        nworld=1,
        H=wp.array(H[None], dtype=float),
        b=wp.array((-c)[None], dtype=float),
        G=wp.array(G[None], dtype=float),
        h=wp.array(h[None], dtype=float),
        rho=wp.array([rho], dtype=float),
        sigma=SIGMA,
        alpha=1.6,
        dq=dq,
    )
    np.testing.assert_allclose(dq.numpy()[0], ref, atol=1e-5)


# --------------------------------------------------------------------------- #
# configuration limit re-expressed as inequalities: general path == box == mink
# --------------------------------------------------------------------------- #
def _posture(model, target, cost=1.0):
    mw_t = mw.PostureTask(model, cost=cost)
    mw_t.set_target(target)
    mk_t = mink.PostureTask(model, cost=cost)
    mk_t.set_target(target)
    return mw_t, mk_t


def test_config_limit_general_path_matches_box_and_mink(arm_model):
    q = np.array([2.9, 0.0])
    target = np.array([3.5, 0.3])  # joint1 target beyond +limit -> row active
    damping = 1e-6

    cfg_box = mw.Configuration(arm_model, q=q, nworld=1)
    mw_t, mk_t = _posture(arm_model, target)
    box = mw.ConstrainedSolver(
        cfg_box, limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)],
        admm_iters=60, damping=damping,
    )
    assert box._use_ineq is False
    v_box = box.solve([mw_t], DT).numpy()[0]

    cfg_gen = mw.Configuration(arm_model, q=q, nworld=1)
    mw_t2, _ = _posture(arm_model, target)
    gen = mw.ConstrainedSolver(
        cfg_gen, limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)],
        admm_iters=80, damping=damping, use_inequalities=True,
    )
    assert gen._use_ineq is True and gen.n_ineq == 4
    v_gen = gen.solve([mw_t2], DT).numpy()[0]

    mk_cfg = mink.Configuration(arm_model, q=q)
    v_mk = mink.solve_ik(
        mk_cfg, [mk_t], DT, solver="daqp", damping=damping,
        limits=[mink.ConfigurationLimit(arm_model, gain=GAIN)],
    )
    np.testing.assert_allclose(v_gen, v_mk, atol=1e-4)
    np.testing.assert_allclose(v_gen, v_box, atol=1e-4)


@pytest.mark.parametrize("admm_iters", [40, 120])
def test_config_limit_general_path_feasible(arm_model, admm_iters):
    import mujoco

    q = np.array([2.9, 0.0])
    target = np.array([3.5, 0.3])
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    mw_t, _ = _posture(arm_model, target)
    gen = mw.ConstrainedSolver(
        cfg, limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)],
        admm_iters=admm_iters, use_inequalities=True,
    )
    dq = gen.solve([mw_t], DT).numpy()[0] * DT
    q_next = q + dq
    for j in range(arm_model.njnt):
        if arm_model.jnt_limited[j] and arm_model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE:
            d = int(arm_model.jnt_dofadr[j])
            lo, hi = arm_model.jnt_range[j]
            assert lo - 1e-3 <= q_next[d] <= hi + 1e-3


# --------------------------------------------------------------------------- #
# arbitrary half-space via LinearInequalityLimit
# --------------------------------------------------------------------------- #
class _MinkOneRow(mink.Limit):
    """Minimal mink limit returning a constant half-space, for the oracle."""

    def __init__(self, G, h):
        self._G = np.asarray(G, dtype=np.float64)
        self._h = np.asarray(h, dtype=np.float64)

    def compute_qp_inequalities(self, configuration, dt):
        from mink.limits.limit import Constraint

        return Constraint(G=self._G.copy(), h=self._h.copy())


def _unconstrained_dq(arm_model, q, target, damping):
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(target)
    v = mw.DLSSolver(cfg, damping=damping).solve([t], DT).numpy()[0]
    return v * DT


def test_linear_inequality_auto_selects_general_path(arm_model):
    cfg = mw.Configuration(arm_model, q=np.array([0.3, -0.2]), nworld=1)
    lim = mw.LinearInequalityLimit([[1.0, 0.0]], [10.0])
    # box_capable=False -> general path even without use_inequalities.
    cs = mw.ConstrainedSolver(cfg, limits=[lim])
    assert cs._use_ineq is True and cs.n_ineq == 1


def test_half_space_matches_mink_active(arm_model):
    q = np.array([0.3, -0.2])
    target = np.array([1.2, -0.2])
    damping = 1e-6
    dq_unc = _unconstrained_dq(arm_model, q, target, damping)

    n = np.array([1.0, 0.0], dtype=np.float64)
    h0 = float(n @ dq_unc) - 0.01  # tighten below the unconstrained step -> active
    lim = mw.LinearInequalityLimit(n[None, :], [h0])

    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(target)
    cs = mw.ConstrainedSolver(cfg, limits=[lim], admm_iters=400, damping=damping)
    v_wp = cs.solve([t], DT).numpy()[0]
    dq_wp = v_wp * DT

    mk_cfg = mink.Configuration(arm_model, q=q)
    mk_t = mink.PostureTask(arm_model, cost=1.0)
    mk_t.set_target(target)
    v_mk = mink.solve_ik(
        mk_cfg, [mk_t], DT, solver="daqp", damping=damping,
        limits=[_MinkOneRow(n[None, :], [h0])],
    )
    np.testing.assert_allclose(v_wp, v_mk, atol=1e-4)
    assert n @ dq_wp <= h0 + 1e-5  # feasible + (nearly) active


def test_half_space_inactive_is_unconstrained(arm_model):
    q = np.array([0.3, -0.2])
    target = np.array([0.5, -0.1])
    damping = 1e-6
    dq_unc = _unconstrained_dq(arm_model, q, target, damping)

    n = np.array([1.0, 0.0], dtype=np.float64)
    h0 = float(n @ dq_unc) + 1.0  # far above -> inactive
    lim = mw.LinearInequalityLimit(n[None, :], [h0])

    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(target)
    cs = mw.ConstrainedSolver(cfg, limits=[lim], admm_iters=200, damping=damping)
    dq = cs.solve([t], DT).numpy()[0] * DT
    np.testing.assert_allclose(dq, dq_unc, atol=1e-4)


def test_two_active_rows_match_mink(arm_model):
    q = np.array([0.3, -0.2])
    target = np.array([1.2, 1.0])
    damping = 1e-6
    dq_unc = _unconstrained_dq(arm_model, q, target, damping)

    G = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    h = np.array([float(dq_unc[0]) - 0.01, float(dq_unc[1]) - 0.01])
    lim = mw.LinearInequalityLimit(G, h)

    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(target)
    cs = mw.ConstrainedSolver(cfg, limits=[lim], admm_iters=500, damping=damping)
    v_wp = cs.solve([t], DT).numpy()[0]
    dq_wp = v_wp * DT

    mk_cfg = mink.Configuration(arm_model, q=q)
    mk_t = mink.PostureTask(arm_model, cost=1.0)
    mk_t.set_target(target)
    v_mk = mink.solve_ik(
        mk_cfg, [mk_t], DT, solver="daqp", damping=damping,
        limits=[_MinkOneRow(G, h)],
    )
    np.testing.assert_allclose(v_wp, v_mk, atol=1e-4)
    assert np.all(G @ dq_wp <= h + 1e-5)


def test_padded_inert_rows_do_not_change_result(arm_model):
    q = np.array([0.3, -0.2])
    target = np.array([1.2, -0.2])
    damping = 1e-6
    dq_unc = _unconstrained_dq(arm_model, q, target, damping)
    n = np.array([1.0, 0.0], dtype=np.float64)
    h0 = float(n @ dq_unc) - 0.01

    def solve(limits):
        cfg = mw.Configuration(arm_model, q=q, nworld=1)
        t = mw.PostureTask(arm_model, cost=1.0)
        t.set_target(target)
        cs = mw.ConstrainedSolver(cfg, limits=limits, admm_iters=400, damping=damping)
        return cs.solve([t], DT).numpy()[0]

    v_one = solve([mw.LinearInequalityLimit(n[None, :], [h0])])
    # A second row that is a zero half-space with huge bound (inert).
    G2 = np.array([[1.0, 0.0], [0.0, 0.0]])
    v_padded = solve([mw.LinearInequalityLimit(G2, [h0, 1e9])])
    np.testing.assert_allclose(v_one, v_padded, atol=1e-4)


def test_batched_mixed_active_inactive(arm_model):
    # World 0's step violates the row (active); world 1 barely moves (inactive).
    # The batch must reproduce two independent nworld=1 solves, not smear them.
    q0 = np.array([0.3, -0.2])
    q1 = np.array([0.3, -0.2])
    tgt0 = np.array([1.2, -0.2])  # large joint1 move -> row active in world 0
    tgt1 = np.array([0.32, -0.2])  # tiny move -> row inactive in world 1
    damping = 1e-6
    dq0_unc = _unconstrained_dq(arm_model, q0, tgt0, damping)
    n = np.array([1.0, 0.0], dtype=np.float64)
    h0 = float(n @ dq0_unc) - 0.01  # active for world 0, slack for world 1
    lim = mw.LinearInequalityLimit(n[None, :], [h0])

    def solve_one(q, tgt):
        cfg = mw.Configuration(arm_model, q=q, nworld=1)
        t = mw.PostureTask(arm_model, cost=1.0)
        t.set_target(tgt)
        return mw.ConstrainedSolver(
            cfg, limits=[mw.LinearInequalityLimit(n[None, :], [h0])],
            admm_iters=400, damping=damping,
        ).solve([t], DT).numpy()[0]

    v0_single = solve_one(q0, tgt0)
    v1_single = solve_one(q1, tgt1)

    cfg = mw.Configuration(arm_model, q=np.stack([q0, q1]), nworld=2)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(np.stack([tgt0, tgt1]))
    v = mw.ConstrainedSolver(
        cfg, limits=[lim], admm_iters=400, damping=damping
    ).solve([t], DT).numpy()

    np.testing.assert_allclose(v[0], v0_single, atol=1e-5)
    np.testing.assert_allclose(v[1], v1_single, atol=1e-5)
    assert n @ (v[0] * DT) <= h0 + 1e-5  # world 0 feasible + active
    assert n @ (v[1] * DT) < h0 - 1e-3  # world 1 slack (inactive)


def test_mixed_box_and_linear_inequality(arm_model):
    # A box limit (config) + an inequality-only limit in one solver: the presence
    # of the inequality-only limit forces the general path, and BOTH must bind.
    q = np.array([2.9, 0.0])
    target = np.array([3.5, 1.0])  # joint1 past +limit; joint2 pushed up
    damping = 1e-6
    dq_unc = _unconstrained_dq(arm_model, q, target, damping)

    n = np.array([0.0, 1.0], dtype=np.float64)  # cap joint2's step
    h0 = float(n @ dq_unc) - 0.02
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(target)
    cs = mw.ConstrainedSolver(
        cfg,
        limits=[
            mw.ConfigurationLimit(arm_model, gain=GAIN),
            mw.LinearInequalityLimit(n[None, :], [h0]),
        ],
        admm_iters=400,
        damping=damping,
    )
    assert cs._use_ineq is True and cs.n_ineq == 5  # 4 config rows + 1 half-space
    dq = cs.solve([t], DT).numpy()[0] * DT
    q_next = q + dq
    # Config limit on joint1 holds, and the half-space on joint2 holds.
    assert q_next[0] <= float(arm_model.jnt_range[0][1]) + 1e-4
    assert n @ dq <= h0 + 1e-4


def test_config_limit_general_inactive_equals_unconstrained(arm_model):
    # Config limit through the GENERAL path, target well inside range -> the rows
    # never bind, so the result must equal the plain unconstrained solve.
    q = np.array([0.3, -0.2])
    target = np.array([0.6, -0.4])
    damping = 1e-6
    dq_unc = _unconstrained_dq(arm_model, q, target, damping)

    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(target)
    cs = mw.ConstrainedSolver(
        cfg, limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)],
        admm_iters=200, damping=damping, use_inequalities=True,
    )
    dq = cs.solve([t], DT).numpy()[0] * DT
    np.testing.assert_allclose(dq, dq_unc, atol=1e-4)


def test_velocity_limit_general_path_respected(arm_model):
    # Large move so the velocity box saturates; the general path must stay
    # feasible AND actually saturate (match the box path), not under-move.
    q = np.array([0.4, -0.7])
    target = np.array([2.0, 1.5])
    vmax = 1.0
    damping = 1e-6

    def solve(use_ineq):
        cfg = mw.Configuration(arm_model, q=q, nworld=1)
        t = mw.PostureTask(arm_model, cost=1.0)
        t.set_target(target)
        cs = mw.ConstrainedSolver(
            cfg,
            limits=[mw.VelocityLimit(arm_model, {"joint1": vmax, "joint2": vmax})],
            admm_iters=300,
            damping=damping,
            use_inequalities=use_ineq,
        )
        return cs, cs.solve([t], DT).numpy()[0] * DT

    cs, dq = solve(True)
    _, dq_box = solve(False)
    assert cs._use_ineq is True and cs.n_ineq == 4
    assert np.all(np.abs(dq) <= DT * vmax + 1e-4)  # feasible
    # Both dofs want to move far -> both saturate at +-dt*vmax; the general path
    # matches the exact box path (rules out an over-conservative / zero solution).
    np.testing.assert_allclose(np.abs(dq), DT * vmax, atol=1e-3)
    np.testing.assert_allclose(dq, dq_box, atol=1e-3)


# --------------------------------------------------------------------------- #
# CUDA-graph capture of the general-inequality path (GPU only)
# --------------------------------------------------------------------------- #
def test_ineq_path_cuda_graph_matches_eager():
    import warp as wp

    if not wp.get_cuda_devices():
        pytest.skip("no CUDA device; graph capture is CUDA-only")
    device = "cuda:0"
    import mujoco

    model = mujoco.MjModel.from_xml_string(
        """
        <mujoco><compiler angle="radian"/><worldbody>
        <body pos="0 0 0.1"><joint name="j1" type="hinge" axis="0 0 1"
          range="-1.0 1.0" limited="true"/><geom type="capsule"
          fromto="0 0 0 0.3 0 0" size="0.03"/>
        <body pos="0.3 0 0"><joint name="j2" type="hinge" axis="0 0 1"
          range="-1.0 1.0" limited="true"/><geom type="capsule"
          fromto="0 0 0 0.25 0 0" size="0.025"/></body></body>
        </worldbody></mujoco>
        """
    )
    q0 = np.array([0.9, 0.0])
    target = np.array([2.0, 0.3])  # drives j1 past its +1.0 limit -> rows active

    def rollout(use_graph):
        cfg = mw.Configuration(model, q=q0, nworld=1, device=device)
        t = mw.PostureTask(model, cost=1.0)
        t.set_target(target)
        s = mw.ConstrainedSolver(
            cfg, limits=[mw.ConfigurationLimit(model, gain=GAIN)],
            admm_iters=60, use_inequalities=True,
        )
        assert s._use_ineq is True
        for _ in range(10):
            s.solve_and_integrate([t], DT, iterations=1, use_graph=use_graph)
        return cfg.q.numpy()[0].copy()

    q_graph = rollout(True)
    q_eager = rollout(False)
    np.testing.assert_allclose(q_graph, q_eager, atol=1e-5)
    assert np.all(q_graph <= 1.0 + 1e-3) and np.all(q_graph >= -1.0 - 1e-3)


# --------------------------------------------------------------------------- #
# tracking parity: constraints that never bind don't distort the trajectory
# --------------------------------------------------------------------------- #
def _rollout(solver, model, q0, target, steps):
    t = mw.PostureTask(model, cost=1.0)
    t.set_target(target)
    for _ in range(steps):
        solver.solve_and_integrate([t], DT, iterations=1)
    return solver.configuration.q.numpy()[0].copy()


def test_tracking_matches_unconstrained_when_limits_inactive(arm_model):
    # Target well inside every joint range -> no limit ever binds.
    q0 = np.array([0.4, -0.7])
    target = np.array([1.0, -1.0])
    damping = 1e-6
    steps = 60

    dls = mw.DLSSolver(mw.Configuration(arm_model, q=q0, nworld=1), damping=damping)
    q_dls = _rollout(dls, arm_model, q0, target, steps)

    box = mw.ConstrainedSolver(
        mw.Configuration(arm_model, q=q0, nworld=1),
        limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)],
        admm_iters=30, damping=damping,
    )
    q_box = _rollout(box, arm_model, q0, target, steps)

    gen = mw.ConstrainedSolver(
        mw.Configuration(arm_model, q=q0, nworld=1),
        limits=[mw.ConfigurationLimit(arm_model, gain=GAIN)],
        admm_iters=120, damping=damping, use_inequalities=True,
    )
    q_gen = _rollout(gen, arm_model, q0, target, steps)

    # None of the limits bind, so both constrained paths track the same target as
    # the unconstrained solver (tracking error is compatible).
    err_dls = np.linalg.norm(q_dls - target)
    err_box = np.linalg.norm(q_box - target)
    err_gen = np.linalg.norm(q_gen - target)
    assert err_dls < 1e-3
    np.testing.assert_allclose(q_box, q_dls, atol=1e-4)
    np.testing.assert_allclose(q_gen, q_dls, atol=1e-3)
    assert abs(err_box - err_dls) < 1e-4
    assert abs(err_gen - err_dls) < 1e-3


# --------------------------------------------------------------------------- #
# guards / capability handling
# --------------------------------------------------------------------------- #
def test_sigma_guard(arm_model):
    cfg = mw.Configuration(arm_model, nworld=1)
    with pytest.raises(ValueError):
        mw.ConstrainedSolver(cfg, sigma=0.0)


def test_solve_ik_rejects_limits_for_unconstrained_solver(arm_model):
    cfg = mw.Configuration(arm_model, q=np.array([0.4, -0.7]), nworld=1)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(np.array([0.5, -0.6]))
    dls = mw.DLSSolver(cfg)
    assert dls.supports_limits is False
    with pytest.raises(ValueError, match="does not support limits"):
        mw.solve_ik(cfg, [t], DT, solver=dls, limits=None)


def test_solve_ik_rejects_limits_with_explicit_constrained_solver(arm_model):
    # Even a ConstrainedSolver: limits= would be silently dropped (the solver
    # uses its own construction-time limits), so passing both must fail loud.
    cfg = mw.Configuration(arm_model, q=np.array([0.4, -0.7]), nworld=1)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(np.array([0.5, -0.6]))
    cs = mw.ConstrainedSolver(cfg, limits=[mw.ConfigurationLimit(arm_model)])
    assert cs.supports_limits is True
    with pytest.raises(ValueError, match="honoured only when the solver is auto-built"):
        mw.solve_ik(cfg, [t], DT, solver=cs, limits=None)


def test_solve_ik_explicit_constrained_solver_no_limits_arg(arm_model):
    # The supported way: build the ConstrainedSolver with its limits, pass no
    # limits= to solve_ik.
    cfg = mw.Configuration(arm_model, q=np.array([0.4, -0.7]), nworld=1)
    t = mw.PostureTask(arm_model, cost=1.0)
    t.set_target(np.array([0.5, -0.6]))
    cs = mw.ConstrainedSolver(cfg, limits=[mw.ConfigurationLimit(arm_model)])
    v = mw.solve_ik(cfg, [t], DT, solver=cs)
    assert v.shape == (1, cfg.nv)


def test_empty_linear_inequality_rejected():
    with pytest.raises(ValueError, match="at least one row"):
        mw.LinearInequalityLimit(np.zeros((0, 2)), np.zeros(0))
