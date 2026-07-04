"""The NumPy box-ADMM reference matches daqp on random SPD box QPs.

Locks the math the GPU kernel implements, independent of Warp (tile Cholesky is
GPU-only). Once this passes, the device kernel is validated against this same
reference on the GPU.
"""

from __future__ import annotations

import numpy as np
import pytest

qpsolvers = pytest.importorskip("qpsolvers")

from helpers_admm import box_qp_admm  # noqa: E402

_HAS_DAQP = "daqp" in qpsolvers.available_solvers


def _daqp_box(H, c, lo, hi):
    import qpsolvers

    n = H.shape[0]
    G = np.vstack([np.eye(n), -np.eye(n)])
    h = np.hstack([hi, -lo])
    return qpsolvers.solve_qp(H, c, G, h, solver="daqp")


def _rand_spd(n, rng, cond=5.0):
    A = rng.standard_normal((n, n))
    return A @ A.T + cond * np.eye(n)


@pytest.mark.skipif(not _HAS_DAQP, reason="daqp not available")
@pytest.mark.parametrize("seed", range(6))
def test_admm_matches_daqp_active_box(seed):
    rng = np.random.default_rng(seed)
    n = rng.integers(3, 12)
    H = _rand_spd(n, rng)
    c = rng.standard_normal(n)

    # Unconstrained optimum, then a box tight enough to make several
    # constraints active (clamp roughly half the components).
    x_unc = np.linalg.solve(H, -c)
    lo = np.minimum(x_unc, 0.0) - 0.05
    hi = np.maximum(x_unc, 0.0) + 0.05
    # Tighten a random subset so those bounds bind.
    mask = rng.random(n) < 0.5
    hi[mask] = np.minimum(hi[mask], x_unc[mask] - 0.1)
    lo = np.minimum(lo, hi - 1e-3)

    x_ref = _daqp_box(H, c, lo, hi)
    assert x_ref is not None

    rho = float(np.mean(np.diag(H)))
    x_admm = box_qp_admm(H, c, lo, hi, rho=rho, iters=800, alpha=1.6)

    # Feasibility is exact (returned z is a clip).
    assert np.all(x_admm <= hi + 1e-9)
    assert np.all(x_admm >= lo - 1e-9)
    # Agreement with the active-set QP optimum.
    np.testing.assert_allclose(x_admm, x_ref, atol=1e-4)
    assert mask.any()  # the test actually exercised active bounds


@pytest.mark.skipif(not _HAS_DAQP, reason="daqp not available")
def test_admm_unconstrained_recovers_normal_equations():
    rng = np.random.default_rng(42)
    n = 8
    H = _rand_spd(n, rng)
    c = rng.standard_normal(n)
    big = 1e6 * np.ones(n)
    x = box_qp_admm(H, c, -big, big, rho=float(np.mean(np.diag(H))), iters=200)
    np.testing.assert_allclose(x, np.linalg.solve(H, -c), atol=1e-5)


def test_admm_feasible_at_any_iteration_count():
    rng = np.random.default_rng(7)
    n = 6
    H = _rand_spd(n, rng)
    c = rng.standard_normal(n)
    lo = -0.01 * np.ones(n)
    hi = 0.01 * np.ones(n)
    for k in (0, 1, 2, 5, 20):
        x = box_qp_admm(H, c, lo, hi, rho=1.0, iters=k)
        assert np.all(x <= hi + 1e-12) and np.all(x >= lo - 1e-12)
