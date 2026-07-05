"""Host NumPy reference for the box-constrained QP the GPU solver implements.

The device kernel (``kernels/constrained.py``) uses ``wp.tile_cholesky`` which is
cuSolverDx / GPU-only, so it cannot run under CPU CI. This NumPy reference encodes
the *exact same* OSQP-style box-ADMM math and is the CPU oracle: it is validated
against ``daqp`` here, and the GPU kernel is validated against it on the GPU.

    min_x  1/2 x^T H x + c^T x   s.t.   lo <= x <= hi

Scaled ADMM splitting f(x)=1/2 x^T H x + c^T x, g(z)=indicator[lo,hi], x = z:

    M = H + rho I                       (factor once)
    b = -c
    x_{k+1} = M^{-1}(rho (z_k - u_k) + b)
    x_hat   = alpha x_{k+1} + (1-alpha) z_k          (over-relaxation)
    z_{k+1} = clip(x_hat + u_k, lo, hi)              (always feasible)
    u_{k+1} = u_k + x_hat - z_{k+1}

Returns ``z`` (feasible at every iteration; the returned step never violates the
box even if the loop is truncated).
"""

from __future__ import annotations

import numpy as np


def box_qp_admm(
    H: np.ndarray,
    c: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    *,
    rho: float,
    iters: int,
    alpha: float = 1.0,
) -> np.ndarray:
    """Solve a single box-constrained QP by scaled ADMM. Returns feasible ``z``."""
    H = np.asarray(H, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    lo = np.asarray(lo, dtype=np.float64)
    hi = np.asarray(hi, dtype=np.float64)
    n = H.shape[0]

    M = H + rho * np.eye(n)
    L = np.linalg.cholesky(M)
    b = -c

    def _solve(rhs: np.ndarray) -> np.ndarray:
        y = np.linalg.solve(L, rhs)
        return np.linalg.solve(L.T, y)

    x = _solve(b)  # warm start from the regularized unconstrained step
    z = np.clip(x, lo, hi)
    u = np.zeros(n)
    for _ in range(iters):
        x = _solve(rho * (z - u) + b)
        x_hat = alpha * x + (1.0 - alpha) * z
        z = np.clip(x_hat + u, lo, hi)
        u = u + x_hat - z
    return z


def ineq_qp_admm(
    H: np.ndarray,
    c: np.ndarray,
    G: np.ndarray,
    h: np.ndarray,
    *,
    rho: float,
    sigma: float,
    iters: int,
    alpha: float = 1.6,
) -> np.ndarray:
    """Host NumPy reference for the general dense-inequality QP the GPU solver runs.

        min_x  1/2 x^T H x + c^T x   s.t.   G x <= h

    This is the reduced (Schur-normal) OSQP-ADMM the device kernel
    (``get_admm_ineq_kernel``) implements. The raw KKT system is quasidefinite,
    so ``z = G x`` is eliminated to the SPD normal matrix

        M = H + sigma I + rho G^T G          (factored once)

    and each iteration is a cached solve + a projection of the constraint image
    onto ``(-inf, h]`` + a dual update::

        x_tilde = M^{-1}(sigma x - c + G^T (rho z - y))
        z_tilde = G x_tilde
        x       = alpha x_tilde + (1-alpha) x
        z_hat   = alpha z_tilde + (1-alpha) z
        z       = min(z_hat + y/rho, h)               (project onto (-inf, h])
        y       = y + rho (z_hat - z)

    Unlike the box solver (which returns the projected ``z`` and is feasible at
    every iteration), the returned ``x`` reaches feasibility *asymptotically*:
    ``G x <= h`` holds to a tolerance that shrinks with ``iters``. ``sigma > 0``
    keeps ``M`` positive-definite even where ``H`` and ``G^T G`` leave a dof
    unpenalized. Returns ``x``.
    """
    H = np.asarray(H, dtype=np.float64)
    c = np.asarray(c, dtype=np.float64)
    G = np.asarray(G, dtype=np.float64)
    h = np.asarray(h, dtype=np.float64)
    n = H.shape[0]
    m = G.shape[0]

    M = H + sigma * np.eye(n) + rho * (G.T @ G)
    L = np.linalg.cholesky(M)

    def _solve(rhs: np.ndarray) -> np.ndarray:
        return np.linalg.solve(L.T, np.linalg.solve(L, rhs))

    x = _solve(-c)  # warm start from the regularized unconstrained step
    z = np.minimum(G @ x, h)
    y = np.zeros(m)
    for _ in range(iters):
        x_tilde = _solve(sigma * x - c + G.T @ (rho * z - y))
        z_tilde = G @ x_tilde
        x = alpha * x_tilde + (1.0 - alpha) * x
        z_hat = alpha * z_tilde + (1.0 - alpha) * z
        z_new = np.minimum(z_hat + y / rho, h)
        y = y + rho * (z_hat - z_new)
        z = z_new
    return x
