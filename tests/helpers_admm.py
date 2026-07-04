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
