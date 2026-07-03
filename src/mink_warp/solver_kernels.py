"""Kernels for assembling and solving the differential IK normal equations."""

from __future__ import annotations

import warp as wp


@wp.kernel
def zero_normal_equations(
    H: wp.array3d[float],
    c: wp.array2d[float],
    mu_total: wp.array[float],
    nv: int,
):
    worldid = wp.tid()
    mu_total[worldid] = 0.0
    for i in range(nv):
        c[worldid, i] = 0.0
        for j in range(nv):
            H[worldid, i, j] = 0.0


@wp.kernel
def accumulate_normal_equations(
    W: wp.array3d[float],
    e: wp.array2d[float],
    mu: wp.array[float],
    k: int,
    nv: int,
    H: wp.array3d[float],
    c: wp.array2d[float],
    mu_total: wp.array[float],
):
    """Accumulate H += WᵀW, c += -eᵀW, mu_total += mu (Mink residual form)."""
    worldid = wp.tid()
    mu_total[worldid] += mu[worldid]
    for i in range(nv):
        s_c = float(0.0)
        for r in range(k):
            s_c += e[worldid, r] * W[worldid, r, i]
        c[worldid, i] -= s_c
        for j in range(nv):
            s_h = float(0.0)
            for r in range(k):
                s_h += W[worldid, r, i] * W[worldid, r, j]
            H[worldid, i, j] += s_h


@wp.kernel
def add_damping_diag(
    H: wp.array3d[float],
    mu_total: wp.array[float],
    damping: float,
    nv: int,
):
    worldid, i = wp.tid()
    H[worldid, i, i] += damping + mu_total[worldid]


@wp.kernel
def scale_velocity(
    dq: wp.array2d[float],
    dt: float,
    nv: int,
    v_out: wp.array2d[float],
):
    worldid = wp.tid()
    inv_dt = 1.0 / dt
    for i in range(nv):
        v_out[worldid, i] = dq[worldid, i] * inv_dt


@wp.kernel
def neg_vec(
    c: wp.array2d[float],
    nv: int,
    b: wp.array2d[float],
):
    """b = -c."""
    worldid = wp.tid()
    for i in range(nv):
        b[worldid, i] = -c[worldid, i]


@wp.kernel
def cholesky_solve_batched(
    A: wp.array3d[float],
    b: wp.array2d[float],
    n: int,
    x: wp.array2d[float],
):
    """Batched Cholesky solve ``A x = b`` for SPD ``A`` (n×n), one world per thread.

    Overwrites ``A`` with its lower-triangular Cholesky factor.
    """
    worldid = wp.tid()

    # A = L Lᵀ, L stored in lower triangle of A.
    for i in range(n):
        s = A[worldid, i, i]
        for k in range(i):
            lik = A[worldid, i, k]
            s -= lik * lik
        if s < 1.0e-12:
            s = 1.0e-12
        diag = wp.sqrt(s)
        A[worldid, i, i] = diag
        for j in range(i + 1, n):
            s2 = A[worldid, j, i]
            for k in range(i):
                s2 -= A[worldid, j, k] * A[worldid, i, k]
            A[worldid, j, i] = s2 / diag

    # Forward substitution: L y = b (y stored in x).
    for i in range(n):
        s = b[worldid, i]
        for k in range(i):
            s -= A[worldid, i, k] * x[worldid, k]
        x[worldid, i] = s / A[worldid, i, i]

    # Back substitution: Lᵀ x = y.
    for ii in range(n):
        i = n - 1 - ii
        s = x[worldid, i]
        for k in range(i + 1, n):
            s -= A[worldid, k, i] * x[worldid, k]
        x[worldid, i] = s / A[worldid, i, i]
