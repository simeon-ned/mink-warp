"""Kernels for assembling and solving the differential IK normal equations."""

from __future__ import annotations

from typing import Any

import warp as wp

# Cache of nv-specialized tile Cholesky kernels (Newton-style).
_CHOLESKY_CACHE: dict[int, Any] = {}

# Threads per block for tile Cholesky (must be a multiple of 32).
TILE_THREADS = 32


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


def get_cholesky_solve_kernel(nv: int):
    """Return a batched SPD solve kernel specialized for size ``nv``.

    Uses Warp tile Cholesky (cuSolverDx when available), Newton-style:
    one **block** per world via :func:`wp.launch_tiled`.
    """
    if nv in _CHOLESKY_CACHE:
        return _CHOLESKY_CACHE[nv]

    NV = int(nv)

    def _cholesky_solve_tiled(
        H: wp.array3d[float],
        rhs: wp.array2d[float],
        dq: wp.array2d[float],
    ):
        # With launch_tiled, tid() is the block/world index; threads in the
        # block cooperate on the tile.
        worldid = wp.tid()
        A = wp.tile_load(H[worldid], shape=(NV, NV))
        b = wp.tile_load(rhs[worldid], shape=(NV,))
        L = wp.tile_cholesky(A)
        x = wp.tile_cholesky_solve(L, b)
        wp.tile_store(dq[worldid], x)

    _cholesky_solve_tiled.__name__ = f"cholesky_solve_tiled_{NV}"
    _cholesky_solve_tiled.__qualname__ = f"cholesky_solve_tiled_{NV}"
    kernel = wp.kernel(enable_backward=False, module="unique")(_cholesky_solve_tiled)
    _CHOLESKY_CACHE[nv] = kernel
    return kernel


def launch_cholesky_solve(
    kernel,
    *,
    nworld: int,
    H: wp.array,
    rhs: wp.array,
    dq: wp.array,
    device: str | None = None,
) -> None:
    """Launch a tiled Cholesky solve: one block per world."""
    wp.launch_tiled(
        kernel,
        dim=[nworld],
        inputs=[H, rhs],
        outputs=[dq],
        block_dim=TILE_THREADS,
        device=device,
    )
