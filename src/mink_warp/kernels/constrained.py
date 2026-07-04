"""Kernels for the constrained IK solver: box assembly + box-ADMM QP solve.

Two families live here:

* **Box assembly** (``init_box`` / ``config_limit_box`` / ``velocity_limit_box``):
  plain per-world kernels that intersect each limit's contribution into a shared
  per-world box ``[lo, hi]`` on the tangent step ``dq``. These run on CPU or GPU.

* **Box-ADMM solve** (``get_admm_box_kernel``): an ``nv``-specialised, one-block-
  per-world tiled kernel that factors ``M = H + rho I`` once with the existing
  tile Cholesky and runs a fixed number of ADMM iterations, returning the box-
  projected step ``dq = z`` (feasible at every iteration). Tile Cholesky is
  cuSolverDx / GPU-only, so this kernel only runs on CUDA; the NumPy reference in
  ``tests/helpers_admm.py`` mirrors its math for CPU validation.
"""

from __future__ import annotations

from typing import Any

import warp as wp

# Sentinel for an unconstrained dof (well below mjMAXVAL, safe in float32).
BOX_INF = wp.constant(1.0e9)


@wp.func
def _fmax(a: float, b: float) -> float:
    return wp.max(a, b)


@wp.func
def _fmin(a: float, b: float) -> float:
    return wp.min(a, b)


@wp.kernel
def init_box(
    lo: wp.array2d[float],
    hi: wp.array2d[float],
):
    """Reset the per-world box to (-inf, +inf) before limits intersect in."""
    worldid, i = wp.tid()
    lo[worldid, i] = -BOX_INF
    hi[worldid, i] = BOX_INF


@wp.kernel
def config_limit_box(
    q: wp.array2d[float],
    qposadr: wp.array[int],
    dofadr: wp.array[int],
    lower: wp.array[float],
    upper: wp.array[float],
    gain: float,
    n_limited: int,
    lo: wp.array2d[float],
    hi: wp.array2d[float],
):
    """Intersect the configuration position limit into ``[lo, hi]``.

    For each limited hinge/slide dof (1 qpos <-> 1 dof), matching mink's
    ``ConfigurationLimit`` (``G=[P;-P]``, ``h=[gain*(upper-q); gain*(q-lower)]``):

        gain*(lower - q)  <=  dq  <=  gain*(upper - q)
    """
    worldid = wp.tid()
    for k in range(n_limited):
        qa = qposadr[k]
        va = dofadr[k]
        qk = q[worldid, qa]
        hb = gain * (upper[k] - qk)
        lb = gain * (lower[k] - qk)
        hi[worldid, va] = wp.min(hi[worldid, va], hb)
        lo[worldid, va] = wp.max(lo[worldid, va], lb)


@wp.kernel
def compute_rho(
    H: wp.array3d[float],
    nv: int,
    rho_scale: float,
    rho_min: float,
    rho_max: float,
    rho_out: wp.array[float],
):
    """Per-world ADMM penalty ``rho = clamp(rho_scale*sqrt(dmin*dmax), min, max)``.

    ``dmin``/``dmax`` are the smallest/largest diagonal entries of ``H``, so
    their geometric mean approximates ``sqrt(lambda_min*lambda_max)`` — the
    rho that minimises ADMM's condition number and thus its iteration count.
    This self-scales across scenes (a well-conditioned 2-dof arm and a rank-
    deficient 9-dof panda need very different absolute rho) while staying
    branchless and device-side (graph-capturable). rho only affects convergence
    *speed*; the ADMM fixed point is the true QP optimum regardless.
    """
    worldid = wp.tid()
    dmin = H[worldid, 0, 0]
    dmax = H[worldid, 0, 0]
    for i in range(1, nv):
        d = H[worldid, i, i]
        dmin = wp.min(dmin, d)
        dmax = wp.max(dmax, d)
    g = wp.sqrt(wp.max(dmin, 0.0) * wp.max(dmax, 0.0))
    rho_out[worldid] = wp.clamp(rho_scale * g, rho_min, rho_max)


@wp.kernel
def velocity_limit_box(
    dofadr: wp.array[int],
    vmax: wp.array[float],
    dt: float,
    nb: int,
    lo: wp.array2d[float],
    hi: wp.array2d[float],
):
    """Intersect the symmetric velocity limit ``+-dt*vmax`` into ``[lo, hi]``."""
    worldid = wp.tid()
    for k in range(nb):
        va = dofadr[k]
        b = dt * vmax[k]
        hi[worldid, va] = wp.min(hi[worldid, va], b)
        lo[worldid, va] = wp.max(lo[worldid, va], -b)


# Cache of (nv, iters)-specialised box-ADMM kernels.
_ADMM_CACHE: dict[tuple[int, int], Any] = {}

# Threads per block for the tiled solve (multiple of 32).
TILE_THREADS = 32


def get_admm_box_kernel(nv: int, iters: int):
    """Return a batched box-ADMM QP kernel specialised for ``(nv, iters)``.

    Solves, one block per world via :func:`wp.launch_tiled`::

        min  1/2 dq^T H dq + c^T dq   s.t.   lo <= dq <= hi

    ``M = H + rho I`` is factored once; each of the ``iters`` inner steps is a
    cached tile Cholesky solve, a box clip, and a dual update. Returns
    ``dq = z`` which lies in ``[lo, hi]`` at every iteration.
    """
    key = (int(nv), int(iters))
    if key in _ADMM_CACHE:
        return _ADMM_CACHE[key]

    NV = int(nv)
    ITERS = int(iters)

    def _admm_box_solve(
        H: wp.array3d[float],
        b: wp.array2d[float],  # b = -c = W^T e
        lo: wp.array2d[float],
        hi: wp.array2d[float],
        rho: wp.array[float],  # per-world ADMM penalty
        alpha: float,
        dq: wp.array2d[float],
    ):
        worldid = wp.tid()
        r = rho[worldid]

        A = wp.tile_load(H[worldid], shape=(NV, NV))
        bt = wp.tile_load(b[worldid], shape=(NV,))
        lot = wp.tile_load(lo[worldid], shape=(NV,))
        hit = wp.tile_load(hi[worldid], shape=(NV,))

        # M = H + rho I, factored once.
        d = wp.tile_ones(shape=(NV,), dtype=float) * r
        M = wp.tile_diag_add(A, d)
        L = wp.tile_cholesky(M)

        # Warm start: z = clip(M^{-1} b), u = 0.
        x = wp.tile_cholesky_solve(L, bt)
        z = wp.tile_map(_fmin, wp.tile_map(_fmax, x, lot), hit)
        u = wp.tile_zeros(shape=(NV,), dtype=float)

        for _ in range(ITERS):
            rhs = (z - u) * r + bt
            x = wp.tile_cholesky_solve(L, rhs)
            x_hat = x * alpha + z * (1.0 - alpha)
            t = x_hat + u
            z = wp.tile_map(_fmin, wp.tile_map(_fmax, t, lot), hit)
            u = u + x_hat - z

        wp.tile_store(dq[worldid], z)

    _admm_box_solve.__name__ = f"admm_box_solve_{NV}_{ITERS}"
    _admm_box_solve.__qualname__ = _admm_box_solve.__name__
    kernel = wp.kernel(enable_backward=False, module="unique")(_admm_box_solve)
    _ADMM_CACHE[key] = kernel
    return kernel


def launch_admm_box_solve(
    kernel,
    *,
    nworld: int,
    H: wp.array,
    b: wp.array,
    lo: wp.array,
    hi: wp.array,
    rho: wp.array,
    alpha: float,
    dq: wp.array,
    device: str | None = None,
) -> None:
    """Launch the box-ADMM solve: one block per world."""
    wp.launch_tiled(
        kernel,
        dim=[nworld],
        inputs=[H, b, lo, hi, rho, alpha],
        outputs=[dq],
        block_dim=TILE_THREADS,
        device=device,
    )


# --------------------------------------------------------------------------- #
# General dense inequality: G dq <= h  (OSQP-ADMM, factor-once)
# --------------------------------------------------------------------------- #
@wp.kernel
def init_ineq(
    G: wp.array3d[float],
    h: wp.array2d[float],
):
    """Reset the padded inequality block to inert rows (``0 dq <= +inf``).

    Launched over ``(nworld, m)``; zeroes the whole row of ``G`` and sets ``h``
    to ``+BOX_INF`` so any row a limit does not overwrite is satisfied trivially
    and contributes nothing to ``G^T G``.
    """
    worldid, i = wp.tid()
    h[worldid, i] = BOX_INF
    nv = G.shape[2]
    for j in range(nv):
        G[worldid, i, j] = 0.0


@wp.kernel
def config_limit_ineq(
    q: wp.array2d[float],
    qposadr: wp.array[int],
    dofadr: wp.array[int],
    lower: wp.array[float],
    upper: wp.array[float],
    gain: float,
    n_limited: int,
    row_offset: int,
    G: wp.array3d[float],
    h: wp.array2d[float],
):
    """Scatter mink's ``G=[P;-P]``, ``h=[gain*(upper-q); gain*(q-lower)]`` rows.

    The same configuration limit the box kernel intersects, written instead as
    ``2*n_limited`` explicit dense inequality rows starting at ``row_offset``
    (the ``+P`` rows first, then the ``-P`` rows), for the general-inequality
    solve path.
    """
    worldid = wp.tid()
    for k in range(n_limited):
        qa = qposadr[k]
        va = dofadr[k]
        qk = q[worldid, qa]
        r_up = row_offset + k
        r_lo = row_offset + n_limited + k
        G[worldid, r_up, va] = 1.0
        h[worldid, r_up] = gain * (upper[k] - qk)
        G[worldid, r_lo, va] = -1.0
        h[worldid, r_lo] = gain * (qk - lower[k])


@wp.kernel
def velocity_limit_ineq(
    dofadr: wp.array[int],
    vmax: wp.array[float],
    dt: float,
    nb: int,
    row_offset: int,
    G: wp.array3d[float],
    h: wp.array2d[float],
):
    """Scatter the symmetric velocity limit as ``+-e_i dq <= dt*vmax`` rows."""
    worldid = wp.tid()
    for k in range(nb):
        va = dofadr[k]
        b = dt * vmax[k]
        r_up = row_offset + k
        r_lo = row_offset + nb + k
        G[worldid, r_up, va] = 1.0
        h[worldid, r_up] = b
        G[worldid, r_lo, va] = -1.0
        h[worldid, r_lo] = b


@wp.kernel
def linear_ineq_scatter(
    Gc: wp.array2d[float],  # (m_rows, nv) constant rows
    hc: wp.array[float],  # (m_rows,)
    row_offset: int,
    G: wp.array3d[float],
    h: wp.array2d[float],
):
    """Broadcast constant dense rows ``Gc dq <= hc`` into every world's block."""
    worldid, i = wp.tid()
    nv = G.shape[2]
    r = row_offset + i
    h[worldid, r] = hc[i]
    for j in range(nv):
        G[worldid, r, j] = Gc[i, j]


# Cache of (nv, m, iters)-specialised inequality-ADMM kernels.
_ADMM_INEQ_CACHE: dict[tuple[int, int, int], Any] = {}


def get_admm_ineq_kernel(nv: int, m: int, iters: int):
    """Return a batched dense-inequality QP kernel specialised for ``(nv, m, iters)``.

    Solves, one block per world via :func:`wp.launch_tiled`::

        min  1/2 dq^T H dq + c^T dq   s.t.   G dq <= h

    by the reduced (Schur-normal) OSQP-ADMM: the SPD normal matrix
    ``M = H + sigma I + rho G^T G`` is factored once, then each of ``iters``
    steps is a cached tile-Cholesky solve, a projection of the constraint image
    ``G dq`` onto ``(-inf, h]``, and a dual update. Returns ``dq = x`` whose
    feasibility ``G dq <= h`` tightens with ``iters`` (asymptotic, unlike the box
    solver's exact-at-every-step projection). ``sigma > 0`` keeps ``M`` SPD.

    Per-world vectors are stored as ``(n,)`` rows and reshaped to ``(n, 1)``
    columns only where the two matmuls (``G x`` and ``G^T w``) need them.
    """
    key = (int(nv), int(m), int(iters))
    if key in _ADMM_INEQ_CACHE:
        return _ADMM_INEQ_CACHE[key]

    NV = int(nv)
    M = int(m)
    ITERS = int(iters)

    def _admm_ineq_solve(
        H: wp.array3d[float],
        b: wp.array2d[float],  # b = -c = W^T e
        G: wp.array3d[float],  # (nworld, m, nv)
        h: wp.array2d[float],  # (nworld, m)
        rho: wp.array[float],  # per-world ADMM penalty
        sigma: float,  # SPD floor added to M's diagonal
        alpha: float,
        dq: wp.array2d[float],
    ):
        worldid = wp.tid()
        r = rho[worldid]

        Hd = wp.tile_load(H[worldid], shape=(NV, NV))
        Gt = wp.tile_load(G[worldid], shape=(M, NV))
        ht = wp.tile_load(h[worldid], shape=(M,))
        bt = wp.tile_load(b[worldid], shape=(NV,))

        GT = wp.tile_transpose(Gt)  # (nv, m)
        GtG = wp.tile_matmul(GT, Gt)  # (nv, nv)

        # M = H + rho G^T G + sigma I, factored once.
        dsig = wp.tile_ones(shape=(NV,), dtype=float) * sigma
        Msys = wp.tile_diag_add(Hd + GtG * r, dsig)
        L = wp.tile_cholesky(Msys)

        # Warm start x = M^{-1} b; z = clip(G x, ., h); y = 0.
        x = wp.tile_cholesky_solve(L, bt)  # (nv,)
        xc = wp.tile_reshape(x, shape=(NV, 1))
        z = wp.tile_reshape(wp.tile_matmul(Gt, xc), shape=(M,))  # G x  (m,)
        z = wp.tile_map(_fmin, z, ht)
        y = wp.tile_zeros(shape=(M,), dtype=float)

        for _ in range(ITERS):
            w1 = wp.tile_reshape(z * r - y, shape=(M, 1))
            Gtw = wp.tile_reshape(wp.tile_matmul(GT, w1), shape=(NV,))  # G^T(rho z - y)
            rhs = x * sigma + bt + Gtw
            xt = wp.tile_cholesky_solve(L, rhs)  # (nv,)
            xtc = wp.tile_reshape(xt, shape=(NV, 1))
            zt = wp.tile_reshape(wp.tile_matmul(Gt, xtc), shape=(M,))  # G x_tilde
            x = xt * alpha + x * (1.0 - alpha)
            z_hat = zt * alpha + z * (1.0 - alpha)
            t = z_hat + y * (1.0 / r)
            z_new = wp.tile_map(_fmin, t, ht)  # project onto (-inf, h]
            y = y + (z_hat - z_new) * r
            z = z_new

        wp.tile_store(dq[worldid], x)

    _admm_ineq_solve.__name__ = f"admm_ineq_solve_{NV}_{M}_{ITERS}"
    _admm_ineq_solve.__qualname__ = _admm_ineq_solve.__name__
    kernel = wp.kernel(enable_backward=False, module="unique")(_admm_ineq_solve)
    _ADMM_INEQ_CACHE[key] = kernel
    return kernel


def launch_admm_ineq_solve(
    kernel,
    *,
    nworld: int,
    H: wp.array,
    b: wp.array,
    G: wp.array,
    h: wp.array,
    rho: wp.array,
    sigma: float,
    alpha: float,
    dq: wp.array,
    device: str | None = None,
) -> None:
    """Launch the dense-inequality ADMM solve: one block per world."""
    wp.launch_tiled(
        kernel,
        dim=[nworld],
        inputs=[H, b, G, h, rho, sigma, alpha],
        outputs=[dq],
        block_dim=TILE_THREADS,
        device=device,
    )
