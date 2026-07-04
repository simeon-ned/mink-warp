"""Kernels for the batched Levenberg-Marquardt IK optimizer.

All solvers minimise the same per-world least-squares cost

    C(q) = 1/2 * sum_tasks || cost_i * error_i(q) ||^2

with weighted Jacobian ``W = cost * J`` and residual ``r = cost * error``.
LM forms the Gauss-Newton system ``H = W^T W``, gradient ``g = W^T r`` and solves
``(H + lambda I) delta = -g`` per world, then accepts/rejects the step with a
trust-region gain ratio (Nielsen's lambda update).
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def zero_lm(
    H: wp.array3d[float],
    g: wp.array2d[float],
    C: wp.array[float],
    nv: int,
):
    """Zero the normal-equation accumulators for one world."""
    worldid = wp.tid()
    C[worldid] = 0.0
    for i in range(nv):
        g[worldid, i] = 0.0
        for j in range(nv):
            H[worldid, i, j] = 0.0


@wp.kernel
def accumulate_lm(
    error: wp.array2d[float],
    jacobian: wp.array3d[float],
    cost: wp.array[float],
    k: int,
    nv: int,
    H: wp.array3d[float],
    g: wp.array2d[float],
    C: wp.array[float],
):
    """Accumulate ``H += W^T W``, ``g += W^T r``, ``C += 1/2 ||r||^2``.

    ``W = cost * jacobian`` and ``r = cost * error`` (Gauss-Newton form of the
    weighted task residual). No ``gain`` / ``lm_damping``: LM owns its damping.
    """
    worldid = wp.tid()
    c_acc = float(0.0)
    for i in range(k):
        ci = cost[i]
        ri = ci * error[worldid, i]
        c_acc += 0.5 * ri * ri
        for a in range(nv):
            wa = ci * jacobian[worldid, i, a]
            g[worldid, a] += wa * ri
            for b in range(nv):
                H[worldid, a, b] += wa * (ci * jacobian[worldid, i, b])
    C[worldid] += c_acc


@wp.kernel
def add_lm_lambda(
    H: wp.array3d[float],
    lam: wp.array[float],
    nv: int,
):
    """Damp the Gauss-Newton Hessian: ``H[i, i] += lambda`` (per world)."""
    worldid, i = wp.tid()
    H[worldid, i, i] += lam[worldid]


@wp.kernel
def lm_pred_reduction(
    delta: wp.array2d[float],
    g: wp.array2d[float],
    lam: wp.array[float],
    nv: int,
    pred: wp.array[float],
):
    """Predicted cost reduction of the LM model: ``1/2 (lambda||delta||^2 - g.delta)``.

    Non-negative because ``delta = -(H + lambda I)^{-1} g``.
    """
    worldid = wp.tid()
    gd = float(0.0)
    dd = float(0.0)
    for i in range(nv):
        gd += g[worldid, i] * delta[worldid, i]
        dd += delta[worldid, i] * delta[worldid, i]
    pred[worldid] = 0.5 * (lam[worldid] * dd - gd)


@wp.kernel
def zero_cost(C: wp.array[float]):
    C[wp.tid()] = 0.0


@wp.kernel
def accumulate_cost(
    error: wp.array2d[float],
    cost: wp.array[float],
    k: int,
    C: wp.array[float],
):
    """Accumulate ``C += 1/2 ||cost * error||^2`` (cost only, for trial eval)."""
    worldid = wp.tid()
    c_acc = float(0.0)
    for i in range(k):
        ri = cost[i] * error[worldid, i]
        c_acc += 0.5 * ri * ri
    C[worldid] += c_acc


@wp.kernel
def lm_accept(
    pred: wp.array[float],
    C_old: wp.array[float],
    C_new: wp.array[float],
    qpos_old: wp.array2d[float],
    delta: wp.array2d[float],
    lam_min: float,
    lam_max: float,
    eps: float,
    nq: int,
    nv: int,
    qpos: wp.array2d[float],
    lam: wp.array[float],
    nu: wp.array[float],
    dq_total: wp.array2d[float],
):
    """Trust-region accept/reject with Nielsen's lambda update, per world.

    Accept (rho > 0): keep the proposed ``qpos``, accumulate ``delta`` into the
    net step, shrink lambda by ``max(1/3, 1 - (2 rho - 1)^3)``, reset nu.
    Reject: restore ``qpos`` from ``qpos_old``, grow lambda by nu, double nu.
    """
    worldid = wp.tid()
    p = pred[worldid]
    actual = C_old[worldid] - C_new[worldid]
    rho = float(-1.0)
    if p > eps:
        rho = actual / p

    if rho > 0.0:
        for i in range(nv):
            dq_total[worldid, i] += delta[worldid, i]
        t = 2.0 * rho - 1.0
        factor = 1.0 - t * t * t
        if factor < (1.0 / 3.0):
            factor = 1.0 / 3.0
        new_lam = lam[worldid] * factor
        nu[worldid] = 2.0
    else:
        for i in range(nq):
            qpos[worldid, i] = qpos_old[worldid, i]
        new_lam = lam[worldid] * nu[worldid]
        nu[worldid] = nu[worldid] * 2.0

    if new_lam < lam_min:
        new_lam = lam_min
    if new_lam > lam_max:
        new_lam = lam_max
    lam[worldid] = new_lam
