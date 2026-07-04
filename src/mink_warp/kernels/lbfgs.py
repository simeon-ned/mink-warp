"""Kernels for the batched L-BFGS IK optimizer.

Minimises the same cost ``C(q) = 1/2 sum || cost_i * error_i(q) ||^2`` as LM, but
quasi-Newton: it never forms the Hessian. Each iteration builds a search
direction from the gradient and a short history of ``(s, y)`` pairs via the
two-loop recursion, then a parallel line search over a fixed set of step sizes
picks the best per world. Ports ``newton/_src/sim/ik/ik_lbfgs_optimizer.py``.
"""

from __future__ import annotations

import warp as wp


@wp.kernel
def zero_grad(g: wp.array2d[float], C: wp.array[float], nv: int):
    worldid = wp.tid()
    C[worldid] = 0.0
    for i in range(nv):
        g[worldid, i] = 0.0


@wp.kernel
def accumulate_grad(
    error: wp.array2d[float],
    jacobian: wp.array3d[float],
    cost: wp.array[float],
    k: int,
    nv: int,
    g: wp.array2d[float],
    C: wp.array[float],
):
    """``g += W^T r``, ``C += 1/2 ||r||^2`` with ``W = cost J``, ``r = cost error``."""
    worldid = wp.tid()
    c_acc = float(0.0)
    for i in range(k):
        ci = cost[i]
        ri = ci * error[worldid, i]
        c_acc += 0.5 * ri * ri
        for a in range(nv):
            g[worldid, a] += (ci * jacobian[worldid, i, a]) * ri
    C[worldid] += c_acc


@wp.kernel
def lbfgs_gamma(
    s_hist: wp.array3d[float],
    y_hist: wp.array3d[float],
    count: int,
    nv: int,
    gamma: wp.array[float],
):
    """Initial-Hessian scaling ``gamma = (s.y)/(y.y)`` from the newest pair."""
    worldid = wp.tid()
    if count <= 0:
        gamma[worldid] = 1.0
        return
    j = count - 1
    sy = float(0.0)
    yy = float(0.0)
    for i in range(nv):
        sy += s_hist[worldid, j, i] * y_hist[worldid, j, i]
        yy += y_hist[worldid, j, i] * y_hist[worldid, j, i]
    if yy > 1.0e-30 and sy > 0.0:
        gamma[worldid] = sy / yy
    else:
        gamma[worldid] = 1.0


@wp.kernel
def lbfgs_two_loop(
    g: wp.array2d[float],
    s_hist: wp.array3d[float],
    y_hist: wp.array3d[float],
    rho_hist: wp.array2d[float],
    gamma: wp.array[float],
    count: int,
    nv: int,
    alpha: wp.array2d[float],
    q_work: wp.array2d[float],
    p: wp.array2d[float],
):
    """Two-loop recursion: ``p = -H_inv g`` (chronological history 0..count-1)."""
    worldid = wp.tid()
    for i in range(nv):
        q_work[worldid, i] = g[worldid, i]
    # First loop: newest -> oldest.
    for jj in range(count):
        j = count - 1 - jj
        sdot = float(0.0)
        for i in range(nv):
            sdot += s_hist[worldid, j, i] * q_work[worldid, i]
        a = rho_hist[worldid, j] * sdot
        alpha[worldid, j] = a
        for i in range(nv):
            q_work[worldid, i] -= a * y_hist[worldid, j, i]
    # Scale by the initial Hessian estimate.
    for i in range(nv):
        q_work[worldid, i] = gamma[worldid] * q_work[worldid, i]
    # Second loop: oldest -> newest.
    for j in range(count):
        ydot = float(0.0)
        for i in range(nv):
            ydot += y_hist[worldid, j, i] * q_work[worldid, i]
        beta = rho_hist[worldid, j] * ydot
        coef = alpha[worldid, j] - beta
        for i in range(nv):
            q_work[worldid, i] += coef * s_hist[worldid, j, i]
    for i in range(nv):
        p[worldid, i] = -q_work[worldid, i]


@wp.kernel
def ls_init(C_best: wp.array[float], best_alpha: wp.array[float]):
    worldid = wp.tid()
    C_best[worldid] = 1.0e30
    best_alpha[worldid] = 0.0


@wp.kernel
def ls_update(
    C_cand: wp.array[float],
    C_old: wp.array[float],
    alpha_val: float,
    eps: float,
    C_best: wp.array[float],
    best_alpha: wp.array[float],
):
    """Keep the lowest-cost candidate that beats the origin (sufficient decrease)."""
    worldid = wp.tid()
    if C_cand[worldid] < C_best[worldid] and C_cand[worldid] < C_old[worldid] - eps:
        C_best[worldid] = C_cand[worldid]
        best_alpha[worldid] = alpha_val


@wp.kernel
def apply_step(
    p: wp.array2d[float],
    best_alpha: wp.array[float],
    nv: int,
    step: wp.array2d[float],
    dq_total: wp.array2d[float],
):
    """``step = best_alpha * p``; accumulate into the net tangent step."""
    worldid = wp.tid()
    for i in range(nv):
        si = best_alpha[worldid] * p[worldid, i]
        step[worldid, i] = si
        dq_total[worldid, i] += si


@wp.kernel
def shift_history(
    m: int,
    nv: int,
    s_hist: wp.array3d[float],
    y_hist: wp.array3d[float],
    rho_hist: wp.array2d[float],
):
    """Drop the oldest pair, sliding history down one slot (ring at capacity)."""
    worldid = wp.tid()
    for j in range(m - 1):
        rho_hist[worldid, j] = rho_hist[worldid, j + 1]
        for i in range(nv):
            s_hist[worldid, j, i] = s_hist[worldid, j + 1, i]
            y_hist[worldid, j, i] = y_hist[worldid, j + 1, i]


@wp.kernel
def push_history(
    step: wp.array2d[float],
    g_old: wp.array2d[float],
    g_new: wp.array2d[float],
    slot: int,
    eps: float,
    nv: int,
    s_hist: wp.array3d[float],
    y_hist: wp.array3d[float],
    rho_hist: wp.array2d[float],
):
    """Store ``(s, y)`` if curvature ``s.y > eps ||y||^2``, else a zero slot.

    Relative test (standard L-BFGS skip rule): an absolute floor would admit a
    near-orthogonal pair with tiny ``s.y``, giving ``rho = 1/(s.y)`` huge and a
    garbage two-loop direction that pollutes the history for up to ``m`` steps.
    """
    worldid = wp.tid()
    sy = float(0.0)
    yy = float(0.0)
    for i in range(nv):
        yi = g_new[worldid, i] - g_old[worldid, i]
        sy += step[worldid, i] * yi
        yy += yi * yi
    if sy > eps * yy:
        for i in range(nv):
            s_hist[worldid, slot, i] = step[worldid, i]
            y_hist[worldid, slot, i] = g_new[worldid, i] - g_old[worldid, i]
        rho_hist[worldid, slot] = 1.0 / sy
    else:
        for i in range(nv):
            s_hist[worldid, slot, i] = 0.0
            y_hist[worldid, slot, i] = 0.0
        rho_hist[worldid, slot] = 0.0
