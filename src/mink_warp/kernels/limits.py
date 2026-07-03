"""Kernels for soft configuration limits."""

from __future__ import annotations

import warp as wp


@wp.kernel
def joint_limit_error_jac(
    q: wp.array2d[float],
    lower: wp.array[float],
    upper: wp.array[float],
    qposadr: wp.array[int],
    dofadr: wp.array[int],
    n_limited: int,
    nv: int,
    error_out: wp.array2d[float],
    jac_out: wp.array3d[float],
):
    """Soft limit residual for hinge/slide joints (1 qpos ↔ 1 dof).

    error[dof] = q - upper if q > upper, q - lower if q < lower, else 0.
    Jacobian is identity on limited dofs (zero elsewhere).
    """
    worldid = wp.tid()
    for i in range(nv):
        error_out[worldid, i] = 0.0
        for j in range(nv):
            jac_out[worldid, i, j] = 0.0

    for k in range(n_limited):
        qadr = qposadr[k]
        vadr = dofadr[k]
        qk = q[worldid, qadr]
        lo = lower[k]
        hi = upper[k]
        e = float(0.0)
        if qk > hi:
            e = qk - hi
        elif qk < lo:
            e = qk - lo
        error_out[worldid, vadr] = e
        jac_out[worldid, vadr, vadr] = 1.0
