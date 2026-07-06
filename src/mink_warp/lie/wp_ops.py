"""Warp device functions for SO(3)/SE(3), matching Mink's formulas.

Quaternions are stored as ``wxyz`` (MuJoCo / Mink convention), not Warp's
native ``xyzw``.
"""

from __future__ import annotations

import warp as wp

_EPS_F32 = 1.0e-5


@wp.func
def quat_conj_wxyz(q: wp.vec4) -> wp.vec4:
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def quat_mul_wxyz(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    return wp.vec4(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


@wp.func
def quat_rotate_wxyz(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    qv = wp.vec4(0.0, v[0], v[1], v[2])
    r = quat_mul_wxyz(quat_mul_wxyz(q, qv), quat_conj_wxyz(q))
    return wp.vec3(r[1], r[2], r[3])


@wp.func
def mat_to_quat_wxyz(m: wp.mat33) -> wp.vec4:
    q = wp.quat_from_matrix(m)  # xyzw
    return wp.vec4(q[3], q[0], q[1], q[2])


@wp.func
def skew(v: wp.vec3) -> wp.mat33:
    return wp.mat33(
        0.0,
        -v[2],
        v[1],
        v[2],
        0.0,
        -v[0],
        -v[1],
        v[0],
        0.0,
    )


@wp.func
def so3_log(q_in: wp.vec4) -> wp.vec3:
    q = q_in
    if q[0] < 0.0:
        q = -q
    w = q[0]
    v = wp.vec3(q[1], q[2], q[3])
    norm = wp.length(v)
    if norm < _EPS_F32:
        return wp.vec3(0.0, 0.0, 0.0)
    return (2.0 * wp.atan2(norm, w)) * (v / norm)


@wp.func
def so3_ljacinv(omega: wp.vec3) -> wp.mat33:
    theta = wp.length(omega)
    t2 = theta * theta
    if theta < _EPS_F32:
        beta = (1.0 / 12.0) * (1.0 + t2 / 60.0 * (1.0 + t2 / 42.0 * (1.0 + t2 / 40.0)))
    else:
        beta = (1.0 / t2) * (
            1.0 - (theta * wp.sin(theta) / (2.0 * (1.0 - wp.cos(theta))))
        )
    outer = wp.outer(omega, omega)
    eye = wp.identity(n=3, dtype=float)
    s2 = outer - t2 * eye
    return eye - 0.5 * skew(omega) + beta * s2


@wp.func
def se3_log(q: wp.vec4, t: wp.vec3) -> wp.spatial_vector:
    omega = so3_log(q)
    theta = wp.length(omega)
    t2 = theta * theta
    skew_omega = skew(omega)
    skew_omega2 = skew_omega * skew_omega
    if t2 < _EPS_F32:
        vinv = wp.identity(n=3, dtype=float) - 0.5 * skew_omega + skew_omega2 / 12.0
    else:
        half_theta = 0.5 * theta
        vinv = (
            wp.identity(n=3, dtype=float)
            - 0.5 * skew_omega
            + (1.0 - 0.5 * theta * wp.cos(half_theta) / wp.sin(half_theta))
            / t2
            * skew_omega2
        )
    v = vinv * t
    return wp.spatial_vector(v[0], v[1], v[2], omega[0], omega[1], omega[2])


@wp.func
def se3_rminus(
    q_self: wp.vec4, t_self: wp.vec3, q_other: wp.vec4, t_other: wp.vec3
) -> wp.spatial_vector:
    """``self.minus(other)`` = ``(other^{-1} @ self).log()``."""
    q_inv = quat_conj_wxyz(q_other)
    t_inv = quat_rotate_wxyz(q_inv, -t_other)
    q = quat_mul_wxyz(q_inv, q_self)
    t = quat_rotate_wxyz(q_inv, t_self) + t_inv
    return se3_log(q, t)


@wp.func
def _get_Q(c: wp.spatial_vector) -> wp.mat33:
    rho = wp.vec3(c[0], c[1], c[2])
    phi = wp.vec3(c[3], c[4], c[5])
    theta = wp.length(phi)
    t2 = theta * theta
    A = 0.5
    if t2 < _EPS_F32:
        B = (1.0 / 6.0) + (1.0 / 120.0) * t2
        C = -(1.0 / 24.0) + (1.0 / 720.0) * t2
        D = -(1.0 / 60.0)
    else:
        t4 = t2 * t2
        sin_theta = wp.sin(theta)
        cos_theta = wp.cos(theta)
        B = (theta - sin_theta) / (t2 * theta)
        C = (1.0 - 0.5 * t2 - cos_theta) / t4
        D = (2.0 * theta - 3.0 * sin_theta + theta * cos_theta) / (2.0 * t4 * theta)
    V = skew(rho)
    W = skew(phi)
    VW = V * W
    WV = wp.transpose(VW)
    WVW = WV * W
    VWW = VW * W
    return (
        A * V
        + B * (WV + VW + WVW)
        - C * (VWW - wp.transpose(VWW) - 3.0 * WVW)
        + D * (WVW * W + W * WVW)
    )


@wp.func
def se3_ljacinv(c: wp.spatial_vector) -> wp.spatial_matrix:
    phi = wp.vec3(c[3], c[4], c[5])
    theta_sq = wp.dot(phi, phi)
    out = wp.spatial_matrix()
    for i in range(6):
        for j in range(6):
            out[i, j] = 0.0
    if theta_sq < _EPS_F32:
        for i in range(6):
            out[i, i] = 1.0
        return out
    Q = _get_Q(c)
    inv_so3 = so3_ljacinv(phi)
    mid = inv_so3 * Q * inv_so3
    for i in range(3):
        for j in range(3):
            out[i, j] = inv_so3[i, j]
            out[i + 3, j + 3] = inv_so3[i, j]
            out[i, j + 3] = -mid[i, j]
            out[i + 3, j] = 0.0
    return out


@wp.func
def se3_rjacinv(c: wp.spatial_vector) -> wp.spatial_matrix:
    return se3_ljacinv(
        wp.spatial_vector(-c[0], -c[1], -c[2], -c[3], -c[4], -c[5])
    )


@wp.func
def se3_jlog(q: wp.vec4, t: wp.vec3) -> wp.spatial_matrix:
    return se3_rjacinv(se3_log(q, t))


@wp.func
def se3_adjoint(q: wp.vec4, t: wp.vec3) -> wp.spatial_matrix:
    """Adjoint matrix Ad(T) for T = (q, t), wxyz quaternion."""
    e0 = quat_rotate_wxyz(q, wp.vec3(1.0, 0.0, 0.0))
    e1 = quat_rotate_wxyz(q, wp.vec3(0.0, 1.0, 0.0))
    e2 = quat_rotate_wxyz(q, wp.vec3(0.0, 0.0, 1.0))
    R = wp.mat33(e0[0], e1[0], e2[0], e0[1], e1[1], e2[1], e0[2], e1[2], e2[2])
    tx = skew(t) * R
    out = wp.spatial_matrix()
    for i in range(6):
        for j in range(6):
            out[i, j] = 0.0
    for i in range(3):
        for j in range(3):
            out[i, j] = R[i, j]
            out[i, j + 3] = tx[i, j]
            out[i + 3, j + 3] = R[i, j]
    return out


@wp.func
def se3_compose_inv(
    q_root: wp.vec4,
    t_root: wp.vec3,
    q_frame: wp.vec4,
    t_frame: wp.vec3,
):
    """Relative pose root^{-1} @ frame as (q, t)."""
    q_inv = quat_conj_wxyz(q_root)
    t_inv = quat_rotate_wxyz(q_inv, -t_root)
    q = quat_mul_wxyz(q_inv, q_frame)
    t = quat_rotate_wxyz(q_inv, t_frame) + t_inv
    return q, t


@wp.func
def spatial_mat_mul_vec(
    m: wp.spatial_matrix, v: wp.spatial_vector
) -> wp.spatial_vector:
    out = wp.spatial_vector(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    for i in range(6):
        s = float(0.0)
        for j in range(6):
            s += m[i, j] * v[j]
        out[i] = s
    return out
