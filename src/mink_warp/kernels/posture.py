"""Posture-task kernels."""

from __future__ import annotations

import warp as wp

from mujoco_warp._src.math import quat_sub
from mujoco_warp._src.types import JointType

@wp.kernel
def posture_error(
    q: wp.array2d[float],
    target_q: wp.array2d[float],
    nv: int,
    error_out: wp.array2d[float],
):
    """Hinge/slide posture error: e = q - q* (matches mj_differentiatePos when nq==nv)."""
    worldid = wp.tid()
    for i in range(nv):
        error_out[worldid, i] = q[worldid, i] - target_q[worldid, i]


@wp.kernel
def posture_error_joints(
    q: wp.array2d[float],
    target_q: wp.array2d[float],
    jnt_type: wp.array[int],
    jnt_qposadr: wp.array[int],
    jnt_dofadr: wp.array[int],
    njnt: int,
    nv: int,
    error_out: wp.array2d[float],
):
    """Device ``mj_differentiatePos`` (dt=1): e = q ⊖ q* for all joint types."""
    worldid = wp.tid()
    for i in range(nv):
        error_out[worldid, i] = 0.0

    for j in range(njnt):
        jtype = jnt_type[j]
        qadr = jnt_qposadr[j]
        vadr = jnt_dofadr[j]

        if jtype == JointType.FREE:
            for k in range(3):
                error_out[worldid, vadr + k] = (
                    q[worldid, qadr + k] - target_q[worldid, qadr + k]
                )
            # MuJoCo/mjwarp quat layout in qpos is wxyz stored as wp.quat(w,x,y,z).
            qa = wp.quat(
                q[worldid, qadr + 3],
                q[worldid, qadr + 4],
                q[worldid, qadr + 5],
                q[worldid, qadr + 6],
            )
            qb = wp.quat(
                target_q[worldid, qadr + 3],
                target_q[worldid, qadr + 4],
                target_q[worldid, qadr + 5],
                target_q[worldid, qadr + 6],
            )
            vel = quat_sub(qa, qb)
            error_out[worldid, vadr + 3] = vel[0]
            error_out[worldid, vadr + 4] = vel[1]
            error_out[worldid, vadr + 5] = vel[2]

        elif jtype == JointType.BALL:
            qa = wp.quat(
                q[worldid, qadr + 0],
                q[worldid, qadr + 1],
                q[worldid, qadr + 2],
                q[worldid, qadr + 3],
            )
            qb = wp.quat(
                target_q[worldid, qadr + 0],
                target_q[worldid, qadr + 1],
                target_q[worldid, qadr + 2],
                target_q[worldid, qadr + 3],
            )
            vel = quat_sub(qa, qb)
            error_out[worldid, vadr + 0] = vel[0]
            error_out[worldid, vadr + 1] = vel[1]
            error_out[worldid, vadr + 2] = vel[2]

        else:
            # Hinge / slide.
            error_out[worldid, vadr] = q[worldid, qadr] - target_q[worldid, qadr]


@wp.kernel
def posture_jacobian_eye(
    nv: int,
    jac_out: wp.array3d[float],
):
    worldid, i, j = wp.tid()
    if i == j:
        jac_out[worldid, i, j] = 1.0
    else:
        jac_out[worldid, i, j] = 0.0


@wp.kernel
def zero_free_joint_rows(
    error: wp.array2d[float],
    jac: wp.array3d[float],
    v_ids: wp.array[int],
    n_free_v: int,
    nv: int,
):
    worldid = wp.tid()
    for k in range(n_free_v):
        vid = v_ids[k]
        error[worldid, vid] = 0.0
        for j in range(nv):
            jac[worldid, j, vid] = 0.0
            # also zero column? Mink zeros jac[:, v_ids] i.e. columns
            # jac is (nv, nv), jac[:, v_ids] = 0 means columns
        for i in range(nv):
            jac[worldid, i, vid] = 0.0


@wp.kernel
def broadcast_q(
    q: wp.array[float],
    nq: int,
    q_out: wp.array2d[float],
):
    worldid = wp.tid()
    for i in range(nq):
        q_out[worldid, i] = q[i]


