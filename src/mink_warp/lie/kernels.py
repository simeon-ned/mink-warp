"""Batched Warp kernels for kinematics helpers and frame tasks."""

from __future__ import annotations

import warp as wp

from .wp_ops import (
    mat_to_quat_wxyz,
    quat_conj_wxyz,
    quat_mul_wxyz,
    quat_rotate_wxyz,
    se3_jlog,
    se3_rminus,
    spatial_mat_mul_vec,
)


@wp.kernel
def fill_body_frame_query(
    xpos: wp.array2d[wp.vec3],
    frame_id: int,
    body_id: int,
    point_out: wp.array[wp.vec3],
    body_out: wp.array[int],
):
    worldid = wp.tid()
    point_out[worldid] = xpos[worldid, frame_id]
    body_out[worldid] = body_id


@wp.kernel
def fill_site_frame_query(
    site_xpos: wp.array2d[wp.vec3],
    site_bodyid: wp.array[int],
    frame_id: int,
    point_out: wp.array[wp.vec3],
    body_out: wp.array[int],
):
    worldid = wp.tid()
    point_out[worldid] = site_xpos[worldid, frame_id]
    body_out[worldid] = site_bodyid[frame_id]


@wp.kernel
def fill_geom_frame_query(
    geom_xpos: wp.array2d[wp.vec3],
    geom_bodyid: wp.array[int],
    frame_id: int,
    point_out: wp.array[wp.vec3],
    body_out: wp.array[int],
):
    worldid = wp.tid()
    point_out[worldid] = geom_xpos[worldid, frame_id]
    body_out[worldid] = geom_bodyid[frame_id]


@wp.kernel
def body_frame_jacobian(
    xmat: wp.array2d[wp.mat33],
    frame_id: int,
    jacp_world: wp.array3d[float],
    jacr_world: wp.array3d[float],
    jac_body: wp.array3d[float],
):
    """Convert world-aligned Jacobians to body-frame: R_fw @ jac_world."""
    worldid, dofid = wp.tid()
    R_wf = xmat[worldid, frame_id]
    # R_fw = R_wf^T
    jp = wp.vec3(
        jacp_world[worldid, 0, dofid],
        jacp_world[worldid, 1, dofid],
        jacp_world[worldid, 2, dofid],
    )
    jr = wp.vec3(
        jacr_world[worldid, 0, dofid],
        jacr_world[worldid, 1, dofid],
        jacr_world[worldid, 2, dofid],
    )
    jp_b = wp.transpose(R_wf) * jp
    jr_b = wp.transpose(R_wf) * jr
    jac_body[worldid, 0, dofid] = jp_b[0]
    jac_body[worldid, 1, dofid] = jp_b[1]
    jac_body[worldid, 2, dofid] = jp_b[2]
    jac_body[worldid, 3, dofid] = jr_b[0]
    jac_body[worldid, 4, dofid] = jr_b[1]
    jac_body[worldid, 5, dofid] = jr_b[2]


@wp.kernel
def frame_pose_wxyz_xyz(
    xpos: wp.array2d[wp.vec3],
    xmat: wp.array2d[wp.mat33],
    frame_id: int,
    pose_out: wp.array2d[float],  # (nworld, 7)
):
    worldid = wp.tid()
    q = mat_to_quat_wxyz(xmat[worldid, frame_id])
    p = xpos[worldid, frame_id]
    pose_out[worldid, 0] = q[0]
    pose_out[worldid, 1] = q[1]
    pose_out[worldid, 2] = q[2]
    pose_out[worldid, 3] = q[3]
    pose_out[worldid, 4] = p[0]
    pose_out[worldid, 5] = p[1]
    pose_out[worldid, 6] = p[2]


@wp.kernel
def broadcast_pose(
    pose: wp.array[float],  # (7,)
    pose_out: wp.array2d[float],  # (nworld, 7)
):
    worldid = wp.tid()
    for i in range(7):
        pose_out[worldid, i] = pose[i]


@wp.kernel
def copy_poses(
    pose_in: wp.array2d[float],
    pose_out: wp.array2d[float],
):
    worldid = wp.tid()
    for i in range(7):
        pose_out[worldid, i] = pose_in[worldid, i]


@wp.kernel
def frame_task_error_jacobian(
    target_pose: wp.array2d[float],  # (nworld, 7) wxyz_xyz
    frame_pose: wp.array2d[float],  # (nworld, 7) wxyz_xyz
    jac_body: wp.array3d[float],  # (nworld, 6, nv)
    nv: int,
    error_out: wp.array2d[float],  # (nworld, 6)
    jac_out: wp.array3d[float],  # (nworld, 6, nv)
):
    """Mink FrameTask: e = target.minus(frame), J = -jlog(T_tb) @ jac_body."""
    worldid = wp.tid()

    q_t = wp.vec4(
        target_pose[worldid, 0],
        target_pose[worldid, 1],
        target_pose[worldid, 2],
        target_pose[worldid, 3],
    )
    t_t = wp.vec3(
        target_pose[worldid, 4],
        target_pose[worldid, 5],
        target_pose[worldid, 6],
    )
    q_f = wp.vec4(
        frame_pose[worldid, 0],
        frame_pose[worldid, 1],
        frame_pose[worldid, 2],
        frame_pose[worldid, 3],
    )
    t_f = wp.vec3(
        frame_pose[worldid, 4],
        frame_pose[worldid, 5],
        frame_pose[worldid, 6],
    )

    # e = target.minus(frame)
    err = se3_rminus(q_t, t_t, q_f, t_f)
    for i in range(6):
        error_out[worldid, i] = err[i]

    # T_tb = target^{-1} @ frame
    q_inv = quat_conj_wxyz(q_t)
    t_inv = quat_rotate_wxyz(q_inv, -t_t)
    q_tb = quat_mul_wxyz(q_inv, q_f)
    t_tb = quat_rotate_wxyz(q_inv, t_f) + t_inv
    jlog = se3_jlog(q_tb, t_tb)

    for dofid in range(nv):
        col = wp.spatial_vector(
            jac_body[worldid, 0, dofid],
            jac_body[worldid, 1, dofid],
            jac_body[worldid, 2, dofid],
            jac_body[worldid, 3, dofid],
            jac_body[worldid, 4, dofid],
            jac_body[worldid, 5, dofid],
        )
        jcol = spatial_mat_mul_vec(jlog, col)
        for i in range(6):
            jac_out[worldid, i, dofid] = -jcol[i]


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


@wp.kernel
def weighted_residual(
    error: wp.array2d[float],
    jacobian: wp.array3d[float],
    cost: wp.array[float],
    gain: float,
    lm_damping: float,
    k: int,
    nv: int,
    weighted_jac: wp.array3d[float],
    weighted_err: wp.array2d[float],
    mu_out: wp.array[float],
):
    worldid = wp.tid()
    mu = float(0.0)
    for i in range(k):
        we = cost[i] * (-gain * error[worldid, i])
        weighted_err[worldid, i] = we
        mu += we * we
        for j in range(nv):
            weighted_jac[worldid, i, j] = cost[i] * jacobian[worldid, i, j]
    mu_out[worldid] = lm_damping * mu
