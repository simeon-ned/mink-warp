"""Kernels for center-of-mass task (Mink-compatible)."""

from __future__ import annotations

import warp as wp

from mujoco_warp._src.support import jac_dof


@wp.func
def _in_subtree(body_parentid: wp.array[int], bodyid: int, root: int) -> bool:
    parentid = bodyid
    while parentid != 0:
        if parentid == root:
            return True
        parentid = body_parentid[parentid]
    return False


@wp.kernel
def com_error(
    subtree_com: wp.array2d[wp.vec3],
    target_com: wp.array2d[float],
    subtree_id: int,
    error_out: wp.array2d[float],
):
    """e = c(q) - c* for subtree ``subtree_id`` (usually 1 = whole robot)."""
    worldid = wp.tid()
    c = subtree_com[worldid, subtree_id]
    error_out[worldid, 0] = c[0] - target_com[worldid, 0]
    error_out[worldid, 1] = c[1] - target_com[worldid, 1]
    error_out[worldid, 2] = c[2] - target_com[worldid, 2]


@wp.kernel
def com_jacobian(
    nbody: int,
    body_parentid: wp.array[int],
    body_rootid: wp.array[int],
    body_mass: wp.array2d[float],
    body_subtreemass: wp.array2d[float],
    dof_bodyid: wp.array[int],
    body_isdofancestor: wp.array2d[int],
    xipos: wp.array2d[wp.vec3],
    subtree_com: wp.array2d[wp.vec3],
    cdof: wp.array2d[wp.spatial_vector],
    subtree_id: int,
    jac_out: wp.array3d[float],
):
    """J = (1/M) sum_i m_i jacp(xipos_i) over bodies in the subtree."""
    worldid, dofid = wp.tid()
    mass_id = worldid % body_mass.shape[0]
    total_mass = body_subtreemass[mass_id, subtree_id]
    if total_mass < 1.0e-12:
        jac_out[worldid, 0, dofid] = 0.0
        jac_out[worldid, 1, dofid] = 0.0
        jac_out[worldid, 2, dofid] = 0.0
        return

    accum = wp.vec3(0.0, 0.0, 0.0)
    for bodyid in range(1, nbody):
        if not _in_subtree(body_parentid, bodyid, subtree_id):
            continue
        m_i = body_mass[mass_id, bodyid]
        if m_i < 1.0e-12:
            continue
        point = xipos[worldid, bodyid]
        jacp, _ = jac_dof(
            body_parentid,
            body_rootid,
            dof_bodyid,
            body_isdofancestor,
            subtree_com,
            cdof,
            point,
            bodyid,
            dofid,
            worldid,
        )
        accum += m_i * jacp

    inv_m = 1.0 / total_mass
    jac_out[worldid, 0, dofid] = accum[0] * inv_m
    jac_out[worldid, 1, dofid] = accum[1] * inv_m
    jac_out[worldid, 2, dofid] = accum[2] * inv_m


@wp.kernel
def broadcast_vec3(
    v: wp.array[float],
    out: wp.array2d[float],
):
    worldid = wp.tid()
    out[worldid, 0] = v[0]
    out[worldid, 1] = v[1]
    out[worldid, 2] = v[2]


@wp.kernel
def copy_vec3_batch(
    src: wp.array2d[float],
    out: wp.array2d[float],
):
    worldid = wp.tid()
    out[worldid, 0] = src[worldid, 0]
    out[worldid, 1] = src[worldid, 1]
    out[worldid, 2] = src[worldid, 2]


@wp.kernel
def subtree_com_to_batch(
    subtree_com: wp.array2d[wp.vec3],
    subtree_id: int,
    out: wp.array2d[float],
):
    worldid = wp.tid()
    c = subtree_com[worldid, subtree_id]
    out[worldid, 0] = c[0]
    out[worldid, 1] = c[1]
    out[worldid, 2] = c[2]
