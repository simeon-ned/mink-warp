"""Device collision broadphase: parallel geom-pair proximity checks."""

from __future__ import annotations

import warp as wp


@wp.kernel
def collision_broadphase(
    geom_xpos: wp.array2d[wp.vec3],  # (nworld, ngeom) batched FK positions
    pair_g1: wp.array[int],  # (npair,)
    pair_g2: wp.array[int],  # (npair,)
    pair_rsum: wp.array[float],  # (npair,) rbound1 + rbound2, or < 0 => always near
    margin: float,
    candidate: wp.array2d[wp.int32],  # (nworld, npair) out, 1 if within band
    world_any: wp.array[int],  # (nworld,) out, MUST be pre-zeroed
):
    """Flag, in parallel over ``(world, pair)``, each pair within ``rsum + margin``.

    The pair distance tests that the host used to run in a serial Python loop
    (per world, then a numpy broadphase over every pair) now run on device
    straight off the batched forward-kinematics ``geom_xpos``. ``candidate[w, p]``
    tells the host exactly which pairs to hand to the exact narrowphase; a world
    with no candidate (``world_any[w] == 0``) skips host work entirely. ``margin``
    carries slack over the detection distance so the float32 device test is a
    conservative superset of the exact host test.
    """
    w, p = wp.tid()
    rs = pair_rsum[p]
    near = wp.int32(0)
    if rs < 0.0:
        # Plane / unbounded pair: no cheap sphere bound, always hand to host.
        near = wp.int32(1)
    else:
        d = geom_xpos[w, pair_g1[p]] - geom_xpos[w, pair_g2[p]]
        bound = rs + margin
        if wp.dot(d, d) <= bound * bound:
            near = wp.int32(1)
    candidate[w, p] = near
    if near == 1:
        world_any[w] = 1


@wp.kernel
def contact_jac_rows(
    # Model / data for the point Jacobian (mirrors mujoco_warp.support.jac_dof).
    body_rootid: wp.array[int],  # (nbody,)
    body_isdofancestor: wp.array2d[int],  # (nbody, nv)
    subtree_com: wp.array2d[wp.vec3],  # (nworld, nbody)
    cdof: wp.array2d[wp.spatial_vector],  # (nworld, nv)
    # Flat per-contact witness geometry (K contacts).
    cw: wp.array[int],  # world of contact k
    crow: wp.array[int],  # row within the collision block
    cp1: wp.array(dtype=wp.vec3),  # witness point on geom 1
    cb1: wp.array[int],  # body of geom 1
    cp2: wp.array(dtype=wp.vec3),  # witness point on geom 2
    cb2: wp.array[int],  # body of geom 2
    cn: wp.array(dtype=wp.vec3),  # contact normal (p1 -> p2)
    csign: wp.array[float],  # +/-1 from the signed distance
    chval: wp.array[float],  # bound h
    row_offset: int,
    G: wp.array3d[float],
    h: wp.array2d[float],
):
    """Build every active contact row ``sign * nᵀ(J2 - J1)`` in one launch.

    One thread per ``(contact, dof)``: the point Jacobian column of each witness
    is evaluated inline from ``cdof`` / ``subtree_com`` (the exact
    ``mujoco_warp.jac`` / ``mj_jac`` formula), so all contacts across all worlds
    assemble in parallel with no per-contact host ``mj_jac`` and no per-world
    round loop.
    """
    k, dofid = wp.tid()
    w = cw[k]
    cd = cdof[w, dofid]
    cang = wp.spatial_top(cd)
    clin = wp.spatial_bottom(cd)
    n = cn[k]

    b1 = cb1[k]
    jp1 = wp.vec3(0.0, 0.0, 0.0)
    if body_isdofancestor[b1, dofid] != 0:
        off1 = cp1[k] - subtree_com[w, body_rootid[b1]]
        jp1 = clin + wp.cross(cang, off1)

    b2 = cb2[k]
    jp2 = wp.vec3(0.0, 0.0, 0.0)
    if body_isdofancestor[b2, dofid] != 0:
        off2 = cp2[k] - subtree_com[w, body_rootid[b2]]
        jp2 = clin + wp.cross(cang, off2)

    r = row_offset + crow[k]
    G[w, r, dofid] = csign[k] * wp.dot(n, jp2 - jp1)
    if dofid == 0:
        h[w, r] = chval[k]
