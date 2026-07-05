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
