"""Device collision broadphase: parallel geom-pair proximity checks."""

from __future__ import annotations

import warp as wp


@wp.kernel
def collision_world_broadphase(
    geom_xpos: wp.array2d[wp.vec3],  # (nworld, ngeom) batched FK positions
    pair_g1: wp.array[int],  # (npair,)
    pair_g2: wp.array[int],  # (npair,)
    pair_rsum: wp.array[float],  # (npair,) rbound1 + rbound2, or < 0 => always near
    margin: float,
    world_any: wp.array[int],  # (nworld,) out, MUST be pre-zeroed
):
    """Flag each world that has any monitored geom pair within ``rsum + margin``.

    One thread per ``(world, pair)`` — the pair distance tests that the host used
    to run in a serial Python loop now run in parallel on device, straight off
    the batched forward-kinematics ``geom_xpos``. A world flagged 0 provably has
    no pair inside the detection band, so its expensive host narrowphase
    (``mj_fwdPosition`` + ``mj_geomDistance`` + ``mj_jac``) can be skipped
    entirely. ``margin`` carries a slack over the detection distance so the
    float32 device test is a conservative superset of the exact host test.
    """
    w, p = wp.tid()
    rs = pair_rsum[p]
    if rs < 0.0:
        # Plane / unbounded pair: no cheap sphere bound, always hand to host.
        world_any[w] = 1
        return
    d = geom_xpos[w, pair_g1[p]] - geom_xpos[w, pair_g2[p]]
    bound = rs + margin
    if wp.dot(d, d) <= bound * bound:
        world_any[w] = 1
