"""CollisionAvoidanceLimit geom pairing and inequality parity vs Mink."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest
import warp as wp

mink = pytest.importorskip("mink")

import mink_warp as mw
from mink_warp.limits import CollisionAvoidanceLimit


_SPHERE_XML = """
<mujoco>
  <worldbody>
    <body>
      <joint type="ball" name="ball"/>
      <geom name="g1" type="sphere" size=".1" mass=".1"/>
      <body>
        <joint type="hinge" name="hinge" range="0 1.57"/>
        <geom name="g2" type="sphere" size=".1" mass=".1"/>
      </body>
    </body>
    <body>
      <joint type="hinge" name="hinge2" range="0 1.57"/>
      <geom name="g3" type="sphere" size=".1" mass=".1"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_geom_pairs_deduplication():
    model = mujoco.MjModel.from_xml_string(_SPHERE_XML)
    limit = CollisionAvoidanceLimit(model=model, geom_pairs=[(["g1"], ["g3"])])
    assert limit.geom_id_pairs == [(0, 2)]

    limit = CollisionAvoidanceLimit(
        model=model, geom_pairs=[(["g1"], ["g3"]), (["g1"], ["g3"])]
    )
    assert limit.geom_id_pairs == [(0, 2)]


def test_inequalities_match_mink_on_ur5e_home():
    xml = (
        Path(__file__).resolve().parents[1]
        / "examples"
        / "universal_robots_ur5e"
        / "scene.xml"
    )
    model = mujoco.MjModel.from_xml_path(xml.as_posix())
    collision_pairs = [(["wrist_3_link"], ["floor", "wall"])]
    dt = 0.02

    cfg_mw = mw.Configuration(model, nworld=1)
    cfg_mk = mink.Configuration(model)
    cfg_mw.update_from_keyframe("home")
    cfg_mk.update_from_keyframe("home")

    limit_mw = CollisionAvoidanceLimit(model, geom_pairs=collision_pairs)
    limit_mk = mink.CollisionAvoidanceLimit(model=model, geom_pairs=collision_pairs)

    cn = limit_mk.compute_qp_inequalities(cfg_mk, dt)
    nrows = cn.G.shape[0]
    nv = model.nv
    with wp.ScopedDevice(cfg_mw.device):
        G = wp.zeros((1, nrows, nv), dtype=float)
        h = wp.zeros((1, nrows), dtype=float)
    limit_mw.scatter_inequalities(cfg_mw, dt, 0, G, h)
    g_mw = G.numpy()[0]
    h_mw = h.numpy()[0]

    active = np.isfinite(cn.h) & (np.abs(cn.h) < 1e20)
    for i in np.where(active)[0]:
        np.testing.assert_allclose(g_mw[i], cn.G[i], atol=1e-4)
        np.testing.assert_allclose(h_mw[i], cn.h[i], atol=1e-4)


# Two spheres, each on its own slide joint: bringing them together / apart lets
# the device broadphase prefilter both skip (far) and keep (near) worlds.
_TWO_SPHERE_XML = """
<mujoco>
  <worldbody>
    <body name="a">
      <joint type="slide" axis="1 0 0" name="ja" range="-3 3"/>
      <geom name="ga" type="sphere" size="0.1"/>
    </body>
    <body name="b" pos="0.3 0 0">
      <joint type="slide" axis="1 0 0" name="jb" range="-3 3"/>
      <geom name="gb" type="sphere" size="0.1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _scatter(limit, cfg, dt=0.02):
    with wp.ScopedDevice(cfg.device):
        G = wp.zeros((cfg.nworld, limit.n_inequalities, cfg.nv), dtype=float)
        h = wp.zeros((cfg.nworld, limit.n_inequalities), dtype=float)
    limit.scatter_inequalities(cfg, dt, 0, G, h)
    return G.numpy(), h.numpy()


def test_device_prefilter_is_output_invariant():
    """The broadphase prefilter is a perf optimization: enabling it must not
    change any G/h row versus brute-force (every world, every pair)."""
    model = mujoco.MjModel.from_xml_string(_TWO_SPHERE_XML)
    nworld = 48
    cfg = mw.Configuration(model, nworld=nworld)
    rng = np.random.default_rng(0)
    # geom centres are 0.3 + qb - qa apart; the pair is near (active) when that
    # is < ~0.25 and far otherwise. Guarantee some near worlds (qa≈0.3, qb≈0),
    # the rest random-far -> the prefilter keeps some, skips most.
    q = rng.uniform(-2.0, 2.0, size=(nworld, model.nq)).astype(np.float32)
    q[:8, 0] = 0.3
    q[:8, 1] = 0.0
    cfg.update(q=q)

    limit = CollisionAvoidanceLimit(model, geom_pairs=[(["ga"], ["gb"])],
                                    collision_detection_distance=0.05)
    limit.broadphase_min_pairs = 1  # force prefilter to engage with a single pair

    limit.broadphase = True
    g_pf, h_pf = _scatter(limit, cfg)
    limit.broadphase = False
    g_bf, h_bf = _scatter(limit, cfg)

    np.testing.assert_array_equal(g_pf, g_bf)
    np.testing.assert_array_equal(np.where(np.isfinite(h_pf), h_pf, -1.0),
                                  np.where(np.isfinite(h_bf), h_bf, -1.0))
    # Sanity: the mix really does exercise both branches (some active rows, and
    # the prefilter really does drop some worlds).
    assert np.isfinite(h_bf).any()
    survivors, candidate = limit._prefilter(cfg)
    assert 0 < len(survivors) < nworld
    # every surviving world has a candidate pair; skipped worlds have none
    assert candidate.shape == (nworld, limit.n_inequalities)
    assert np.all(candidate[survivors].any(axis=1))


def test_scatter_ineq_block_places_at_offset():
    from mink_warp.kernels.constrained import scatter_ineq_block

    nworld, m, nv, off, total = 3, 2, 4, 5, 9
    rng = np.random.default_rng(1)
    g_src = rng.standard_normal((nworld, m, nv)).astype(np.float32)
    h_src = rng.standard_normal((nworld, m)).astype(np.float32)
    with wp.ScopedDevice(None):
        gsd = wp.array(g_src, dtype=float)
        hsd = wp.array(h_src, dtype=float)
        G = wp.zeros((nworld, total, nv), dtype=float)
        h = wp.zeros((nworld, total), dtype=float)
        wp.launch(scatter_ineq_block, dim=(nworld, m),
                  inputs=[gsd, hsd, off], outputs=[G, h])
    Gn, hn = G.numpy(), h.numpy()
    np.testing.assert_allclose(Gn[:, off:off + m, :], g_src, atol=0)
    np.testing.assert_allclose(hn[:, off:off + m], h_src, atol=0)
    # Rows outside the block are untouched (still zero).
    assert np.all(Gn[:, :off, :] == 0) and np.all(Gn[:, off + m:, :] == 0)
