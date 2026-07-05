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
