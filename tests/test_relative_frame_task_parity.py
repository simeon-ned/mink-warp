"""Parity tests for RelativeFrameTask vs Mink."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

mink = pytest.importorskip("mink")

import mink_warp as mw


def _g1_model() -> mujoco.MjModel:
    xml = Path(__file__).resolve().parents[1] / "examples" / "unitree_g1" / "scene.xml"
    return mujoco.MjModel.from_xml_path(xml.as_posix())


def test_relative_frame_parity_g1():
    model = _g1_model()
    cfg_mw = mw.Configuration(model, nworld=1)
    cfg_mk = mink.Configuration(model)
    cfg_mw.update_from_keyframe("stand")
    cfg_mk.update_from_keyframe("stand")

    rel_mw = mw.RelativeFrameTask(
        "left_palm", "site",
        "torso_link", "body",
        position_cost=1.0, orientation_cost=0.5,
    )
    rel_mk = mink.RelativeFrameTask(
        frame_name="left_palm",
        frame_type="site",
        root_name="torso_link",
        root_type="body",
        position_cost=1.0,
        orientation_cost=0.5,
    )
    rel_mw.set_target_from_configuration(cfg_mw)
    rel_mk.set_target_from_configuration(cfg_mk)

    e_mw = rel_mw.compute_error(cfg_mw).numpy()[0]
    e_mk = rel_mk.compute_error(cfg_mk)
    np.testing.assert_allclose(e_mw, e_mk, atol=1e-4)

    j_mw = rel_mw.compute_jacobian(cfg_mw).numpy()[0]
    j_mk = rel_mk.compute_jacobian(cfg_mk)
    np.testing.assert_allclose(j_mw, j_mk, atol=1e-3)
