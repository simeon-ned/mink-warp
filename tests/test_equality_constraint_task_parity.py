"""Parity tests for EqualityConstraintTask vs Mink (Cassie closed chain)."""

from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np
import pytest

mink = pytest.importorskip("mink")

import mink_warp as mw


def _cassie_model() -> mujoco.MjModel:
    xml = Path(__file__).resolve().parents[1] / "examples" / "agility_cassie" / "scene.xml"
    return mujoco.MjModel.from_xml_path(xml.as_posix())


def test_equality_zero_error_at_home():
    model = _cassie_model()
    cfg_mw = mw.Configuration(model, nworld=1)
    cfg_mk = mink.Configuration(model)
    cfg_mw.update_from_keyframe("home")
    cfg_mk.update_from_keyframe("home")

    task_mw = mw.EqualityConstraintTask(model=model, cost=1.0)
    task_mk = mink.EqualityConstraintTask(model=model, cost=1.0)

    e_mw = task_mw.compute_error(cfg_mw).numpy()[0]
    e_mk = task_mk.compute_error(cfg_mk)
    np.testing.assert_allclose(e_mw, e_mk, atol=1e-5)

    j_mw = task_mw.compute_jacobian(cfg_mw).numpy()[0]
    j_mk = task_mk.compute_jacobian(cfg_mk)
    np.testing.assert_allclose(j_mw, j_mk, atol=1e-4)
