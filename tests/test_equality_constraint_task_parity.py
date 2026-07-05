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


def test_equality_batched_matches_mink_per_world():
    """Each world's mj_fwdPosition-built rows match mink's per-config rows."""
    model = _cassie_model()
    nworld = 5
    cfg_mw = mw.Configuration(model, nworld=nworld)
    cfg_mw.update_from_keyframe("home")
    q0 = cfg_mw.q.numpy()[0].copy()
    rng = np.random.default_rng(0)
    qs = (q0[None, :] + rng.uniform(-0.1, 0.1, size=(nworld, model.nq))).astype(np.float32)
    cfg_mw.update(q=qs)

    task_mw = mw.EqualityConstraintTask(model=model, cost=1.0)
    e_mw = task_mw.compute_error(cfg_mw).numpy()
    j_mw = task_mw.compute_jacobian(cfg_mw).numpy()

    cfg_mk = mink.Configuration(model)
    task_mk = mink.EqualityConstraintTask(model=model, cost=1.0)
    for w in range(nworld):
        cfg_mk.update(qs[w].astype(np.float64))
        np.testing.assert_allclose(e_mw[w], task_mk.compute_error(cfg_mk), atol=1e-5)
        np.testing.assert_allclose(j_mw[w], task_mk.compute_jacobian(cfg_mk), atol=1e-4)
