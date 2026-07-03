"""Lie-group operations match Mink."""

from __future__ import annotations

import numpy as np
import pytest

mink = pytest.importorskip("mink")

import mink_warp as mw


def test_so3_log_exp_roundtrip():
    omega = np.array([0.3, -0.2, 0.5])
    mw_q = mw.SO3.exp(omega)
    mk_q = mink.SO3.exp(omega)
    np.testing.assert_allclose(mw_q.wxyz, mk_q.wxyz, atol=1e-12)
    np.testing.assert_allclose(mw_q.log(), mk_q.log(), atol=1e-12)


def test_se3_minus_jlog_match_mink():
    np.random.seed(0)
    for _ in range(20):
        a = mw.SE3.sample_uniform()
        b = mw.SE3.sample_uniform()
        a_mk = mink.SE3(wxyz_xyz=a.wxyz_xyz.copy())
        b_mk = mink.SE3(wxyz_xyz=b.wxyz_xyz.copy())

        np.testing.assert_allclose(a.minus(b), a_mk.minus(b_mk), atol=1e-10)
        t_ab = a.inverse() @ b
        t_ab_mk = a_mk.inverse() @ b_mk
        np.testing.assert_allclose(t_ab.jlog(), t_ab_mk.jlog(), atol=1e-10)


def test_se3_rotation_adjoint_matches_mink():
    R = mw.SO3.exp(np.array([0.1, -0.4, 0.2]))
    T = mw.SE3.from_rotation(R.inverse())
    T_mk = mink.SE3.from_rotation(mink.SO3(wxyz=R.inverse().wxyz.copy()))
    np.testing.assert_allclose(T.adjoint(), T_mk.adjoint(), atol=1e-12)
