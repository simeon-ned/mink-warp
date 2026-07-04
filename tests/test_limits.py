"""Batched limit box assembly matches mink's ConfigurationLimit / VelocityLimit.

These run on whatever warp device is available (the box kernels are plain, not
tiled) so they exercise the parity of the bounds independently of the GPU-only
ADMM solve.
"""

from __future__ import annotations

import numpy as np
import pytest
import warp as wp

mink = pytest.importorskip("mink")

import mink_warp as mw  # noqa: E402
from mink_warp.kernels.constrained import BOX_INF, init_box  # noqa: E402
from mink_warp.limits import ConfigurationLimit, VelocityLimit  # noqa: E402


def _fresh_box(nworld, nv, device):
    with wp.ScopedDevice(device):
        lo = wp.zeros((nworld, nv), dtype=float)
        hi = wp.zeros((nworld, nv), dtype=float)
        wp.launch(init_box, dim=(nworld, nv), outputs=[lo, hi])
    return lo, hi


def test_config_limit_box_matches_mink(arm_model):
    q = np.array([1.5, -2.0])
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    lo, hi = _fresh_box(1, cfg.nv, cfg.device)

    ConfigurationLimit(arm_model, gain=0.95).apply_box(cfg, 0.02, lo, hi)
    lo_np, hi_np = lo.numpy()[0], hi.numpy()[0]

    mk_cfg = mink.Configuration(arm_model, q=q)
    cn = mink.ConfigurationLimit(arm_model, gain=0.95).compute_qp_inequalities(
        mk_cfg, 0.02
    )
    # G = [P; -P], h = [p_max; p_min]; indices are the limited dofs (here 0, 1).
    nb = cn.h.shape[0] // 2
    p_max, p_min = cn.h[:nb], cn.h[nb:]
    np.testing.assert_allclose(hi_np, p_max, atol=1e-5)
    np.testing.assert_allclose(lo_np, -p_min, atol=1e-5)


def test_config_limit_box_near_bound_is_tight(arm_model):
    # q close to joint1 upper limit -> tiny positive room, large negative room.
    q = np.array([3.10, 0.0])  # upper = 3.14
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    lo, hi = _fresh_box(1, cfg.nv, cfg.device)
    ConfigurationLimit(arm_model, gain=0.95).apply_box(cfg, 0.02, lo, hi)
    hi_np, lo_np = hi.numpy()[0], lo.numpy()[0]
    assert hi_np[0] == pytest.approx(0.95 * (3.14 - 3.10), abs=1e-5)
    assert lo_np[0] == pytest.approx(-0.95 * (3.10 - (-3.14)), abs=1e-4)
    assert hi_np[0] < 0.05  # very little upward room


def test_velocity_limit_box_matches_mink(arm_model):
    q = np.array([0.4, -0.7])
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    lo, hi = _fresh_box(1, cfg.nv, cfg.device)

    dt = 0.02
    VelocityLimit(arm_model, {"joint1": 1.5}).apply_box(cfg, dt, lo, hi)
    lo_np, hi_np = lo.numpy()[0], hi.numpy()[0]

    # joint1 (dof 0) boxed at +-dt*vmax; joint2 (dof 1) unconstrained.
    assert hi_np[0] == pytest.approx(dt * 1.5, abs=1e-6)
    assert lo_np[0] == pytest.approx(-dt * 1.5, abs=1e-6)
    assert hi_np[1] == pytest.approx(float(BOX_INF))
    assert lo_np[1] == pytest.approx(-float(BOX_INF))


def test_limits_compose_by_tightening(arm_model):
    q = np.array([3.10, -0.7])
    cfg = mw.Configuration(arm_model, q=q, nworld=1)
    lo, hi = _fresh_box(1, cfg.nv, cfg.device)
    ConfigurationLimit(arm_model, gain=0.95).apply_box(cfg, 0.02, lo, hi)
    VelocityLimit(arm_model, {"joint1": 0.1}).apply_box(cfg, 0.02, lo, hi)
    hi_np = hi.numpy()[0]
    # joint1 upper is min(config room 0.038, velocity room dt*0.1=0.002).
    assert hi_np[0] == pytest.approx(0.002, abs=1e-5)


_BALL_XML = """
<mujoco>
  <worldbody>
    <body pos="0 0 0.1">
      <joint name="ball1" type="ball" limited="true" range="0 0.5"/>
      <geom type="sphere" size="0.05"/>
    </body>
  </worldbody>
</mujoco>
"""


def test_configuration_limit_rejects_limited_ball_joint():
    import mujoco

    # mink enforces limited ball joints; we don't yet, so fail loud rather
    # than silently drop a hard limit.
    model = mujoco.MjModel.from_xml_string(_BALL_XML)
    with pytest.raises(NotImplementedError):
        ConfigurationLimit(model)
