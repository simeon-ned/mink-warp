"""Batched IK benchmark scenarios for mink-warp.

Mirrors mink/benchmarks/scenes.py, but every scene is *batched* (``nworld``
worlds solved at once) and restricted to the soft, unconstrained task stack
that mink-warp already supports **with identical math in mink** — so a scene
can be replayed through both libraries for the CPU-vs-GPU accuracy check
(``bench_parity.py``) as well as the throughput sweep (``bench_ik.py``).

A scene exposes ``setup_mw`` / ``update_mw`` (mink-warp, always) and, when
``parity=True``, ``setup_mink`` / ``update_mink`` (mink CPU oracle, one world).
``update_*`` moves the frame-task target for sim time ``t`` but never solves,
so the same trajectory drives every harness.

Add a scenario = one Scene entry with builder callables; nothing else changes.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import mujoco
import numpy as np

import mink_warp as mw

_HERE = Path(__file__).parent
_EXAMPLES = _HERE.parent / "examples"

#: Default IK integration timestep [s] (100 Hz).
DT = 0.01

State = dict

# Frame-task end-effector trajectory: a small circle in the world x-z plane.
_AMP = 0.08
_FREQ = 0.5


def _circle_targets(base: np.ndarray, t: float, phase: np.ndarray | float = 0.0) -> np.ndarray:
    """Offset a batch of ``wxyz_xyz`` poses along an x-z circle at time ``t``."""
    tg = base.copy()
    a = 2.0 * math.pi * _FREQ * t + phase
    tg[:, 4] += _AMP * np.cos(a)
    tg[:, 6] += _AMP * np.sin(a)
    return tg


# ---------------------------------------------------------------------------
# Panda: 7-DoF fixed-base arm, frame task + posture. Parity-safe.
# ---------------------------------------------------------------------------

_PANDA_XML = _EXAMPLES / "franka_emika_panda" / "mjx_scene.xml"
_PANDA_FRAME = ("attachment_site", "site")


def _panda_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(_PANDA_XML.as_posix())


def setup_panda_mw(nworld: int, device: str | None = None, seed: int = 0,
                   perturb: bool = False) -> State:
    model = _panda_model()
    cfg = mw.Configuration(model, nworld=nworld, device=device)
    cfg.update_from_keyframe("home")
    if perturb:
        rng = np.random.default_rng(seed)
        q = cfg.q.numpy().copy()
        q += rng.uniform(-0.05, 0.05, size=q.shape)
        cfg.update(q=q.astype(np.float32))

    frame = mw.FrameTask(*_PANDA_FRAME, position_cost=1.0, orientation_cost=1.0,
                         lm_damping=1e-3)
    posture = mw.PostureTask(model, cost=1e-2)
    frame.set_target_from_configuration(cfg)
    posture.set_target_from_configuration(cfg)
    base = cfg.get_transform_frame_to_world(*_PANDA_FRAME).numpy().copy()
    # Distinct per-world phase so worlds do different work (world 0 == mink).
    phase = np.arange(nworld) * (2.0 * math.pi / max(nworld, 1))

    solver = mw.IKSolver(cfg)
    return dict(configuration=cfg, tasks=[frame, posture], solver=solver,
                frame=frame, base=base, phase=phase, damping=1e-3, targets=base.copy())


def update_panda_mw(s: State, t: float) -> None:
    tg = _circle_targets(s["base"], t, s["phase"])
    s["targets"] = tg
    s["frame"].set_target(tg, configuration=s["configuration"])


def setup_panda_mink(seed: int = 0, perturb: bool = False) -> State:
    import mink
    model = _panda_model()
    cfg = mink.Configuration(model)
    frame = mink.FrameTask(frame_name=_PANDA_FRAME[0], frame_type=_PANDA_FRAME[1],
                           position_cost=1.0, orientation_cost=1.0, lm_damping=1e-3)
    posture = mink.PostureTask(model, cost=1e-2)
    cfg.update_from_keyframe("home")
    if perturb:
        rng = np.random.default_rng(seed)
        q = cfg.q.copy()
        q += rng.uniform(-0.05, 0.05, size=q.shape)
        cfg.update(q)
    frame.set_target_from_configuration(cfg)
    posture.set_target_from_configuration(cfg)
    return dict(configuration=cfg, tasks=[frame, posture], frame=frame, damping=1e-3)


def update_panda_mink(s: State, target_wxyz_xyz: np.ndarray) -> None:
    import mink
    s["frame"].set_target(mink.SE3(wxyz_xyz=np.asarray(target_wxyz_xyz, dtype=np.float64)))


# ---------------------------------------------------------------------------
# G1 humanoid: 49-DoF floating base, pelvis frame + posture + CoM. Throughput.
# ---------------------------------------------------------------------------

_G1_XML = _EXAMPLES / "unitree_g1" / "scene.xml"
_G1_FRAME = ("pelvis", "body")


def _g1_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(_G1_XML.as_posix())


def setup_g1_mw(nworld: int, device: str | None = None, seed: int = 0,
                perturb: bool = False) -> State:
    model = _g1_model()
    cfg = mw.Configuration(model, nworld=nworld, device=device)
    cfg.update_from_keyframe("stand")

    pelvis = mw.FrameTask(*_G1_FRAME, position_cost=[3.0, 3.0, 5.0],
                          orientation_cost=1.0, gain=0.7, lm_damping=1.0)
    posture = mw.PostureTask(model, cost=5e-2)
    com = mw.ComTask(cost=[10.0, 10.0, 10.0])
    pelvis.set_target_from_configuration(cfg)
    posture.set_target_from_configuration(cfg)
    com.set_target_from_configuration(cfg)
    base = cfg.get_transform_frame_to_world(*_G1_FRAME).numpy().copy()
    phase = np.arange(nworld) * (2.0 * math.pi / max(nworld, 1))

    solver = mw.IKSolver(cfg)
    return dict(configuration=cfg, tasks=[pelvis, posture, com], solver=solver,
                frame=pelvis, base=base, phase=phase, damping=1e-1, targets=base.copy())


def update_g1_mw(s: State, t: float) -> None:
    tg = s["base"].copy()
    # Gentle vertical squat of the pelvis frame.
    tg[:, 6] += 0.06 * np.sin(2.0 * math.pi * _FREQ * t + s["phase"])
    s["targets"] = tg
    s["frame"].set_target(tg, configuration=s["configuration"])


# ---------------------------------------------------------------------------
# Registry.
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Scene:
    key: str
    label: str
    nv: int
    parity: bool
    setup_mw: Callable[..., State]
    update_mw: Callable[[State, float], None]
    setup_mink: Callable[..., State] | None = None
    update_mink: Callable[[State, np.ndarray], None] | None = None


SCENES: dict[str, Scene] = {
    s.key: s
    for s in [
        Scene("panda", "Franka Panda (frame + posture)", 9, True,
              setup_panda_mw, update_panda_mw, setup_panda_mink, update_panda_mink),
        Scene("g1", "Unitree G1 (pelvis frame + posture + CoM)", 49, False,
              setup_g1_mw, update_g1_mw),
    ]
}
