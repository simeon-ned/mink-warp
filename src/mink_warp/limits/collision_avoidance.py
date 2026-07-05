"""Collision avoidance hard limit (configuration-dependent ``G dq <= h``)."""

from __future__ import annotations

import itertools
from typing import Sequence

import mujoco
import numpy as np
import warp as wp

from ..configuration import Configuration
from ..kernels.collision import collision_broadphase
from ..kernels.constrained import reset_ineq_block, scatter_ineq_active
from .limit import Limit

Geom = int | str
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]

_BROADPHASE_MIN_PAIRS = 16


def _compute_contact_normal_jacobian(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    geom1_id: int,
    geom2_id: int,
    fromto: np.ndarray,
    normal: np.ndarray,
    jac1: np.ndarray,
    jac2: np.ndarray,
) -> np.ndarray:
    normal[:] = fromto[3:] - fromto[:3]
    mujoco.mju_normalize3(normal)
    geom_bodyid = model.geom_bodyid
    mujoco.mj_jac(model, data, jac2, None, fromto[3:], geom_bodyid[geom2_id])
    mujoco.mj_jac(model, data, jac1, None, fromto[:3], geom_bodyid[geom1_id])
    jac2 -= jac1
    return normal @ jac2


def _is_welded_together(model: mujoco.MjModel, geom_id1: int, geom_id2: int) -> bool:
    body1 = model.geom_bodyid[geom_id1]
    body2 = model.geom_bodyid[geom_id2]
    return model.body_weldid[body1] == model.body_weldid[body2]


def _are_geom_bodies_parent_child(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    body_id1 = model.geom_bodyid[geom_id1]
    body_id2 = model.geom_bodyid[geom_id2]
    weld1 = model.body_weldid[body_id1]
    weld2 = model.body_weldid[body_id2]
    parent1 = model.body_weldid[model.body_parentid[weld1]]
    parent2 = model.body_weldid[model.body_parentid[weld2]]
    return weld1 == parent2 or weld2 == parent1


def _passes_contype_conaffinity(
    model: mujoco.MjModel, geom_id1: int, geom_id2: int
) -> bool:
    return bool(model.geom_contype[geom_id1] & model.geom_conaffinity[geom_id2]) or bool(
        model.geom_contype[geom_id2] & model.geom_conaffinity[geom_id1]
    )


class CollisionAvoidanceLimit(Limit):
    """Normal-velocity limit between geom pairs (Mink-compatible)."""

    box_capable = False
    supports_cuda_graph = False

    def __init__(
        self,
        model: mujoco.MjModel,
        geom_pairs: CollisionPairs,
        gain: float = 0.85,
        minimum_distance_from_collisions: float = 0.005,
        collision_detection_distance: float = 0.01,
        bound_relaxation: float = 0.0,
        broadphase: bool = True,
    ):
        self.model = model
        self.gain = float(gain)
        self.minimum_distance_from_collisions = float(minimum_distance_from_collisions)
        self.collision_detection_distance = float(collision_detection_distance)
        self.bound_relaxation = float(bound_relaxation)
        self.broadphase = broadphase
        self.broadphase_min_pairs = _BROADPHASE_MIN_PAIRS
        self.geom_id_pairs = self._construct_geom_id_pairs(geom_pairs)
        self.max_num_contacts = len(self.geom_id_pairs)
        self.n_inequalities = self.max_num_contacts
        self._host_data = mujoco.MjData(model)
        # Persistent device buffers for the broadphase prefilter, allocated on
        # first scatter (keyed by device + nworld).
        self._world_any: wp.array | None = None
        self._candidate: wp.array | None = None
        self._pair_g1_dev: wp.array | None = None
        self._pair_g2_dev: wp.array | None = None
        self._pair_rsum_dev: wp.array | None = None
        self._dev_key: tuple[str, int] | None = None
        self._init_device_broadphase(model)

    def scatter_inequalities(
        self,
        configuration: Configuration,
        dt: float,
        row_offset: int,
        G: wp.array,
        h: wp.array,
    ) -> None:
        if self.max_num_contacts == 0:
            return
        model = self.model
        nworld = configuration.nworld
        nv = configuration.nv
        m = self.max_num_contacts
        device = configuration.device
        self._ensure_dev_stage(device, nworld)

        q_np = configuration.q.numpy()
        distmax = self.collision_detection_distance
        min_dist = self.minimum_distance_from_collisions
        gain = self.gain
        relaxation = self.bound_relaxation
        use_broadphase = self.broadphase and m >= self.broadphase_min_pairs

        # Device broadphase: in parallel over (world, pair), flag which pairs are
        # near enough to matter. A world with no candidate pair is skipped, and
        # the candidate mask replaces the per-world host numpy broadphase (which
        # cost more than the FK). When disabled, every world/pair is processed.
        if use_broadphase:
            worlds, candidate = self._prefilter(configuration)
        else:
            worlds, candidate = range(nworld), None
        data = self._host_data
        fromto = np.empty(6)
        normal = np.empty(3)
        jac1 = np.empty((3, nv))
        jac2 = np.empty((3, nv))
        # Collect only the ACTIVE (near-collision) rows; the rest of the block
        # stays inert, so per step only these few rows (not the whole padded
        # block) cross host -> device.
        aw: list[int] = []
        aidx: list[int] = []
        ag: list[np.ndarray] = []
        ah: list[float] = []
        for w in worlds:
            data.qpos[:] = q_np[w]
            # Collision rows need only geom poses (mj_kinematics) and the dof
            # Jacobian bases (mj_comPos -> cdof); the rest of mj_fwdPosition
            # (crb, factorM, constraints) is unused here — ~3.4x cheaper, with
            # bit-identical mj_geomDistance / mj_jac output.
            mujoco.mj_kinematics(model, data)
            mujoco.mj_comPos(model, data)
            indices = np.nonzero(candidate[w])[0] if candidate is not None else range(m)
            for idx in indices:
                geom1_id, geom2_id = self.geom_id_pairs[idx]
                dist = mujoco.mj_geomDistance(
                    model, data, geom1_id, geom2_id, distmax, fromto
                )
                if abs(dist - distmax) < 1e-12:
                    continue
                row = _compute_contact_normal_jacobian(
                    model, data, geom1_id, geom2_id, fromto, normal, jac1, jac2
                )
                h_val = (
                    (gain * (dist - min_dist) / dt) + relaxation
                    if dist > min_dist
                    else relaxation
                )
                sign = -1.0 if dist >= 0 else 1.0
                aw.append(w)
                aidx.append(int(idx))
                ag.append(sign * row)
                ah.append(h_val)

        # Reset this limit's block to inert (0, +inf) on device, then scatter only
        # the K active rows (K*(nv+1) floats) — no full (nworld, m, nv) upload.
        with wp.ScopedDevice(device):
            wp.launch(
                reset_ineq_block, dim=(nworld, m),
                inputs=[int(row_offset)], outputs=[G, h],
            )
            k = len(aw)
            if k:
                aw_d = wp.array(np.asarray(aw, dtype=np.int32), dtype=wp.int32)
                aidx_d = wp.array(np.asarray(aidx, dtype=np.int32), dtype=wp.int32)
                ag_d = wp.array(np.asarray(ag, dtype=np.float32), dtype=float)
                ah_d = wp.array(np.asarray(ah, dtype=np.float32), dtype=float)
                wp.launch(
                    scatter_ineq_active, dim=k,
                    inputs=[aw_d, aidx_d, ag_d, ah_d, int(row_offset)],
                    outputs=[G, h],
                )

    def _prefilter(self, configuration: Configuration):
        """Device broadphase over the batched ``geom_xpos``.

        Returns ``(worlds, candidate)`` where ``worlds`` are the world indices
        with at least one near pair (host narrowphase runs only on these) and
        ``candidate`` is the ``(nworld, npair)`` mask of near pairs (replaces the
        per-world host broadphase). The float32 test with a slack margin is a
        conservative superset of the exact host filter, so the resulting rows are
        identical to processing every world / pair.
        """
        nworld = configuration.nworld
        with wp.ScopedDevice(configuration.device):
            self._world_any.zero_()
            wp.launch(
                collision_broadphase,
                dim=(nworld, self.max_num_contacts),
                inputs=[
                    configuration.wp_data.geom_xpos,
                    self._pair_g1_dev,
                    self._pair_g2_dev,
                    self._pair_rsum_dev,
                    float(self._bp_margin),
                ],
                outputs=[self._candidate, self._world_any],
            )
            candidate = self._candidate.numpy()
            worlds = np.nonzero(self._world_any.numpy())[0]
        return worlds, candidate

    def _init_device_broadphase(self, model: mujoco.MjModel) -> None:
        """Precompute per-pair geom ids + bounding-sphere sums for the device test."""
        pairs = np.array(self.geom_id_pairs, dtype=np.int32).reshape(-1, 2)
        if pairs.size == 0:
            self._pair_g1_np = np.zeros(0, dtype=np.int32)
            self._pair_g2_np = np.zeros(0, dtype=np.int32)
            self._pair_rsum_np = np.zeros(0, dtype=np.float32)
            self._bp_margin = self.collision_detection_distance
            return
        g1, g2 = pairs[:, 0], pairs[:, 1]
        rbound = model.geom_rbound
        both_bounded = (rbound[g1] > 0.0) & (rbound[g2] > 0.0)
        # rsum < 0 marks plane / unbounded pairs -> always handed to host.
        rsum = np.where(both_bounded, rbound[g1] + rbound[g2], -1.0)
        self._pair_g1_np = g1.copy()
        self._pair_g2_np = g2.copy()
        self._pair_rsum_np = rsum.astype(np.float32)
        # Slack keeps the float32 device test a conservative superset of the
        # exact float64 host sphere test (host margin == detection distance).
        self._bp_margin = self.collision_detection_distance + 0.01

    def _ensure_dev_stage(self, device: str, nworld: int) -> None:
        key = (device, nworld)
        if self._dev_key == key:
            return
        with wp.ScopedDevice(device):
            self._world_any = wp.zeros(nworld, dtype=wp.int32)
            self._candidate = wp.zeros(
                (nworld, self.max_num_contacts), dtype=wp.int32
            )
            self._pair_g1_dev = wp.array(self._pair_g1_np, dtype=wp.int32)
            self._pair_g2_dev = wp.array(self._pair_g2_np, dtype=wp.int32)
            self._pair_rsum_dev = wp.array(self._pair_rsum_np, dtype=float)
        self._dev_key = key

    def _homogenize_geom_id_list(self, geom_list: GeomSequence) -> list[int]:
        out: list[int] = []
        for g in geom_list:
            if isinstance(g, int):
                out.append(g)
            else:
                out.append(self.model.geom(g).id)
        return out

    def _collision_pairs_to_geom_id_pairs(self, collision_pairs: CollisionPairs):
        geom_id_pairs = []
        for collision_pair in collision_pairs:
            id_pair_a = self._homogenize_geom_id_list(collision_pair[0])
            id_pair_b = self._homogenize_geom_id_list(collision_pair[1])
            geom_id_pairs.append((list(set(id_pair_a)), list(set(id_pair_b))))
        return geom_id_pairs

    def _construct_geom_id_pairs(self, geom_pairs: CollisionPairs):
        geom_id_pairs = []
        for id_pair in self._collision_pairs_to_geom_id_pairs(geom_pairs):
            for geom_a, geom_b in itertools.product(*id_pair):
                if not _is_welded_together(self.model, geom_a, geom_b):
                    if not _are_geom_bodies_parent_child(self.model, geom_a, geom_b):
                        if _passes_contype_conaffinity(self.model, geom_a, geom_b):
                            geom_id_pairs.append(
                                (min(geom_a, geom_b), max(geom_a, geom_b))
                            )
        return list(set(geom_id_pairs))

