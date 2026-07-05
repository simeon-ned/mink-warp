"""Collision avoidance hard limit (configuration-dependent ``G dq <= h``)."""

from __future__ import annotations

import itertools
from typing import Sequence

import mujoco
import numpy as np
import warp as wp

from ..configuration import Configuration
from ..kernels.collision import collision_world_broadphase
from ..kernels.constrained import scatter_ineq_block
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
        # Persistent device staging buffers for the host-built collision block +
        # broadphase prefilter, allocated on first scatter (keyed by device +
        # nworld).
        self._g_dev: wp.array | None = None
        self._h_dev: wp.array | None = None
        self._world_any: wp.array | None = None
        self._pair_g1_dev: wp.array | None = None
        self._pair_g2_dev: wp.array | None = None
        self._pair_rsum_dev: wp.array | None = None
        self._dev_key: tuple[str, int] | None = None
        self._init_broadphase(model)
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
        g_np = np.zeros((nworld, m, nv), dtype=np.float32)
        h_np = np.full((nworld, m), np.inf, dtype=np.float32)
        distmax = self.collision_detection_distance
        min_dist = self.minimum_distance_from_collisions
        gain = self.gain
        relaxation = self.bound_relaxation
        use_broadphase = self.broadphase and m >= self.broadphase_min_pairs

        # Device broadphase: in parallel over (world, pair), flag worlds with any
        # pair near enough to matter. Worlds flagged 0 provably have no active
        # row, so their host narrowphase is skipped. When the device test is
        # unavailable/disabled every world is processed (identical result).
        worlds = self._prefilter_worlds(configuration) if use_broadphase else None
        data = self._host_data
        fromto = np.empty(6)
        normal = np.empty(3)
        jac1 = np.empty((3, nv))
        jac2 = np.empty((3, nv))
        for w in worlds if worlds is not None else range(nworld):
            data.qpos[:] = q_np[w]
            mujoco.mj_fwdPosition(model, data)
            indices = self._broadphase_survivors(data) if use_broadphase else range(m)
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
                if dist > min_dist:
                    h_np[w, idx] = (gain * (dist - min_dist) / dt) + relaxation
                else:
                    h_np[w, idx] = relaxation
                sign = -1.0 if dist >= 0 else 1.0
                g_np[w, idx] = sign * row.astype(np.float32)

        # Upload only the collision block and scatter it into G/h at row_offset,
        # instead of downloading + re-uploading the whole padded QP buffer.
        with wp.ScopedDevice(device):
            self._g_dev.assign(g_np)
            self._h_dev.assign(h_np)
            wp.launch(
                scatter_ineq_block,
                dim=(nworld, m),
                inputs=[self._g_dev, self._h_dev, int(row_offset)],
                outputs=[G, h],
            )

    def _prefilter_worlds(self, configuration: Configuration) -> np.ndarray:
        """World indices with any monitored pair within the detection band.

        Runs :func:`collision_world_broadphase` on the batched ``geom_xpos`` and
        returns the surviving world indices (host narrowphase runs only on these).
        """
        nworld = configuration.nworld
        with wp.ScopedDevice(configuration.device):
            self._world_any.zero_()
            wp.launch(
                collision_world_broadphase,
                dim=(nworld, self.max_num_contacts),
                inputs=[
                    configuration.wp_data.geom_xpos,
                    self._pair_g1_dev,
                    self._pair_g2_dev,
                    self._pair_rsum_dev,
                    float(self._bp_margin),
                ],
                outputs=[self._world_any],
            )
        return np.nonzero(self._world_any.numpy())[0]

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
            self._g_dev = wp.zeros(
                (nworld, self.max_num_contacts, self.model.nv), dtype=float
            )
            self._h_dev = wp.zeros((nworld, self.max_num_contacts), dtype=float)
            self._world_any = wp.zeros(nworld, dtype=wp.int32)
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

    def _init_broadphase(self, model: mujoco.MjModel) -> None:
        pairs = np.array(self.geom_id_pairs, dtype=int).reshape(-1, 2)
        if pairs.size == 0:
            self._ss_idx = np.array([], dtype=int)
            self._pg_idx = np.array([], dtype=int)
            self._keep_idx = np.array([], dtype=int)
            return
        g1, g2 = pairs[:, 0], pairs[:, 1]
        rbound = model.geom_rbound
        is_plane = model.geom_type == mujoco.mjtGeom.mjGEOM_PLANE
        both_bounded = (rbound[g1] > 0.0) & (rbound[g2] > 0.0)
        plane1 = is_plane[g1] & (rbound[g2] > 0.0)
        plane2 = is_plane[g2] & (rbound[g1] > 0.0)
        plane_pair = (plane1 | plane2) & ~both_bounded
        self._ss_idx = np.where(both_bounded)[0]
        self._ss_g1 = g1[self._ss_idx]
        self._ss_g2 = g2[self._ss_idx]
        self._ss_rsum = rbound[g1[self._ss_idx]] + rbound[g2[self._ss_idx]]
        pg = np.where(plane_pair)[0]
        plane_is_g1 = plane1[pg]
        self._pg_idx = pg
        self._pg_plane = np.where(plane_is_g1, g1[pg], g2[pg])
        self._pg_other = np.where(plane_is_g1, g2[pg], g1[pg])
        self._pg_rother = rbound[self._pg_other]
        self._keep_idx = np.where(~both_bounded & ~plane_pair)[0]

    def _broadphase_survivors(self, data: mujoco.MjData) -> np.ndarray:
        margin = self.collision_detection_distance
        xpos = data.geom_xpos
        survivors = [self._keep_idx]
        if self._ss_idx.size:
            diff = xpos[self._ss_g1] - xpos[self._ss_g2]
            dist_sq = np.einsum("ij,ij->i", diff, diff)
            bound = self._ss_rsum + margin
            survivors.append(self._ss_idx[dist_sq <= bound * bound])
        if self._pg_idx.size:
            normal = data.geom_xmat[self._pg_plane][:, [2, 5, 8]]
            diff = xpos[self._pg_other] - xpos[self._pg_plane]
            signed_dist = np.einsum("ij,ij->i", normal, diff)
            survivors.append(self._pg_idx[signed_dist <= margin + self._pg_rother])
        return np.concatenate(survivors)
