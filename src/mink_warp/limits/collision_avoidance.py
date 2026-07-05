"""Collision avoidance hard limit (configuration-dependent ``G dq <= h``)."""

from __future__ import annotations

import itertools
from typing import Sequence

import mujoco
import numpy as np
import warp as wp

from ..configuration import Configuration
from ..kernels.collision import collision_broadphase, contact_jac_rows
from ..kernels.constrained import reset_ineq_block
from .limit import Limit

Geom = int | str
GeomSequence = Sequence[Geom]
CollisionPair = tuple[GeomSequence, GeomSequence]
CollisionPairs = Sequence[CollisionPair]

_BROADPHASE_MIN_PAIRS = 16


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
        # Host narrowphase collects only witness geometry (points, bodies,
        # normal, bound) per active contact — NOT the Jacobian. mj_kinematics is
        # enough for mj_geomDistance; the expensive per-contact mj_jac pair is
        # replaced by batched mujoco_warp.jac on device below.
        # Host narrowphase: geom distances write witness points straight into a
        # preallocated buffer; only light per-contact bookkeeping happens in the
        # Python loop (world / row / dist). Normals, bounds and signs are derived
        # vectorised afterwards, and every contact's Jacobian is built on device.
        data = self._host_data
        g1_of = self._pair_g1_np
        g2_of = self._pair_g2_np
        maxk = 0
        if candidate is not None:
            maxk = int(candidate[np.asarray(worlds)].sum()) if len(worlds) else 0
        else:
            maxk = nworld * m
        fromto = np.empty((max(maxk, 1), 6), dtype=np.float64)
        cw = np.empty(maxk, dtype=np.int32)
        crow = np.empty(maxk, dtype=np.int32)
        cdist = np.empty(maxk, dtype=np.float64)
        c = 0
        for w in worlds:
            data.qpos[:] = q_np[w]
            mujoco.mj_kinematics(model, data)
            indices = np.nonzero(candidate[w])[0] if candidate is not None else range(m)
            for idx in indices:
                dist = mujoco.mj_geomDistance(
                    model, data, int(g1_of[idx]), int(g2_of[idx]), distmax, fromto[c]
                )
                if abs(dist - distmax) < 1e-12:
                    continue
                cw[c] = w
                crow[c] = idx
                cdist[c] = dist
                c += 1

        fromto = fromto[:c]
        cw = cw[:c]
        crow = crow[:c]
        cdist = cdist[:c]
        # Vectorised: normal = normalize(p2 - p1), h and sign from the distance.
        p1 = fromto[:, :3]
        p2 = fromto[:, 3:]
        cn = p2 - p1
        nrm = np.linalg.norm(cn, axis=1, keepdims=True)
        np.divide(cn, nrm, out=cn, where=nrm > 0.0)
        cb1 = self._pair_b1_np[crow]
        cb2 = self._pair_b2_np[crow]
        ch = np.where(cdist > min_dist, gain * (cdist - min_dist) / dt + relaxation,
                      relaxation)
        csign = np.where(cdist >= 0.0, -1.0, 1.0)

        self._scatter_contacts(
            configuration, int(row_offset), G, h,
            c, cw, crow, p1, cb1, p2, cb2, cn, csign, ch,
        )

    def _scatter_contacts(
        self, configuration, row_offset, G, h,
        k, cw, crow, cp1, cb1, cp2, cb2, cn, csign, ch,
    ) -> None:
        """Reset the block, then build every active contact row on device.

        All ``k`` active contacts (across every world) are assembled in a single
        ``(k, nv)`` kernel launch: each contact's point Jacobians at the two
        witness points are evaluated inline from ``cdof`` / ``subtree_com`` — the
        exact ``mj_jac`` formula — so the serial host ``mj_jac`` calls disappear.
        """
        device = configuration.device
        nworld = configuration.nworld
        wm = configuration.wp_model
        wd = configuration.wp_data
        nv = configuration.nv
        with wp.ScopedDevice(device):
            wp.launch(reset_ineq_block, dim=(nworld, self.max_num_contacts),
                      inputs=[row_offset], outputs=[G, h])
            if not k:
                return
            cw_d = wp.array(np.ascontiguousarray(cw, dtype=np.int32), dtype=wp.int32)
            crow_d = wp.array(np.ascontiguousarray(crow, dtype=np.int32), dtype=wp.int32)
            cp1_d = wp.array(np.ascontiguousarray(cp1, dtype=np.float32), dtype=wp.vec3)
            cb1_d = wp.array(np.ascontiguousarray(cb1, dtype=np.int32), dtype=wp.int32)
            cp2_d = wp.array(np.ascontiguousarray(cp2, dtype=np.float32), dtype=wp.vec3)
            cb2_d = wp.array(np.ascontiguousarray(cb2, dtype=np.int32), dtype=wp.int32)
            cn_d = wp.array(np.ascontiguousarray(cn, dtype=np.float32), dtype=wp.vec3)
            csign_d = wp.array(np.ascontiguousarray(csign, dtype=np.float32), dtype=float)
            ch_d = wp.array(np.ascontiguousarray(ch, dtype=np.float32), dtype=float)
            wp.launch(
                contact_jac_rows, dim=(k, nv),
                inputs=[wm.body_rootid, wm.body_isdofancestor, wd.subtree_com,
                        wd.cdof, cw_d, crow_d, cp1_d, cb1_d, cp2_d, cb2_d, cn_d,
                        csign_d, ch_d, row_offset],
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
            self._pair_b1_np = np.zeros(0, dtype=np.int32)
            self._pair_b2_np = np.zeros(0, dtype=np.int32)
            self._pair_rsum_np = np.zeros(0, dtype=np.float32)
            self._bp_margin = self.collision_detection_distance
            return
        g1, g2 = pairs[:, 0], pairs[:, 1]
        self._pair_b1_np = model.geom_bodyid[g1].astype(np.int32)
        self._pair_b2_np = model.geom_bodyid[g2].astype(np.int32)
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

