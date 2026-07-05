"""Equality constraint task (host MuJoCo constraint rows, batched upload)."""

from __future__ import annotations

import logging
from typing import Sequence

import mujoco
import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..constants import constraint_width
from ..exceptions import InvalidConstraint, TaskDefinitionError
from .task import Task


def _dense_efc_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    if mujoco.mj_isSparse(model):
        efc_j = np.empty((data.nefc, model.nv))
        mujoco.mju_sparse2dense(
            efc_j,
            data.efc_J,
            data.efc_J_rownnz,
            data.efc_J_rowadr,
            data.efc_J_colind,
        )
        return efc_j
    # View onto the live efc_J buffer; the caller's row mask (fancy index) makes
    # the copy, so a full-matrix .copy() here would be redundant work.
    return data.efc_J.reshape((data.nefc, model.nv))


class EqualityConstraintTask(Task):
    """Regulate MuJoCo equality constraints (loop joints, welds, etc.).

    Uses host ``mj_forward`` + ``efc_pos`` / ``efc_J`` per world (MuJoCo Warp
    does not yet expose batched equality rows). Suitable for closed-chain
    mechanisms at moderate ``nworld``.
    """

    supports_cuda_graph = False

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: npt.ArrayLike,
        equalities: Sequence[int | str] | None = None,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        self._logger = logging.getLogger(__package__)
        self.model = model
        self._eq_ids = self._resolve_equality_ids(model, equalities)
        self._eq_types = model.eq_type[self._eq_ids].copy()
        self._host_data = mujoco.MjData(model)
        self._mask_cache: np.ndarray | None = None

        dim = int(sum(constraint_width(int(t)) for t in self._eq_types))
        super().__init__(cost=np.zeros(dim), gain=gain, lm_damping=lm_damping)
        self.k = dim
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        cost = np.atleast_1d(np.asarray(cost, dtype=np.float64))
        neq = len(self._eq_ids)
        if cost.ndim != 1 or cost.shape[0] not in (1, neq):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be shape (1,) or ({neq},); "
                f"got {cost.shape}."
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost must be >= 0")
        per_eq = (
            np.full((neq,), cost[0], dtype=np.float64)
            if cost.shape[0] == 1
            else cost.copy()
        )
        repeats = [constraint_width(int(t)) for t in self._eq_types]
        self.cost = np.repeat(per_eq, repeats)
        self._cost_dev = None

    def _resolve_equality_ids(
        self, model: mujoco.MjModel, equalities: Sequence[int | str] | None
    ) -> np.ndarray:
        eq_ids: list[int] = []
        if equalities is not None:
            for eq_id_or_name in equalities:
                if isinstance(eq_id_or_name, str):
                    eq_id = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_id_or_name
                    )
                    if eq_id == -1:
                        raise InvalidConstraint(
                            f"Equality constraint '{eq_id_or_name}' not found."
                        )
                else:
                    eq_id = int(eq_id_or_name)
                    if eq_id < 0 or eq_id >= model.neq:
                        raise InvalidConstraint(
                            f"Equality constraint index {eq_id} out of range "
                            f"[0, {model.neq})."
                        )
                if not model.eq_active0[eq_id]:
                    raise InvalidConstraint(
                        f"Equality constraint {eq_id} is not active at the "
                        "initial configuration."
                    )
                eq_ids.append(eq_id)
            if len(eq_ids) != len(set(eq_ids)):
                raise TaskDefinitionError(
                    f"Duplicate equality constraint IDs provided: {eq_ids}."
                )
        else:
            eq_ids = list(range(model.neq))
            self._logger.info("Regulating %d equality constraints", len(eq_ids))
        if len(eq_ids) == 0:
            raise TaskDefinitionError(
                f"{self.__class__.__name__} found no equality constraints in this model."
            )
        return np.asarray(eq_ids, dtype=np.int32)

    def _equality_mask(self, data: mujoco.MjData) -> np.ndarray:
        return (data.efc_type == mujoco.mjtConstraint.mjCNSTR_EQUALITY) & np.isin(
            data.efc_id, self._eq_ids
        )

    def _eval(self, configuration: Configuration) -> None:
        self._ensure_buffers(configuration)
        assert self._error is not None
        assert self._jacobian is not None

        model = self.model
        nworld = configuration.nworld
        nv = configuration.nv
        q_np = configuration.q.numpy()
        err_np = np.zeros((nworld, self.k), dtype=np.float32)
        jac_np = np.zeros((nworld, self.k, nv), dtype=np.float32)
        data = self._host_data

        for w in range(nworld):
            data.qpos[:] = q_np[w]
            # Equality rows (efc_pos / efc_J) are position-only, so the position
            # pipeline (kinematics -> ... -> makeConstraint) is sufficient; the
            # velocity / actuation / acceleration stages of mj_forward are not
            # (~1.7x cheaper here, identical efc_pos / efc_J).
            mujoco.mj_fwdPosition(model, data)
            mask = self._equality_mask(data)
            if not np.any(mask):
                continue
            rows = data.efc_pos[mask].astype(np.float32)
            j_rows = _dense_efc_jacobian(model, data)[mask].astype(np.float32)
            if rows.shape[0] != self.k:
                raise RuntimeError(
                    f"Active equality rows {rows.shape[0]} != task dim {self.k}. "
                    "Some constraints may have deactivated."
                )
            err_np[w] = rows
            jac_np[w] = j_rows

        with wp.ScopedDevice(configuration.device):
            self._error.assign(err_np)
            self._jacobian.assign(jac_np)
