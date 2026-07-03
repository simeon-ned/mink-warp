"""Posture task (device-side)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TaskDefinitionError
from ..kernels.posture import (
    broadcast_q,
    posture_error,
    posture_error_joints,
    posture_jacobian_eye,
    zero_free_joint_rows,
)
from ..utils import get_freejoint_dims
from .task import TargetedTask


class PostureTask(TargetedTask):
    """Regulate configuration toward a target posture.

    Uses device ``mj_differentiatePos`` for free/ball/hinge/slide joints.
    Free-joint dofs are zeroed in the residual (Mink behavior).
    """

    def __init__(
        self,
        model,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros(model.nv), gain=gain, lm_damping=lm_damping)
        self.k = model.nv
        self.target_width = model.nq
        self.nq = model.nq
        self.model = model
        _, v_ids = get_freejoint_dims(model)
        self._v_ids_np = np.asarray(v_ids, dtype=np.int32) if v_ids else None
        self._v_ids: wp.array | None = None
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        cost = np.atleast_1d(np.asarray(cost, dtype=np.float64))
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"cost must be shape (1,) or ({self.k},), got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError("cost must be >= 0")
        self.cost[: self.k] = cost
        self._cost_dev = None

    def set_target(
        self,
        target_q: wp.array | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        if not isinstance(target_q, wp.array):
            arr = np.asarray(target_q, dtype=np.float32)
            if arr.ndim == 1 and arr.shape[0] != self.nq:
                raise InvalidTarget(
                    f"Expected target shape ({self.nq},), got {arr.shape}"
                )
            if arr.ndim == 2 and arr.shape[1] != self.nq:
                raise InvalidTarget(
                    f"Expected target shape (nworld, {self.nq}), got {arr.shape}"
                )
        self._set_pending(target_q, configuration=configuration)

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        self._ensure_buffers(configuration)
        assert self._target is not None
        with wp.ScopedDevice(configuration.device):
            wp.copy(self._target, configuration.q)
        self._pending = None
        self._target_set = True

    def _alloc_extra_buffers(self, configuration: Configuration) -> None:
        super()._alloc_extra_buffers(configuration)
        if self._v_ids_np is not None and len(self._v_ids_np) > 0:
            self._v_ids = wp.array(self._v_ids_np, dtype=int)

    def _broadcast_target(self, src: wp.array, nworld: int) -> None:
        assert self._target is not None
        wp.launch(
            broadcast_q,
            dim=nworld,
            inputs=[src, self.nq],
            outputs=[self._target],
        )

    def _eval(self, configuration: Configuration) -> None:
        target = self._require_target(configuration)
        assert self._error is not None
        assert self._jacobian is not None
        m = configuration.wp_model
        with wp.ScopedDevice(configuration.device):
            if self.nq == configuration.nv:
                wp.launch(
                    posture_error,
                    dim=configuration.nworld,
                    inputs=[configuration.q, target, configuration.nv],
                    outputs=[self._error],
                )
            else:
                wp.launch(
                    posture_error_joints,
                    dim=configuration.nworld,
                    inputs=[
                        configuration.q,
                        target,
                        m.jnt_type,
                        m.jnt_qposadr,
                        m.jnt_dofadr,
                        configuration.model.njnt,
                        configuration.nv,
                    ],
                    outputs=[self._error],
                )
            wp.launch(
                posture_jacobian_eye,
                dim=(configuration.nworld, configuration.nv, configuration.nv),
                inputs=[configuration.nv],
                outputs=[self._jacobian],
            )
            if self._v_ids is not None:
                wp.launch(
                    zero_free_joint_rows,
                    dim=configuration.nworld,
                    inputs=[
                        self._error,
                        self._jacobian,
                        self._v_ids,
                        len(self._v_ids_np),
                        configuration.nv,
                    ],
                )
