"""Soft joint configuration limits as a least-squares task."""

from __future__ import annotations

import mujoco
import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError
from ..kernels.limits import joint_limit_error_jac
from .task import Task


class JointLimitTask(Task):
    r"""Soft hinge/slide joint limits as a least-squares penalty.

    When :math:`q_i` violates bounds :math:`[q_i^{\min}, q_i^{\max}]`:

    .. math::

        e_i = \begin{cases}
            q_i - q_i^{\max} & q_i > q_i^{\max} \\
            q_i - q_i^{\min} & q_i < q_i^{\min} \\
            0 & \text{otherwise}
        \end{cases}

    with :math:`J_{ii} = 1` on limited dofs. Free and ball joints are ignored.
    For **hard** limits use :class:`~mink_warp.ConfigurationLimit`.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: npt.ArrayLike = 1.0,
        gain: float = 1.0,
        lm_damping: float = 0.0,
        min_distance_from_limits: float = 0.0,
    ):
        super().__init__(cost=np.zeros(model.nv), gain=gain, lm_damping=lm_damping)
        self.k = model.nv
        self.model = model

        qposadr: list[int] = []
        dofadr: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        for jnt in range(model.njnt):
            jnt_type = model.jnt_type[jnt]
            if jnt_type in (
                mujoco.mjtJoint.mjJNT_FREE,
                mujoco.mjtJoint.mjJNT_BALL,
            ):
                continue
            if not model.jnt_limited[jnt]:
                continue
            qposadr.append(int(model.jnt_qposadr[jnt]))
            dofadr.append(int(model.jnt_dofadr[jnt]))
            lo, hi = model.jnt_range[jnt]
            lower.append(float(lo + min_distance_from_limits))
            upper.append(float(hi - min_distance_from_limits))

        self._n_limited = len(qposadr)
        self._qposadr_np = np.asarray(qposadr, dtype=np.int32)
        self._dofadr_np = np.asarray(dofadr, dtype=np.int32)
        self._lower_np = np.asarray(lower, dtype=np.float32)
        self._upper_np = np.asarray(upper, dtype=np.float32)
        self._qposadr: wp.array | None = None
        self._dofadr: wp.array | None = None
        self._lower: wp.array | None = None
        self._upper: wp.array | None = None
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

    def _alloc_extra_buffers(self, configuration: Configuration) -> None:
        if self._n_limited > 0:
            self._qposadr = wp.array(self._qposadr_np, dtype=int)
            self._dofadr = wp.array(self._dofadr_np, dtype=int)
            self._lower = wp.array(self._lower_np, dtype=float)
            self._upper = wp.array(self._upper_np, dtype=float)
        else:
            self._qposadr = wp.zeros(1, dtype=int)
            self._dofadr = wp.zeros(1, dtype=int)
            self._lower = wp.zeros(1, dtype=float)
            self._upper = wp.zeros(1, dtype=float)

    def _eval(self, configuration: Configuration) -> None:
        assert self._error is not None
        assert self._jacobian is not None
        if self._n_limited == 0:
            self._error.zero_()
            self._jacobian.zero_()
            return
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                joint_limit_error_jac,
                dim=configuration.nworld,
                inputs=[
                    configuration.q,
                    self._lower,
                    self._upper,
                    self._qposadr,
                    self._dofadr,
                    self._n_limited,
                    configuration.nv,
                ],
                outputs=[self._error, self._jacobian],
            )


# Backward-compatible alias.
ConfigurationLimitTask = JointLimitTask
