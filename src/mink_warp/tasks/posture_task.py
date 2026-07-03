"""Posture task implementation (device-side)."""

from __future__ import annotations

import mujoco
import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TargetNotSet, TaskDefinitionError
from ..interop import to_wp
from ..lie.kernels import (
    broadcast_q,
    posture_error,
    posture_jacobian_eye,
    weighted_residual,
    zero_free_joint_rows,
)
from ..utils import get_freejoint_dims
from .task import Objective, Task


class PostureTask(Task):
    """Regulate joint angles towards a target posture.

    Targets and residuals are ``wp.array``. Hinge/slide error is ``q - q*`` on
    device (matches ``mj_differentiatePos`` when ``nq == nv``). Free-joint dofs
    are zeroed like Mink.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(
            cost=np.zeros((model.nv,)),
            gain=gain,
            lm_damping=lm_damping,
        )
        self._target_wp: wp.array | None = None
        self._pending_target: wp.array | None = None
        self._error_wp: wp.array | None = None
        self._jac_wp: wp.array | None = None
        self._nworld: int | None = None
        self._device: str | None = None
        self._target_set = False
        self._cost_wp: wp.array | None = None
        self._weighted_jac_wp: wp.array | None = None
        self._weighted_err_wp: wp.array | None = None
        self._mu_wp: wp.array | None = None
        self._H_wp: wp.array | None = None
        self._c_wp: wp.array | None = None

        _, v_ids = get_freejoint_dims(model)
        self._v_ids_np = np.asarray(v_ids, dtype=np.int32) if v_ids else None
        self._v_ids_wp: wp.array | None = None

        self.k = model.nv
        self.nq = model.nq
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        cost = np.atleast_1d(np.asarray(cost, dtype=np.float64))
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} cost must be shape (1,) or ({self.k},). "
                f"Got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError(f"{self.__class__.__name__} cost should be >= 0")
        self.cost[: self.k] = cost
        self._cost_wp = None

    def set_target(
        self,
        target_q: wp.array | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        """Set target posture.

        Primary path: ``wp.array`` of shape ``(nworld, nq)`` or ``(nq,)``.
        Optional: NumPy — uploaded once via :func:`~mink_warp.interop.to_wp`.
        """
        if isinstance(target_q, wp.array):
            src = target_q
        else:
            arr = np.asarray(target_q, dtype=np.float32)
            if arr.ndim == 1:
                if arr.shape[0] != self.nq:
                    raise InvalidTarget(
                        f"Expected target posture shape ({self.nq},), got {arr.shape}"
                    )
            elif arr.ndim == 2:
                if arr.shape[1] != self.nq:
                    raise InvalidTarget(
                        f"Expected target posture shape (nworld, {self.nq}), "
                        f"got {arr.shape}"
                    )
            else:
                raise InvalidTarget(
                    f"Expected 1D or 2D target posture, got {arr.shape}"
                )
            src = to_wp(arr, dtype=float)

        self._pending_target = src
        self._target_set = True
        if configuration is not None:
            self._ensure_buffers(configuration)
            self._copy_pending_target(configuration)
        elif self._target_wp is not None and self._nworld is not None:
            self._copy_pending_into_existing()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Copy current ``q`` into the target buffer (device→device)."""
        self._ensure_buffers(configuration)
        assert self._target_wp is not None
        with wp.ScopedDevice(configuration.device):
            wp.copy(self._target_wp, configuration.q)
        self._pending_target = None
        self._target_set = True

    def compute_error(self, configuration: Configuration) -> wp.array:
        self._eval(configuration)
        assert self._error_wp is not None
        return self._error_wp

    def compute_jacobian(self, configuration: Configuration) -> wp.array:
        self._eval(configuration)
        assert self._jac_wp is not None
        return self._jac_wp

    def compute_qp_residual(
        self, configuration: Configuration
    ) -> tuple[wp.array, wp.array, wp.array]:
        self._eval(configuration)
        self._ensure_buffers(configuration)
        assert self._error_wp is not None
        assert self._jac_wp is not None
        assert self._weighted_jac_wp is not None
        assert self._weighted_err_wp is not None
        assert self._mu_wp is not None

        with wp.ScopedDevice(configuration.device):
            if self._cost_wp is None:
                self._cost_wp = wp.array(self.cost.astype(np.float32), dtype=float)
            wp.launch(
                weighted_residual,
                dim=configuration.nworld,
                inputs=[
                    self._error_wp,
                    self._jac_wp,
                    self._cost_wp,
                    float(self.gain),
                    float(self.lm_damping),
                    self.k,
                    configuration.nv,
                ],
                outputs=[self._weighted_jac_wp, self._weighted_err_wp, self._mu_wp],
            )
        return self._weighted_jac_wp, self._weighted_err_wp, self._mu_wp

    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        W, e, mu = self.compute_qp_residual(configuration)
        W_np, e_np, mu_np = W.numpy(), e.numpy(), mu.numpy()
        H = np.einsum("bki,bkj->bij", W_np, W_np)
        if np.any(mu_np > 0.0):
            H = H + mu_np[:, None, None] * np.eye(configuration.nv)[None, :, :]
        c = -np.einsum("bk,bki->bi", e_np, W_np)
        with wp.ScopedDevice(configuration.device):
            self._H_wp = wp.array(H.astype(np.float32), dtype=float)
            self._c_wp = wp.array(c.astype(np.float32), dtype=float)
        return Objective(self._H_wp, self._c_wp)

    def _ensure_buffers(self, configuration: Configuration) -> None:
        nworld = configuration.nworld
        device = configuration.device
        if self._nworld == nworld and self._device == device and self._error_wp is not None:
            return
        with wp.ScopedDevice(device):
            self._target_wp = wp.zeros((nworld, self.nq), dtype=float)
            self._error_wp = wp.zeros((nworld, configuration.nv), dtype=float)
            self._jac_wp = wp.zeros(
                (nworld, configuration.nv, configuration.nv), dtype=float
            )
            self._cost_wp = wp.array(self.cost.astype(np.float32), dtype=float)
            self._weighted_jac_wp = wp.zeros(
                (nworld, configuration.nv, configuration.nv), dtype=float
            )
            self._weighted_err_wp = wp.zeros((nworld, configuration.nv), dtype=float)
            self._mu_wp = wp.zeros(nworld, dtype=float)
            if self._v_ids_np is not None and len(self._v_ids_np) > 0:
                self._v_ids_wp = wp.array(self._v_ids_np, dtype=int)
            else:
                self._v_ids_wp = None
        self._nworld = nworld
        self._device = device

    def _copy_pending_into_existing(self) -> None:
        assert self._target_wp is not None and self._nworld is not None
        assert self._device is not None
        src = self._pending_target
        if src is None:
            return
        with wp.ScopedDevice(self._device):
            if src.shape == (self.nq,):
                wp.launch(
                    broadcast_q,
                    dim=self._nworld,
                    inputs=[src, self.nq],
                    outputs=[self._target_wp],
                )
            elif src.shape == (self._nworld, self.nq):
                wp.copy(self._target_wp, src)
            else:
                raise TargetNotSet(
                    f"{self.__class__.__name__}: target shape {src.shape} "
                    f"incompatible with nworld={self._nworld}"
                )
        self._pending_target = None

    def _copy_pending_target(self, configuration: Configuration) -> None:
        assert self._target_wp is not None
        src = self._pending_target
        if src is None:
            if not self._target_set:
                raise TargetNotSet(self.__class__.__name__)
            return
        with wp.ScopedDevice(configuration.device):
            if src.shape == (self.nq,):
                wp.launch(
                    broadcast_q,
                    dim=configuration.nworld,
                    inputs=[src, self.nq],
                    outputs=[self._target_wp],
                )
            elif src.shape == (configuration.nworld, self.nq):
                wp.copy(self._target_wp, src)
            else:
                raise TargetNotSet(
                    f"{self.__class__.__name__}: target shape {src.shape} "
                    f"incompatible with nworld={configuration.nworld}"
                )
        self._pending_target = None

    def _eval(self, configuration: Configuration) -> None:
        if not self._target_set:
            raise TargetNotSet(self.__class__.__name__)

        self._ensure_buffers(configuration)
        self._copy_pending_target(configuration)
        assert self._target_wp is not None
        assert self._error_wp is not None
        assert self._jac_wp is not None

        with wp.ScopedDevice(configuration.device):
            if self.nq == configuration.nv:
                wp.launch(
                    posture_error,
                    dim=configuration.nworld,
                    inputs=[configuration.q, self._target_wp, configuration.nv],
                    outputs=[self._error_wp],
                )
            else:
                # Free-joint models: host differentiatePos until a device kernel lands.
                q = configuration.q.numpy()
                t = self._target_wp.numpy()
                err = np.zeros(
                    (configuration.nworld, configuration.nv), dtype=np.float32
                )
                for i in range(configuration.nworld):
                    mujoco.mj_differentiatePos(
                        configuration.model, err[i], 1.0, t[i], q[i]
                    )
                self._error_wp.assign(err)

            wp.launch(
                posture_jacobian_eye,
                dim=(configuration.nworld, configuration.nv, configuration.nv),
                inputs=[configuration.nv],
                outputs=[self._jac_wp],
            )

            if self._v_ids_wp is not None:
                wp.launch(
                    zero_free_joint_rows,
                    dim=configuration.nworld,
                    inputs=[
                        self._error_wp,
                        self._jac_wp,
                        self._v_ids_wp,
                        len(self._v_ids_np),
                        configuration.nv,
                    ],
                )
