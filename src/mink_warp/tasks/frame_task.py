"""Frame task implementation (device-side, Mink formulas)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import TargetNotSet, TaskDefinitionError
from ..interop import to_wp
from ..lie import SE3
from ..lie.kernels import (
    broadcast_pose,
    copy_poses,
    frame_task_error_jacobian,
    weighted_residual,
)
from .task import Objective, Task


class FrameTask(Task):
    """Regulate the position and orientation of a frame of interest.

    Targets and residuals are ``wp.array`` (Newton-style). NumPy / ``SE3`` are
    accepted only as optional one-shot uploads in :meth:`set_target`.

    * :math:`e(q) = \\log(T_{bt})` (body twist)
    * :math:`J(q) = -\\mathrm{jlog}(T_{tb})\\, {}_b J_{wb}`
    """

    k: int = 6

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros((self.k,)), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self._target_wp: wp.array | None = None
        self._pending_target: wp.array | None = None
        self._target_set = False
        self._nworld: int | None = None
        self._device: str | None = None
        self._error_wp: wp.array | None = None
        self._jac_wp: wp.array | None = None
        self._frame_pose_wp: wp.array | None = None
        self._cost_wp: wp.array | None = None
        self._weighted_jac_wp: wp.array | None = None
        self._weighted_err_wp: wp.array | None = None
        self._mu_wp: wp.array | None = None
        self._H_wp: wp.array | None = None
        self._c_wp: wp.array | None = None

        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(self, position_cost: npt.ArrayLike) -> None:
        position_cost = np.atleast_1d(np.asarray(position_cost, dtype=np.float64))
        if position_cost.ndim != 1 or position_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be a vector of shape "
                f"1 or (3,) but got {position_cost.shape}"
            )
        if not np.all(position_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} position cost should be >= 0"
            )
        self.cost[:3] = position_cost
        self._cost_wp = None

    def set_orientation_cost(self, orientation_cost: npt.ArrayLike) -> None:
        orientation_cost = np.atleast_1d(
            np.asarray(orientation_cost, dtype=np.float64)
        )
        if orientation_cost.ndim != 1 or orientation_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} orientation cost should be a vector of "
                f"shape 1 or (3,) but got {orientation_cost.shape}"
            )
        if not np.all(orientation_cost >= 0.0):
            raise TaskDefinitionError(
                f"{self.__class__.__name__} orientation cost should be >= 0"
            )
        self.cost[3:] = orientation_cost
        self._cost_wp = None

    def set_target(
        self,
        transform_target_to_world: wp.array | SE3 | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        """Set target pose(s).

        Primary path: ``wp.array`` of shape ``(nworld, 7)`` or ``(7,)`` (broadcast).
        Optional: ``SE3`` / NumPy — uploaded once via :func:`~mink_warp.interop.to_wp`.

        If ``configuration`` is given (or buffers were already allocated), the
        target is copied into the owned device buffer immediately. Otherwise it
        is applied on the next :meth:`compute_error` / :meth:`compute_jacobian`.
        """
        if isinstance(transform_target_to_world, SE3):
            src = to_wp(
                transform_target_to_world.wxyz_xyz.astype(np.float32), dtype=float
            )
        elif isinstance(transform_target_to_world, wp.array):
            src = transform_target_to_world
        else:
            arr = np.asarray(transform_target_to_world, dtype=np.float32)
            if arr.shape != (7,) and not (arr.ndim == 2 and arr.shape[1] == 7):
                raise TaskDefinitionError(
                    f"Expected target shape (7,) or (nworld, 7), got {arr.shape}"
                )
            src = to_wp(arr, dtype=float)

        self._pending_target = src
        self._target_set = True
        if configuration is not None:
            self._ensure_buffers(configuration)
            self._copy_pending_target(configuration)
        elif self._target_wp is not None and self._nworld is not None:
            # Buffers exist from a prior eval; need a configuration-like nworld.
            self._copy_pending_into_existing()

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """Copy current frame poses into the target buffer (device→device)."""
        self._ensure_buffers(configuration)
        assert self._target_wp is not None
        pose = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        with wp.ScopedDevice(configuration.device):
            wp.copy(self._target_wp, pose)
        self._pending_target = None
        self._target_set = True

    def compute_error(self, configuration: Configuration) -> wp.array:
        """Device error, shape ``(nworld, 6)``."""
        self._eval(configuration)
        assert self._error_wp is not None
        return self._error_wp

    def compute_jacobian(self, configuration: Configuration) -> wp.array:
        """Device Jacobian, shape ``(nworld, 6, nv)``."""
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
        # Host assemble for now; Phase 3 moves this fully on device.
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

    # Internal.

    def _ensure_buffers(self, configuration: Configuration) -> None:
        nworld = configuration.nworld
        device = configuration.device
        if (
            self._nworld == nworld
            and self._device == device
            and self._error_wp is not None
        ):
            return

        with wp.ScopedDevice(device):
            self._target_wp = wp.zeros((nworld, 7), dtype=float)
            self._frame_pose_wp = wp.zeros((nworld, 7), dtype=float)
            self._error_wp = wp.zeros((nworld, 6), dtype=float)
            self._jac_wp = wp.zeros((nworld, 6, configuration.nv), dtype=float)
            self._cost_wp = wp.array(self.cost.astype(np.float32), dtype=float)
            self._weighted_jac_wp = wp.zeros((nworld, 6, configuration.nv), dtype=float)
            self._weighted_err_wp = wp.zeros((nworld, 6), dtype=float)
            self._mu_wp = wp.zeros(nworld, dtype=float)
        self._nworld = nworld
        self._device = device

    def _copy_pending_into_existing(self) -> None:
        assert self._target_wp is not None and self._nworld is not None
        assert self._device is not None
        src = self._pending_target
        if src is None:
            return
        with wp.ScopedDevice(self._device):
            if src.shape == (7,):
                wp.launch(
                    broadcast_pose,
                    dim=self._nworld,
                    inputs=[src],
                    outputs=[self._target_wp],
                )
            elif src.shape == (self._nworld, 7):
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
            if src.shape == (7,):
                wp.launch(
                    broadcast_pose,
                    dim=configuration.nworld,
                    inputs=[src],
                    outputs=[self._target_wp],
                )
            elif src.shape == (configuration.nworld, 7):
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
        assert self._frame_pose_wp is not None
        assert self._error_wp is not None
        assert self._jac_wp is not None

        frame_pose = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        jac_body = configuration.get_frame_jacobian(
            self.frame_name, self.frame_type
        )

        with wp.ScopedDevice(configuration.device):
            # Snapshot pose: configuration buffer is reused.
            wp.launch(
                copy_poses,
                dim=configuration.nworld,
                inputs=[frame_pose],
                outputs=[self._frame_pose_wp],
            )
            wp.launch(
                frame_task_error_jacobian,
                dim=configuration.nworld,
                inputs=[
                    self._target_wp,
                    self._frame_pose_wp,
                    jac_body,
                    configuration.nv,
                ],
                outputs=[self._error_wp, self._jac_wp],
            )
