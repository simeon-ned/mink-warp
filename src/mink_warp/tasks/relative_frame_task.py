"""Relative frame task (device-side, Mink formulas)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import TaskDefinitionError
from ..kernels.frame import (
    broadcast_pose,
    copy_poses,
    relative_frame_task_error_jacobian,
    relative_pose_wxyz_xyz,
)
from ..lie import SE3
from .task import TargetedTask


class RelativeFrameTask(TargetedTask):
    """Regulate the pose of a frame relative to another frame."""

    k: int = 6
    target_width: int = 7

    def __init__(
        self,
        frame_name: str,
        frame_type: str,
        root_name: str,
        root_type: str,
        position_cost: npt.ArrayLike,
        orientation_cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros(6), gain=gain, lm_damping=lm_damping)
        self.frame_name = frame_name
        self.frame_type = frame_type
        self.root_name = root_name
        self.root_type = root_type
        self._frame_pose: wp.array | None = None
        self._root_pose: wp.array | None = None
        self._rel_pose: wp.array | None = None
        self._jac_frame: wp.array | None = None
        self._jac_root: wp.array | None = None
        self.set_position_cost(position_cost)
        self.set_orientation_cost(orientation_cost)

    def set_position_cost(self, position_cost: npt.ArrayLike) -> None:
        position_cost = np.atleast_1d(np.asarray(position_cost, dtype=np.float64))
        if position_cost.ndim != 1 or position_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"position cost must be shape (1,) or (3,), got {position_cost.shape}"
            )
        if not np.all(position_cost >= 0.0):
            raise TaskDefinitionError("position cost must be >= 0")
        self.cost[:3] = position_cost
        self._cost_dev = None

    def set_orientation_cost(self, orientation_cost: npt.ArrayLike) -> None:
        orientation_cost = np.atleast_1d(
            np.asarray(orientation_cost, dtype=np.float64)
        )
        if orientation_cost.ndim != 1 or orientation_cost.shape[0] not in (1, 3):
            raise TaskDefinitionError(
                f"orientation cost must be shape (1,) or (3,), got "
                f"{orientation_cost.shape}"
            )
        if not np.all(orientation_cost >= 0.0):
            raise TaskDefinitionError("orientation cost must be >= 0")
        self.cost[3:] = orientation_cost
        self._cost_dev = None

    def set_target(
        self,
        transform_target_to_root: wp.array | SE3 | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        if isinstance(transform_target_to_root, SE3):
            transform_target_to_root = transform_target_to_root.wxyz_xyz
        self._set_pending(transform_target_to_root, configuration=configuration)

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        pose = configuration.get_transform(
            self.frame_name,
            self.frame_type,
            self.root_name,
            self.root_type,
        )
        self._ensure_buffers(configuration)
        assert self._target is not None
        with wp.ScopedDevice(configuration.device):
            wp.copy(self._target, pose)
        self._pending = None
        self._target_set = True

    def _alloc_extra_buffers(self, configuration: Configuration) -> None:
        super()._alloc_extra_buffers(configuration)
        nworld = configuration.nworld
        nv = configuration.nv
        self._frame_pose = wp.zeros((nworld, 7), dtype=float)
        self._root_pose = wp.zeros((nworld, 7), dtype=float)
        self._rel_pose = wp.zeros((nworld, 7), dtype=float)
        self._jac_frame = wp.zeros((nworld, 6, nv), dtype=float)
        self._jac_root = wp.zeros((nworld, 6, nv), dtype=float)

    def _eval(self, configuration: Configuration) -> None:
        target = self._require_target(configuration)
        assert self._frame_pose is not None
        assert self._root_pose is not None
        assert self._rel_pose is not None
        assert self._jac_frame is not None
        assert self._jac_root is not None
        assert self._error is not None
        assert self._jacobian is not None

        frame_world = configuration.get_transform_frame_to_world(
            self.frame_name, self.frame_type
        )
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                copy_poses,
                dim=configuration.nworld,
                inputs=[frame_world],
                outputs=[self._frame_pose],
            )
        root_world = configuration.get_transform_frame_to_world(
            self.root_name, self.root_type
        )
        jac_frame = configuration.get_frame_jacobian(
            self.frame_name, self.frame_type
        )
        with wp.ScopedDevice(configuration.device):
            wp.copy(self._jac_frame, jac_frame)
        jac_root = configuration.get_frame_jacobian(
            self.root_name, self.root_type
        )
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                copy_poses,
                dim=configuration.nworld,
                inputs=[root_world],
                outputs=[self._root_pose],
            )
            wp.copy(self._jac_root, jac_root)
            wp.launch(
                relative_pose_wxyz_xyz,
                dim=configuration.nworld,
                inputs=[self._root_pose, self._frame_pose],
                outputs=[self._rel_pose],
            )
            wp.launch(
                relative_frame_task_error_jacobian,
                dim=configuration.nworld,
                inputs=[
                    target,
                    self._rel_pose,
                    self._jac_frame,
                    self._jac_root,
                    configuration.nv,
                ],
                outputs=[self._error, self._jacobian],
            )
