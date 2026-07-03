"""Velocity-damping regularization task."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..kernels.posture import posture_jacobian_eye, zero_free_joint_rows
from ..utils import get_freejoint_dims
from .posture_task import PostureTask


class DampingTask(PostureTask):
    """L2 regularization on joint velocities (error is identically zero)."""

    def __init__(self, model, cost: npt.ArrayLike):
        super().__init__(model, cost, gain=0.0, lm_damping=0.0)
        self._target_set = True

    def set_target(self, target_q=None, *, configuration=None) -> None:
        self._target_set = True

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        self._target_set = True

    def _eval(self, configuration: Configuration) -> None:
        self._ensure_buffers(configuration)
        assert self._error is not None
        assert self._jacobian is not None
        self._error.zero_()
        with wp.ScopedDevice(configuration.device):
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
