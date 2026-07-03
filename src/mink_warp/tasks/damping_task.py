"""Damping task implementation (device-side)."""

from __future__ import annotations

import mujoco
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..lie.kernels import posture_jacobian_eye, zero_free_joint_rows
from .posture_task import PostureTask


class DampingTask(PostureTask):
    r"""L2-regularization on joint velocities (velocity damping).

    Matches Mink: error is identically zero; contributes
    :math:`\frac12 \Delta q^\top \Lambda^2 \Delta q` via the cost-weighted
    identity Jacobian. No target is required.
    """

    def __init__(self, model: mujoco.MjModel, cost: npt.ArrayLike):
        super().__init__(model, cost, gain=0.0, lm_damping=0.0)
        self._target_set = True

    def set_target(self, target_q=None, *, configuration=None) -> None:
        """No-op; damping has no target."""
        self._target_set = True

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        """No-op; damping has no target."""
        self._target_set = True

    def compute_error(self, configuration: Configuration) -> wp.array:
        self._ensure_buffers(configuration)
        assert self._error_wp is not None
        self._error_wp.zero_()
        return self._error_wp

    def _eval(self, configuration: Configuration) -> None:
        self._ensure_buffers(configuration)
        assert self._error_wp is not None
        assert self._jac_wp is not None
        self._error_wp.zero_()
        with wp.ScopedDevice(configuration.device):
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
