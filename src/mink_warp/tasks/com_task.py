"""Center-of-mass task (device-side, Mink-compatible)."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import InvalidTarget, TaskDefinitionError
from ..kernels.com import (
    broadcast_vec3,
    com_error,
    com_jacobian,
    copy_vec3_batch,
    subtree_com_to_batch,
)
from .task import TargetedTask

_SUBTREE_ID = 1


class ComTask(TargetedTask):
    r"""Regulate the center of mass of subtree body 1 (whole robot).

    .. math::

        e(q) = c(q) - c^\star, \qquad J(q) = \frac{\partial c}{\partial q}

    where :math:`c(q) \in \mathbb{R}^3` is the mass-weighted subtree CoM.
    Cost units: :math:`[\mathrm{cost}] / [\mathrm{m}]` per axis.
    """

    k: int = 3
    target_width: int = 3

    def __init__(
        self,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        super().__init__(cost=np.zeros(3), gain=gain, lm_damping=lm_damping)
        self.set_cost(cost)

    def set_cost(self, cost: npt.ArrayLike) -> None:
        cost = np.atleast_1d(np.asarray(cost, dtype=np.float64))
        if cost.ndim != 1 or cost.shape[0] not in (1, self.k):
            raise TaskDefinitionError(
                f"cost must be shape (1,) or ({self.k},), got {cost.shape}"
            )
        if not np.all(cost >= 0.0):
            raise TaskDefinitionError("cost must be >= 0")
        self.cost[:] = cost
        self._cost_dev = None

    def set_target(
        self,
        target_com: wp.array | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        if not isinstance(target_com, wp.array):
            arr = np.asarray(target_com, dtype=np.float32)
            if arr.shape != (3,) and not (arr.ndim == 2 and arr.shape[1] == 3):
                raise InvalidTarget(
                    f"Expected CoM shape (3,) or (nworld, 3), got {arr.shape}"
                )
        self._set_pending(target_com, configuration=configuration)

    def set_target_from_configuration(self, configuration: Configuration) -> None:
        self._ensure_buffers(configuration)
        assert self._target is not None
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                subtree_com_to_batch,
                dim=configuration.nworld,
                inputs=[configuration.wp_data.subtree_com, _SUBTREE_ID],
                outputs=[self._target],
            )
        self._pending = None
        self._target_set = True

    def _broadcast_target(self, src: wp.array, nworld: int) -> None:
        assert self._target is not None
        wp.launch(broadcast_vec3, dim=nworld, inputs=[src], outputs=[self._target])

    def _flush_pending(self, nworld: int, device: str | None) -> None:
        assert self._target is not None and device is not None
        src = self._pending
        if src is None:
            from ..exceptions import TargetNotSet

            if not self._target_set:
                raise TargetNotSet(self.__class__.__name__)
            return
        with wp.ScopedDevice(device):
            if src.shape == (3,):
                self._broadcast_target(src, nworld)
            elif src.shape == (nworld, 3):
                wp.launch(
                    copy_vec3_batch,
                    dim=nworld,
                    inputs=[src],
                    outputs=[self._target],
                )
            else:
                from ..exceptions import TargetNotSet

                raise TargetNotSet(
                    f"{self.__class__.__name__}: target shape {src.shape} "
                    f"incompatible with nworld={nworld}"
                )
        self._pending = None

    def _eval(self, configuration: Configuration) -> None:
        target = self._require_target(configuration)
        assert self._error is not None
        assert self._jacobian is not None
        m = configuration.wp_model
        d = configuration.wp_data
        with wp.ScopedDevice(configuration.device):
            wp.launch(
                com_error,
                dim=configuration.nworld,
                inputs=[d.subtree_com, target, _SUBTREE_ID],
                outputs=[self._error],
            )
            wp.launch(
                com_jacobian,
                dim=(configuration.nworld, configuration.nv),
                inputs=[
                    configuration.model.nbody,
                    m.body_parentid,
                    m.body_rootid,
                    m.body_mass,
                    m.body_subtreemass,
                    m.dof_bodyid,
                    m.body_isdofancestor,
                    d.xipos,
                    d.subtree_com,
                    d.cdof,
                    _SUBTREE_ID,
                ],
                outputs=[self._jacobian],
            )
