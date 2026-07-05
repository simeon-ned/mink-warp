"""Hard joint velocity limit.

Batched device form of mink's ``VelocityLimit``: a symmetric, configuration-
independent box ``-dt*vmax <= dq <= dt*vmax`` on the selected hinge / slide dofs.
"""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..kernels.constrained import velocity_limit_box, velocity_limit_ineq
from .limit import Limit

_SCALAR_JOINTS = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)


class VelocityLimit(Limit):
    """Box on ``dq`` bounding each selected joint's per-step displacement."""

    def __init__(
        self,
        model: mujoco.MjModel,
        velocities: Mapping[str, npt.ArrayLike] = {},
    ):
        dofadr: list[int] = []
        vmax: list[float] = []
        for joint_name, max_vel in velocities.items():
            jid = model.joint(joint_name).id
            jnt_type = model.jnt_type[jid]
            if jnt_type not in _SCALAR_JOINTS:
                raise ValueError(
                    f"VelocityLimit supports only hinge/slide joints; "
                    f"{joint_name!r} is not one."
                )
            v = float(np.atleast_1d(max_vel)[0])
            if v < 0.0:
                raise ValueError(f"Velocity limit for {joint_name!r} must be >= 0.")
            dofadr.append(int(model.jnt_dofadr[jid]))
            vmax.append(v)

        self.model = model
        self.nb = len(dofadr)
        # Dense inequality form: +-e_i dq <= dt*vmax -> two rows per bounded dof.
        self.n_inequalities = 2 * self.nb
        self._dofadr_np = np.asarray(dofadr, dtype=np.int32)
        self._vmax_np = np.asarray(vmax, dtype=np.float32)
        self._dev: dict[str, tuple[wp.array, wp.array]] = {}

    def _ensure_dev(self, device: str):
        cached = self._dev.get(device)
        if cached is not None:
            return cached
        with wp.ScopedDevice(device):
            arrs = (
                wp.array(self._dofadr_np, dtype=wp.int32),
                wp.array(self._vmax_np, dtype=float),
            )
        self._dev[device] = arrs
        return arrs

    def apply_box(
        self,
        configuration: Configuration,
        dt: float,
        lo: wp.array,
        hi: wp.array,
    ) -> None:
        if self.nb == 0:
            return
        device = configuration.device
        dofadr, vmax = self._ensure_dev(device)
        with wp.ScopedDevice(device):
            wp.launch(
                velocity_limit_box,
                dim=configuration.nworld,
                inputs=[dofadr, vmax, float(dt), self.nb],
                outputs=[lo, hi],
            )

    def scatter_inequalities(
        self,
        configuration: Configuration,
        dt: float,
        row_offset: int,
        G: wp.array,
        h: wp.array,
    ) -> None:
        if self.nb == 0:
            return
        device = configuration.device
        dofadr, vmax = self._ensure_dev(device)
        with wp.ScopedDevice(device):
            wp.launch(
                velocity_limit_ineq,
                dim=configuration.nworld,
                inputs=[dofadr, vmax, float(dt), self.nb, int(row_offset)],
                outputs=[G, h],
            )
