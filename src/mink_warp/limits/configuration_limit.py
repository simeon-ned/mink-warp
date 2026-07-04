"""Hard joint position limit (configuration limit).

Batched device form of mink's ``ConfigurationLimit``. For each limited hinge /
slide joint (one qpos <-> one dof) the tangent step is boxed as

    gain*(lower - q)  <=  dq  <=  gain*(upper - q)

which is mink's ``G=[P;-P]``, ``h=[gain*(upper-q); gain*(q-lower)]`` written as a
box. Free and ball joints are ignored (they are unlimited here), matching mink's
free-joint skip; ball-joint limits are not yet supported.
"""

from __future__ import annotations

import mujoco
import numpy as np
import warp as wp

from ..configuration import Configuration
from ..exceptions import InvalidGain
from ..kernels.constrained import config_limit_box, config_limit_ineq
from .limit import Limit

_SCALAR_JOINTS = (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE)


class ConfigurationLimit(Limit):
    """Box on ``dq`` keeping each limited joint inside its range."""

    def __init__(
        self,
        model: mujoco.MjModel,
        gain: float = 0.95,
        min_distance_from_limits: float = 0.0,
    ):
        if not 0.0 < gain <= 1.0:
            raise InvalidGain(gain)

        qposadr: list[int] = []
        dofadr: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        for jnt in range(model.njnt):
            jnt_type = model.jnt_type[jnt]
            if not model.jnt_limited[jnt]:
                continue
            if jnt_type == mujoco.mjtJoint.mjJNT_BALL:
                # mink enforces limited ball joints (3 dofs via mj_differentiatePos);
                # we don't yet. Fail loud rather than silently drop a hard limit.
                raise NotImplementedError(
                    f"ConfigurationLimit does not yet support the limited ball "
                    f"joint {model.joint(jnt).name!r}; only hinge/slide joints. "
                    f"mink enforces this limit, so silently ignoring it would "
                    f"break parity and safety."
                )
            if jnt_type not in _SCALAR_JOINTS:
                continue  # free joints are never limited here (mink skips them too)
            lo, hi = model.jnt_range[jnt]
            qposadr.append(int(model.jnt_qposadr[jnt]))
            dofadr.append(int(model.jnt_dofadr[jnt]))
            lower.append(float(lo) + min_distance_from_limits)
            upper.append(float(hi) - min_distance_from_limits)

        self.model = model
        self.gain = float(gain)
        self.n_limited = len(dofadr)
        # Dense inequality form: mink's G=[P;-P] -> two rows per limited dof.
        self.n_inequalities = 2 * self.n_limited
        self._qposadr_np = np.asarray(qposadr, dtype=np.int32)
        self._dofadr_np = np.asarray(dofadr, dtype=np.int32)
        self._lower_np = np.asarray(lower, dtype=np.float32)
        self._upper_np = np.asarray(upper, dtype=np.float32)
        self._dev: dict[str, tuple[wp.array, wp.array, wp.array, wp.array]] = {}

    def _ensure_dev(self, device: str):
        cached = self._dev.get(device)
        if cached is not None:
            return cached
        with wp.ScopedDevice(device):
            arrs = (
                wp.array(self._qposadr_np, dtype=wp.int32),
                wp.array(self._dofadr_np, dtype=wp.int32),
                wp.array(self._lower_np, dtype=float),
                wp.array(self._upper_np, dtype=float),
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
        del dt  # Configuration limits are timestep-independent.
        if self.n_limited == 0:
            return
        device = configuration.device
        qposadr, dofadr, lower, upper = self._ensure_dev(device)
        with wp.ScopedDevice(device):
            wp.launch(
                config_limit_box,
                dim=configuration.nworld,
                inputs=[
                    configuration.q,
                    qposadr,
                    dofadr,
                    lower,
                    upper,
                    self.gain,
                    self.n_limited,
                ],
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
        del dt  # Configuration limits are timestep-independent.
        if self.n_limited == 0:
            return
        device = configuration.device
        qposadr, dofadr, lower, upper = self._ensure_dev(device)
        with wp.ScopedDevice(device):
            wp.launch(
                config_limit_ineq,
                dim=configuration.nworld,
                inputs=[
                    configuration.q,
                    qposadr,
                    dofadr,
                    lower,
                    upper,
                    self.gain,
                    self.n_limited,
                    int(row_offset),
                ],
                outputs=[G, h],
            )
