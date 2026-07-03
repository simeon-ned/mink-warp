"""Batched configuration space of a robot model on MuJoCo Warp."""

from __future__ import annotations

import mujoco
import mujoco_warp as mjwarp
import numpy as np
import numpy.typing as npt
import warp as wp

from . import constants as consts
from . import exceptions
from .interop import to_wp
from .lie import SE3
from .lie.kernels import (
    body_frame_jacobian,
    broadcast_q,
    fill_body_frame_query,
    fill_geom_frame_query,
    fill_site_frame_query,
    frame_pose_wxyz_xyz,
)


class Configuration:
    """Batched robot configuration backed by MuJoCo Warp.

    Device-native API (Newton-style): ``q``, Jacobians, and poses are
    ``wp.array``. Use ``.numpy()`` or the ``*_numpy`` / ``*_se3`` helpers only
    at host boundaries.
    """

    def __init__(
        self,
        model: mujoco.MjModel,
        q: npt.ArrayLike | wp.array | None = None,
        nworld: int = 1,
        device: str | None = None,
    ):
        if nworld < 1:
            raise ValueError(f"nworld must be >= 1, got {nworld}")

        self.model = model
        self.nworld = nworld
        self.device = device if device is not None else str(wp.get_device())
        self._frame_id_cache: dict[tuple[str, str], int] = {}
        self._body_id_cache: dict[tuple[str, str], int] = {}

        with wp.ScopedDevice(self.device):
            self.wp_model = mjwarp.put_model(model)
            self.wp_data = mjwarp.make_data(model, nworld=nworld)
            self._jacp_wp = wp.zeros((nworld, 3, model.nv), dtype=float)
            self._jacr_wp = wp.zeros((nworld, 3, model.nv), dtype=float)
            self._jac_body_wp = wp.zeros((nworld, 6, model.nv), dtype=float)
            self._point_wp = wp.zeros(nworld, dtype=wp.vec3)
            self._body_wp = wp.zeros(nworld, dtype=wp.int32)
            self._pose_wp = wp.zeros((nworld, 7), dtype=float)
            self._q_broadcast_wp = wp.zeros(model.nq, dtype=float)
            self._q_out_wp = wp.zeros((nworld, model.nq), dtype=float)
            self._v_wp = wp.zeros((nworld, model.nv), dtype=float)
            self._dt_wp = wp.zeros(1, dtype=float)

        if q is None:
            q = np.broadcast_to(model.qpos0, (nworld, model.nq)).copy()
        self.update(q=q)

    def update(self, q: npt.ArrayLike | wp.array | None = None) -> None:
        """Run forward kinematics on device.

        Args:
            q: ``wp.array`` of shape ``(nworld, nq)`` or ``(nq,)``, or a NumPy
                array (uploaded once). Prefer device arrays in the hot path.
        """
        with wp.ScopedDevice(self.device):
            if q is not None:
                self._assign_qpos(q)
            mjwarp.kinematics(self.wp_model, self.wp_data)
            mjwarp.com_pos(self.wp_model, self.wp_data)

    def update_from_keyframe(self, key_name: str) -> None:
        key_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_KEY, key_name)
        if key_id == -1:
            raise exceptions.InvalidKeyframe(key_name, self.model)
        self.update(q=self.model.key_qpos[key_id])

    def get_frame_jacobian(self, frame_name: str, frame_type: str) -> wp.array:
        """Body-frame Jacobian on device, shape ``(nworld, 6, nv)``.

        The returned buffer is owned by this configuration and overwritten on
        the next Jacobian query — ``wp.copy`` if you need to keep it.
        """
        frame_id = self._resolve_frame_id(frame_name, frame_type)
        body_id = self._resolve_body_id(frame_name, frame_type, frame_id)
        xpos_arr, xmat_arr = self._frame_arrays(frame_type)

        with wp.ScopedDevice(self.device):
            self._launch_frame_query(frame_type, frame_id, body_id, xpos_arr)
            mjwarp.jac(
                self.wp_model,
                self.wp_data,
                self._jacp_wp,
                self._jacr_wp,
                self._point_wp,
                self._body_wp,
            )
            wp.launch(
                body_frame_jacobian,
                dim=(self.nworld, self.nv),
                inputs=[xmat_arr, frame_id, self._jacp_wp, self._jacr_wp],
                outputs=[self._jac_body_wp],
            )
        return self._jac_body_wp

    def get_transform_frame_to_world(
        self, frame_name: str, frame_type: str
    ) -> wp.array:
        """Frame poses on device as ``wxyz_xyz``, shape ``(nworld, 7)``.

        The returned buffer is owned by this configuration and overwritten on
        the next pose query — ``wp.copy`` if you need to keep it.
        """
        frame_id = self._resolve_frame_id(frame_name, frame_type)
        xpos_arr, xmat_arr = self._frame_arrays(frame_type)
        with wp.ScopedDevice(self.device):
            wp.launch(
                frame_pose_wxyz_xyz,
                dim=self.nworld,
                inputs=[xpos_arr, xmat_arr, frame_id],
                outputs=[self._pose_wp],
            )
        return self._pose_wp

    def get_transform_frame_to_world_se3(
        self, frame_name: str, frame_type: str
    ) -> SE3:
        """Host ``SE3`` for world 0 (optional convenience)."""
        pose = self.get_transform_frame_to_world(frame_name, frame_type).numpy()
        return SE3(wxyz_xyz=pose[0].astype(np.float64))

    def integrate(self, velocity: npt.ArrayLike | wp.array, dt: float) -> wp.array:
        """Integrate velocity on device; returns new ``q`` of shape ``(nworld, nq)``.

        Uses MuJoCo Warp's ``_next_position`` kernel (same as the simulator's
        position integrate: free / ball / hinge / slide).
        """
        with wp.ScopedDevice(self.device):
            wp.copy(self._q_out_wp, self.wp_data.qpos)
            self._integrate_into(self._q_out_wp, velocity, dt)
            return self._q_out_wp

    def integrate_inplace(
        self, velocity: npt.ArrayLike | wp.array, dt: float
    ) -> None:
        """Integrate velocity into ``q`` on device and refresh kinematics."""
        with wp.ScopedDevice(self.device):
            self._integrate_into(self.wp_data.qpos, velocity, dt)
            mjwarp.kinematics(self.wp_model, self.wp_data)
            mjwarp.com_pos(self.wp_model, self.wp_data)

    def _integrate_into(
        self,
        q_out: wp.array,
        velocity: npt.ArrayLike | wp.array,
        dt: float,
    ) -> None:
        """Write ``q_out = integrate(q_out, velocity, dt)`` using mjwarp."""
        from mujoco_warp._src.forward import _next_position

        v = self._as_velocity_batch(velocity)
        self._dt_wp.assign(np.array([dt], dtype=np.float32))
        wp.launch(
            _next_position,
            dim=(self.nworld, self.model.njnt),
            inputs=[
                self._dt_wp,
                self.wp_model.jnt_type,
                self.wp_model.jnt_qposadr,
                self.wp_model.jnt_dofadr,
                q_out,
                v,
                1.0,
            ],
            outputs=[q_out],
        )

    def _as_velocity_batch(self, velocity: npt.ArrayLike | wp.array) -> wp.array:
        if isinstance(velocity, wp.array):
            if velocity.shape == (self.nv,):
                from .lie.kernels import broadcast_q

                wp.launch(
                    broadcast_q,
                    dim=self.nworld,
                    inputs=[velocity, self.nv],
                    outputs=[self._v_wp],
                )
                return self._v_wp
            if velocity.shape == (self.nworld, self.nv):
                return velocity
            raise ValueError(
                f"Expected velocity shape ({self.nv},) or ({self.nworld}, {self.nv}), "
                f"got {velocity.shape}"
            )
        from .interop import to_wp

        v_wp = to_wp(velocity, dtype=float, device=self.device)
        return self._as_velocity_batch(v_wp)

    @property
    def q(self) -> wp.array:
        """Device configuration, shape ``(nworld, nq)``."""
        return self.wp_data.qpos

    @property
    def nv(self) -> int:
        return self.model.nv

    @property
    def nq(self) -> int:
        return self.model.nq

    # Internal.

    def _assign_qpos(self, q: npt.ArrayLike | wp.array) -> None:
        if isinstance(q, wp.array):
            if q.shape == (self.nq,):
                wp.launch(
                    broadcast_q,
                    dim=self.nworld,
                    inputs=[q, self.nq],
                    outputs=[self.wp_data.qpos],
                )
            elif q.shape == (self.nworld, self.nq):
                wp.copy(self.wp_data.qpos, q)
            else:
                raise ValueError(f"Unexpected q shape {q.shape}")
            return

        # Optional host upload.
        q_wp = to_wp(q, dtype=float, device=self.device)
        if q_wp.shape == (self.nq,):
            wp.launch(
                broadcast_q,
                dim=self.nworld,
                inputs=[q_wp, self.nq],
                outputs=[self.wp_data.qpos],
            )
        elif q_wp.shape == (self.nworld, self.nq):
            wp.copy(self.wp_data.qpos, q_wp)
        else:
            raise ValueError(
                f"Expected q shape ({self.nq},) or ({self.nworld}, {self.nq}), "
                f"got {q_wp.shape}"
            )

    def _resolve_frame_id(self, frame_name: str, frame_type: str) -> int:
        key = (frame_name, frame_type)
        cached = self._frame_id_cache.get(key)
        if cached is not None:
            return cached
        if frame_type not in consts.SUPPORTED_FRAMES:
            raise exceptions.UnsupportedFrame(frame_type, consts.SUPPORTED_FRAMES)
        frame_id = mujoco.mj_name2id(
            self.model, consts.FRAME_TO_ENUM[frame_type], frame_name
        )
        if frame_id == -1:
            raise exceptions.InvalidFrame(
                frame_name=frame_name,
                frame_type=frame_type,
                model=self.model,
            )
        self._frame_id_cache[key] = frame_id
        return frame_id

    def _resolve_body_id(
        self, frame_name: str, frame_type: str, frame_id: int
    ) -> int:
        key = (frame_name, frame_type)
        cached = self._body_id_cache.get(key)
        if cached is not None:
            return cached
        if frame_type == "body":
            body_id = frame_id
        elif frame_type == "site":
            body_id = int(self.model.site_bodyid[frame_id])
        else:
            assert frame_type == "geom"
            body_id = int(self.model.geom_bodyid[frame_id])
        self._body_id_cache[key] = body_id
        return body_id

    def _frame_arrays(self, frame_type: str):
        if frame_type == "body":
            return self.wp_data.xpos, self.wp_data.xmat
        if frame_type == "site":
            return self.wp_data.site_xpos, self.wp_data.site_xmat
        assert frame_type == "geom"
        return self.wp_data.geom_xpos, self.wp_data.geom_xmat

    def _launch_frame_query(
        self,
        frame_type: str,
        frame_id: int,
        body_id: int,
        xpos_arr: wp.array,
    ) -> None:
        if frame_type == "body":
            wp.launch(
                fill_body_frame_query,
                dim=self.nworld,
                inputs=[xpos_arr, frame_id, body_id],
                outputs=[self._point_wp, self._body_wp],
            )
        elif frame_type == "site":
            wp.launch(
                fill_site_frame_query,
                dim=self.nworld,
                inputs=[xpos_arr, self.wp_model.site_bodyid, frame_id],
                outputs=[self._point_wp, self._body_wp],
            )
        else:
            wp.launch(
                fill_geom_frame_query,
                dim=self.nworld,
                inputs=[xpos_arr, self.wp_model.geom_bodyid, frame_id],
                outputs=[self._point_wp, self._body_wp],
            )
