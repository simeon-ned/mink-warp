"""Position integration via MuJoCo Warp."""

from __future__ import annotations

import warp as wp

from mujoco_warp._src.forward import _next_position


def integrate_qpos(
    *,
    q_in: wp.array,
    q_out: wp.array,
    velocity: wp.array,
    dt_buf: wp.array,
    jnt_type: wp.array,
    jnt_qposadr: wp.array,
    jnt_dofadr: wp.array,
    nworld: int,
    njnt: int,
) -> None:
    """``q_out = integrate(q_in, velocity, dt)`` using mjwarp ``_next_position``.

    ``q_in`` and ``q_out`` must be distinct buffers (required for CUDA graphs).
    ``dt_buf`` must already hold the timestep (do not host-assign inside a graph).
    """
    wp.launch(
        _next_position,
        dim=(nworld, njnt),
        inputs=[
            dt_buf,
            jnt_type,
            jnt_qposadr,
            jnt_dofadr,
            q_in,
            velocity,
            1.0,
        ],
        outputs=[q_out],
    )
