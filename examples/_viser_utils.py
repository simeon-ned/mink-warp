"""Shared helpers for mjviser examples."""

from __future__ import annotations

import math

import mujoco
import mink_warp as mw
from mjviser import ViserMujocoScene


def subtree_collision_geom_ids(model: mujoco.MjModel, root_body_name: str) -> list[int]:
    """Collision geom ids on ``root_body_name`` and its descendants."""
    root_id = model.body(root_body_name).id
    ids: list[int] = []
    for geom_id in range(model.ngeom):
        if not (model.geom_contype[geom_id] and model.geom_conaffinity[geom_id]):
            continue
        body_id = model.geom_bodyid[geom_id]
        while True:
            if body_id == root_id:
                ids.append(geom_id)
                break
            if body_id <= 0:
                break
            body_id = model.body_parentid[body_id]
    return ids


def grid_origins(n: int, spacing: float):
    import numpy as np

    cols = math.ceil(math.sqrt(n))
    rows = math.ceil(n / cols)
    origins = np.zeros((n, 3), dtype=np.float32)
    for i in range(n):
        r, c = divmod(i, cols)
        origins[i, 0] = (c - 0.5 * (cols - 1)) * spacing
        origins[i, 1] = (r - 0.5 * (rows - 1)) * spacing
    return origins


def sync_scene(
    scene: ViserMujocoScene,
    cfg: mw.Configuration,
    origins,
) -> None:
    """Push current FK state to mjviser."""
    xpos = cfg.wp_data.xpos.numpy().copy()
    xpos += origins[:, None, :]
    scene.update_from_arrays(xpos, cfg.wp_data.xmat.numpy(), qpos=cfg.q.numpy())


def warmup_solver(solver, tasks, dt: float, *, label: str = "solver") -> None:
    """One dry run so CUDA kernel compile happens before the viewer opens."""
    print(f"Warming up {label} (first step may compile GPU kernels)...")
    solver.solve_and_integrate(tasks, dt, iterations=1, use_graph=False)
    print("Warmup done.")
