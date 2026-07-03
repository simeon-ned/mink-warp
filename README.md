# mink-warp

Batched differential inverse kinematics on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), with a [Mink](https://github.com/kevinzakka/mink)-shaped API.

Mink is the source of truth for task definitions, Jacobian conventions (body-frame twists), and Lie-group error formulas. mink-warp runs the same math over `nworld` configurations on device via Warp / MuJoCo Warp.

## Status

Early development. Device-native pipeline: FK / Jacobians / tasks, normal-equation assembly, **batched Cholesky solve**, and **mjwarp position integrate**. Batched Panda demo via mjviser. B=1 parity tests against Mink (unconstrained soft tasks).

## Installation

```bash
uv sync --extra dev
```

## Design

- **API**: Mink names (`Configuration`, `FrameTask`, `set_target`, …), Newton-style device arrays.
- **Hot path**: `wp.array` only — targets, `q`, errors, Jacobians. Prefer `wp.copy` / update kernels (no host round-trip).
- **Host boundary**: NumPy / `SE3` accepted as optional one-shot uploads via `set_target(...)` or `mink_warp.to_wp(...)`. Use `.numpy()` only in tests / debugging.
- **Jacobians**: Body-frame, matching Mink (`Rᵀ @ jac_world`, frame-task `J = -jlog(T_tb) @ jac_body`).
- **Dependencies**: `mujoco`, `mujoco-warp`, `warp-lang`, `numpy`. No mjlab / Newton / mjinx.

```python
cfg = mink_warp.Configuration(model, nworld=B, device="cuda")
cfg.update(q_wp)  # wp.array (B, nq)

frame = mink_warp.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
frame.set_target(poses_wp)                   # wp.array (B, 7)
posture = mink_warp.PostureTask(model, cost=1e-2)
posture.set_target_from_configuration(cfg)

solver = mink_warp.IKSolver(cfg)
v = solver.solve([frame, posture], dt=0.01)  # wp.array (B, nv)
cfg.integrate_inplace(v, dt)
```

Batched Franka Panda demo (Mink assets + mjviser):

```bash
uv sync --extra dev --extra examples
uv run examples/batched_panda_ik.py
```

Panda MJCF and meshes live under `examples/franka_emika_panda/` (vendored from Mink).

## License

Apache-2.0
