# mink-warp

Batched differential inverse kinematics on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), with a [Mink](https://github.com/kevinzakka/mink)-shaped API.

## Status

Device-native pipeline: FK / Jacobians / tasks, normal-equation assembly, batched Cholesky solve, mjwarp position integrate, optional CUDA graph capture.

**Tasks:** `FrameTask`, `PostureTask`, `DampingTask`, `ComTask`, `JointLimitTask` (alias `ConfigurationLimitTask`).

**Layout:**

```text
mink_warp/
  configuration.py   # batched mjwarp state
  solve_ik.py        # IKSolver (+ optional CUDA graph)
  integrate.py       # mjwarp position integrate wrapper
  interop.py         # optional host → device upload
  lie/               # SE3 / SO3 (host) + wp_ops (device)
  kernels/           # all Warp kernels
  tasks/             # Task / TargetedTask + concrete tasks
```

## Install

```bash
uv sync --extra dev --extra examples
```

## Usage

```python
cfg = mink_warp.Configuration(model, nworld=B, device="cuda")
frame = mink_warp.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
frame.set_target(poses_wp)  # wp.array (B, 7)
posture = mink_warp.PostureTask(model, cost=1e-2)
posture.set_target_from_configuration(cfg)

solver = mink_warp.IKSolver(cfg)
solver.solve_and_integrate([frame, posture], dt=0.01, use_graph=True)
```

## Demos

```bash
uv run examples/batched_panda_ik.py
uv run examples/batched_g1_ik.py
```

Assets: `examples/franka_emika_panda/`, `examples/unitree_g1/` (vendored from Mink).

## License

Apache-2.0
