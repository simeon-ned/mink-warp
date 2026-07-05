# MINK-WARP

Batched differential inverse kinematics on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), with a [Mink](https://github.com/kevinzakka/mink)-shaped API.

## Status

Device-native pipeline: FK / Jacobians / tasks, normal-equation assembly, batched Cholesky solve, **constrained ADMM** (box + general inequalities), mjwarp position integrate, optional CUDA graph capture.

**Tasks:** `FrameTask`, `RelativeFrameTask`, `PostureTask`, `DampingTask`, `ComTask`, `EqualityConstraintTask`, `JointLimitTask` (alias `ConfigurationLimitTask`).

**Hard limits:** `ConfigurationLimit`, `VelocityLimit`, `CollisionAvoidanceLimit`, `LinearInequalityLimit` via `ConstrainedSolver` or `solve_ik(..., limits=…)` (Mink QP equivalent).

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
  limits/            # ConfigurationLimit, VelocityLimit, LinearInequalityLimit
  solvers/           # DLS, LM, L-BFGS, ConstrainedSolver (ADMM)
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

# Hard joint limits (Mink limits=None equivalent):
v = mink_warp.solve_ik(cfg, [frame, posture], dt=0.01, limits=None)
```

## Demos

```bash
uv run examples/batched_panda_ik.py
uv run examples/batched_g1_ik.py
uv run examples/constrained_ur5e.py      # batched hard limits (config + collision + velocity)
uv run examples/self_collision_dual_panda.py  # batched inter-arm self-collision
uv run examples/equality_cassie.py       # batched closed-chain equality task
uv run examples/relative_frame_g1.py     # batched RelativeFrameTask + collision
```

Assets: `examples/franka_emika_panda/`, `examples/unitree_g1/`, `examples/universal_robots_ur5e/`, `examples/agility_cassie/` (vendored from Mink).

## Documentation

```bash
uv sync --extra dev --group docs
make docs        # docs/_build/index.html
make docs-watch  # live reload
```

User guide, concepts, and API reference: [docs/](docs/) (Sphinx, Mink-style layout).

## Benchmarks

Batched throughput and CPU-vs-mink accuracy parity (see `benchmarks/README.md`):

```bash
uv sync --extra dev
uv run python benchmarks/bench_ik.py            # solves/sec vs batch size
uv run python benchmarks/bench_constrained.py  # hard limits vs mink daqp
uv run python benchmarks/bench_parity.py        # agreement with mink (oracle)
```

See `benchmarks/RESULTS.md` for current CPU + GPU numbers.

## License

Apache-2.0
