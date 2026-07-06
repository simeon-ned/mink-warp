# mink-warp

<p align="center">
  <img src="docs/_static/cassie_equality.gif" alt="1024 Cassie robots with equality-constraint IK (examples/03_equality_cassie.py)" width="800">
</p>

[![Build](https://img.shields.io/github/actions/workflow/status/simeon-ned/mink-warp/ci.yml?branch=main)](https://github.com/simeon-ned/mink-warp/actions)
[![Documentation](https://img.shields.io/github/actions/workflow/status/simeon-ned/mink-warp/docs.yml?branch=main&label=docs)](https://simeon-ned.github.io/mink-warp/)
[![License](https://img.shields.io/github/license/simeon-ned/mink-warp)](https://github.com/simeon-ned/mink-warp/blob/main/LICENSE)
[![PyPI](https://img.shields.io/pypi/v/mink-warp)](https://pypi.org/project/mink-warp/)
[![PyPI downloads](https://img.shields.io/pypi/dm/mink-warp?color=blue)](https://pypistats.org/packages/mink-warp)

**mink-warp** is batched differential inverse kinematics on [MuJoCo Warp](https://github.com/google-deepmind/mujoco_warp), with a [Mink](https://github.com/kevinzakka/mink)-shaped API. Given a robot configuration and a stack of task-space objectives, it computes joint velocities for **many parallel worlds** on the GPU — the same control-loop niche as Mink, scaled to `nworld`.

Features include:

* **Mink-compatible tasks** — `FrameTask`, `RelativeFrameTask`, `PostureTask`, `ComTask`, `DampingTask`, `EqualityConstraintTask`, soft `JointLimitTask`;
* **Hard limits** — `ConfigurationLimit`, `VelocityLimit`, `CollisionAvoidanceLimit`, `LinearInequalityLimit` via GPU ADMM (`ConstrainedSolver`);
* **Device-native hot path** — FK, Jacobians, residual assembly, and linear solves as Warp kernels on `wp.array` buffers;
* **Interchangeable solvers** — damped least squares (default), Levenberg–Marquardt, L-BFGS, constrained ADMM;
* **Optional CUDA graph capture** for fixed task sets in real-time loops;
* **Runnable mjviser demos** — numbered `examples/01_…` through `05_…` (Panda → UR5e → Cassie → dual iiwa → G1).

For usage, concepts, and API reference, see the [documentation](https://simeon-ned.github.io/mink-warp/).

## Installation

From PyPI:

```bash
uv add mink-warp
```

Or clone and run locally:

```bash
git clone https://github.com/simeon-ned/mink-warp.git && cd mink-warp
uv sync --extra dev --extra examples
```

Requires Python 3.10+, MuJoCo 3.8+, [mujoco-warp](https://github.com/google-deepmind/mujoco_warp), and [NVIDIA Warp](https://github.com/NVIDIA/warp). A CUDA-capable GPU is recommended for batched workloads.

## Usage

```python
import mink_warp as mw

cfg = mw.Configuration(model, nworld=512, device="cuda")

frame = mw.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
frame.set_target_from_configuration(cfg)
posture = mw.PostureTask(model, cost=1e-2)
posture.set_target_from_configuration(cfg)

solver = mw.IKSolver(cfg)
solver.solve_and_integrate([frame, posture], dt=0.01, use_graph=True)

# Hard joint limits (Mink limits=None equivalent):
v = mw.solve_ik(cfg, [frame, posture], dt=0.01, limits=None)
```

## Examples

Examples are ordered by increasing complexity:

```bash
uv run examples/01_panda_ik.py                  # FrameTask + soft limits, CUDA graph
uv run examples/02_constrained_ur5e.py          # hard config / velocity / collision limits
uv run examples/03_equality_cassie.py           # closed-chain EqualityConstraintTask
uv run examples/04_self_collision_dual_iiwa.py  # dual Kuka, inter-arm collision
uv run examples/05_relative_frame_g1.py         # G1 RelativeFrameTask + collision
```

Assets are vendored under `examples/` from Mink (Panda, UR5e, Cassie, Kuka iiwa14, Unitree G1). See [examples](examples/) and the [examples guide](https://simeon-ned.github.io/mink-warp/source/examples.html) in the docs.

## Benchmarks

Batched throughput and CPU-vs-Mink parity (`benchmarks/README.md`):

```bash
uv run python benchmarks/bench_ik.py             # solves/sec vs batch size
uv run python benchmarks/bench_constrained.py    # hard limits vs mink daqp
uv run python benchmarks/bench_parity.py         # agreement with mink (oracle)
```

See `benchmarks/RESULTS.md` for sample numbers.

## Acknowledgements

mink-warp mirrors the API and conventions of [Mink](https://github.com/kevinzakka/mink), which is a MuJoCo port of [Pink](https://github.com/stephane-caron/pink) (Pinocchio). The Lie-group helpers and task Jacobian conventions follow the same lineage as Mink and Pink.

Implementation patterns also draw from GPU IK / physics stacks in the MuJoCo Warp ecosystem and from [Newton](https://github.com/newton-physics/newton) (NVIDIA + Google DeepMind), though mink-warp is **not** a wrapper around Newton — it targets **differential** IK with a Mink-shaped API on `mujoco.MjModel` + mjwarp.


## Citation

If you use mink-warp in your research, please cite it as follows:

```bibtex
@software{nedelchev2026minkwarp,
  author       = {Nedelchev, Simeon and Domrachev, Ivan},
  title        = {{mink-warp: Batched differential inverse kinematics on MuJoCo Warp}},
  year         = {2026},
  version      = {0.1.0},
  url          = {https://github.com/simeon-ned/mink-warp},
  repository   = {https://github.com/simeon-ned/mink-warp},
  license      = {Apache-2.0},
}
```

Also available as [CITATION.cff](CITATION.cff) and [CITATION.bib](CITATION.bib). Method papers: [docs/references.bib](docs/references.bib).


## License

Apache-2.0 — see [LICENSE](LICENSE).
