
Performance and accuracy harness, mirroring `mink/benchmarks/` but batched.
mink measures single-world per-step latency; mink-warp additionally measures
**throughput** (IK solves/sec vs world count) and **parity** (does the batched
GPU solve agree with mink's CPU QP).

Scenes are defined once in `scenes.py` and restricted to the soft, unconstrained
task stack that mink-warp supports today with math identical to mink
(`FrameTask`, `PostureTask`, `ComTask`) — so the same trajectory drives both the
throughput sweep and the CPU-vs-GPU parity check.

**Current measured numbers**: see [`RESULTS.md`](RESULTS.md).

## Scripts

| Script | Measures |
|--------|----------|
| `bench_ik.py`     | Throughput sweep: per-step wall time, µs/solve, solves/sec across batch sizes (eager or CUDA graph). `--solver {dls,lm,lbfgs}`, `--iters`. |
| `bench_solvers.py`| Solver comparison: throughput **and** end-effector tracking accuracy (`|Δpos|`) per backend on the same trajectory. |
| `bench_constrained.py`| Constrained solver: throughput + **max joint-limit violation** vs `dls`, and `--check` accuracy sweep over `admm_iters` vs mink `daqp`+`ConfigurationLimit`. `--solvers` accepts `constrained` (fast box path) and `constrained-ineq` (general `G Δq ≤ h` path). |
| `bench_parity.py` | Accuracy: replays a trajectory through mink (CPU, `daqp`, `limits=[]`) and mink-warp (world 0), reports tangent-velocity `|Δv|` and configuration `|Δq|`. |
| `bench_tasks.py`  | Closed-kinematics (`EqualityConstraintTask`, Cassie) + collision-avoidance (`CollisionAvoidanceLimit`, dual iiwa): throughput, **host-assembly component split** (`hot %`), device-broadphase skip-rate (`bp-skip`), and `--check` parity vs mink (equality `e`/`J`, collision `G`/`h`). `--collision-motion {dense,sparse}`. These tasks' hot path is host MuJoCo, not the GPU solve. |
| `common.py`       | `summarize` (ported verbatim from mink) + batched helpers (`throughput`, `sync`). |
| `scenes.py`       | Batched scene registry: `panda` (fixed base, parity-safe), `g1` (floating base, throughput). |

### Solver backends

All three minimise the same weighted task cost and share
`solve_and_integrate(tasks, dt, iterations=...)`:

| `--solver` | Method | Per call |
|-----------|--------|----------|
| `dls` (default) | damped Gauss-Newton (Mink's differential step) | one step |
| `lm`  | Levenberg-Marquardt: adaptive damping + trust-region accept/reject | `iters` steps |
| `lbfgs` | limited-memory BFGS: two-loop recursion + parallel line search | `iters` steps |
| `constrained` | box-ADMM enforcing **hard** joint limits (`lo ≤ Δq ≤ hi`); factor `H+ρI` once, `admm_iters` clip+dual steps. Also solves general `G Δq ≤ h` via factor-once OSQP-ADMM (`M=H+σI+ρGᵀG`) when given inequality limits | `admm_iters` steps |

## Run

```bash
uv sync --extra dev                       # installs mink (parity oracle) + daqp

# Throughput sweep (defaults: panda, batches 1..4096)
uv run python benchmarks/bench_ik.py
uv run python benchmarks/bench_ik.py g1 --batches 1 64 1024 4096
uv run python benchmarks/bench_ik.py --graph --save gpu.json     # CUDA graph capture
uv run python benchmarks/bench_ik.py --compare cpu.json gpu.json # A/B speedup

# Solver backends
uv run python benchmarks/bench_ik.py --solver lm --graph          # LM throughput
uv run python benchmarks/bench_solvers.py                         # dls/lm/lbfgs: speed vs accuracy
uv run python benchmarks/bench_solvers.py g1 --motion aggressive  # fast target: LM tracks tighter
uv run python benchmarks/bench_solvers.py g1 --motion aggressive --nworld 4096 --graph --device cuda:0

# Accuracy parity vs mink (DLS backend)
uv run python benchmarks/bench_parity.py            # PASS/FAIL at --tol
uv run python benchmarks/bench_parity.py --steps 500 --tol 2e-3

# Closed-kinematics + collision-avoidance tasks (host-assembly hot path)
uv run python benchmarks/bench_tasks.py --check                       # parity vs mink
uv run python benchmarks/bench_tasks.py --profile --device cuda:0 --nworld 256 1024 4096
uv run python benchmarks/bench_tasks.py dual_iiwa --profile --collision-motion sparse --device cuda:0
```

## Notes

- **float32.** mink-warp solves in float32; mink in float64. Peak `|Δv|` parity is
  therefore ~1e-3 (mean ~1e-4), so `bench_parity` defaults to `--tol 5e-3`.
- **Solver choice.** With a slow (`--motion gentle`) target `dls` (1 GN step/tick)
  already converges, so `lm` only matches it. With a fast (`--motion aggressive`)
  target a single step lags and `lm` tracks tighter — up to ~8× on `g1` (`|Δpos|`
  3.8e-2 → 4.9e-3) — by taking full undamped Newton steps. `lm` converges in ~1–2
  iterations (default `iters=2`); `lbfgs` starts from steepest descent, needs a few
  (default 5), and stalls on stiff problems like `g1`. Optimizer backends also add
  robustness where a plain GN step overshoots (ill-conditioned Jacobians,
  near-singular or unreachable targets). Raise `--iters` for harder problems. `lbfgs`
  runs eager only (per-candidate line search is not graph-capturable).
- **CUDA graph.** `--graph` only engages on a CUDA device (and `dls`/`lm`); CPU runs eager.
- **Throughput scaling.** On CPU, solves/sec rises with batch until cores saturate;
  the win is on GPU, where thousands of worlds solve in one launch.
- Scenes are intentionally soft-only. Hard limits / geometric SE3 tasks land via
  the roadmap epics (`../ROADMAP.md`); new parity-safe scenes plug into `SCENES`.
