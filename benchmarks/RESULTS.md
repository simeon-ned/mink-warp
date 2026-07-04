# mink-warp benchmark results

Reference numbers from the harness in this directory. Reproduce with the commands
under each table. Throughput = IK **solves per second** (worlds × steps ÷ wall time);
`us/solve` = per-step wall time ÷ world count.

## Environment

| | CPU run | GPU run |
|---|---|---|
| Host | Apple Silicon laptop (dev machine) | NVIDIA **RTX 4070 Ti SUPER** (16 GB, driver 580.126) |
| Backend | warp 1.14 **CPU** (no CUDA) | warp 1.14 **CUDA**, `device=cuda:0` |
| Solve mode | eager | eager + **CUDA graph** (`--graph`) |
| dtype | float32 | float32 |
| dt | 10 ms (100 Hz) | 10 ms |

## Throughput — `panda` scene (fixed base, `FrameTask` + `PostureTask`, nv=9)

| worlds | CPU solves/s | CPU µs/solve | GPU solves/s | GPU µs/solve | GPU/CPU |
|-------:|-------------:|-------------:|-------------:|-------------:|--------:|
| 1      |        2,931 |       341.2  |        6,708 |       149.1  |   2.3×  |
| 16     |       39,794 |        25.1  |      102,634 |         9.74 |   2.6×  |
| 64     |       96,154 |        10.4  |      362,228 |         2.76 |   3.8×  |
| 256    |      163,529 |         6.12 |      875,620 |         1.14 |   5.4×  |
| 1024   |      195,326 |         5.12 |    3,309,956 |         0.30 |  16.9×  |
| 4096   |          —   |         —    |   10,245,969 |         0.10 |    —    |
| 16384  |          —   |         —    |   16,091,936 |         0.06 |    —    |

```bash
uv run python benchmarks/bench_ik.py panda --batches 1 16 64 256 1024 --steps 120       # CPU
uv run python benchmarks/bench_ik.py panda --graph --batches 1 16 64 256 1024 4096 16384 # GPU
```

## Throughput — `g1` scene (floating base, pelvis `FrameTask` + `PostureTask` + `ComTask`, nv=49)

| worlds | CPU solves/s | CPU µs/solve | GPU solves/s | GPU µs/solve | GPU/CPU |
|-------:|-------------:|-------------:|-------------:|-------------:|--------:|
| 1      |        2,101 |       476.0  |          715 |      1399.1  |   0.34× |
| 16     |        8,846 |       113.1  |        9,350 |       106.9  |   1.06× |
| 64     |       10,639 |        94.0  |       10,430 |        95.9  |   0.98× |
| 256    |       11,186 |        89.4  |       24,547 |        40.7  |   2.2×  |
| 1024   |          —   |         —    |       96,791 |        10.3  |    —    |
| 4096   |          —   |         —    |      372,495 |         2.68 |    —    |

```bash
uv run python benchmarks/bench_ik.py g1 --batches 1 16 64 256 --steps 60                 # CPU
uv run python benchmarks/bench_ik.py g1 --graph --batches 1 16 64 256 1024 4096 --steps 150 # GPU
```

## Single-environment baseline (1 world) — vs mink

| library | device | solves/s | µs/solve |
|---|---|-------:|-------:|
|mink| CPU | 18,371 | 54.4 |
| mink-warp | CPU | 2,977 | 335.9 |
| mink-warp | GPU | 6,708 | 149.1 |

```bash
uv run python benchmarks/bench_parity.py panda --steps 200   # prints this baseline + the parity below
```

## Accuracy parity vs mink (`panda`, oracle = mink CPU `daqp`, `limits=[]`)

| metric | CPU mink-warp | GPU mink-warp |
|---|---|---|
| tangent-velocity `\|Δv\|` mean | 2.59e-4 | 2.67e-4 |
| `\|Δv\|` max | 2.52e-3 | 2.52e-3 |
| `\|Δv\|` p99 | 2.43e-3 | 2.45e-3 |
| configuration `\|Δq\|` max | 2.56e-5 | 2.55e-5 |

```bash
uv run python benchmarks/bench_parity.py panda --steps 200        # CPU / GPU
```

## Solver backends — throughput vs tracking accuracy (`panda`)

Three interchangeable backends minimise the same weighted task cost behind one
`solve_and_integrate` API: **`dls`** (one damped Gauss-Newton step/tick, default),
**`lm`** (Levenberg-Marquardt), **`lbfgs`** (limited-memory BFGS). Optimizer
backends run `iters` inner iterations per call (default 5). `|Δpos|` = world-0
end-effector distance to its moving target [m]; lower = tighter tracking.

On this gentle circle target DLS already converges each tick, so `lm` matches its
tracking at ~8× the per-call cost and `lbfgs` (iteration-bounded at 5) trails —
the optimizer backends earn their cost on **hard / far / redundant** starts where a
single GN step diverges. Raising `--iters` tightens `|Δpos|` (e.g. `lbfgs --iters 12`
→ `|Δpos|` mean 3.4e-4).

**CPU** (Apple Silicon, eager, float32):

| solver | iters | 1 world (solves/s) | 256 worlds (solves/s) | µs/solve @256 | `\|Δpos\|` mean | `\|Δpos\|` max |
|---|--:|--:|--:|--:|--:|--:|
| dls   | 1 |  2,973 | 159,116 |   6.3 | 2.3e-4 | 7.4e-4 |
| lm    | 5 |    366 |  20,005 |  50.0 | 2.5e-4 | 8.0e-4 |
| lbfgs | 5 |    122 |   8,400 | 119.1 | 3.4e-3 | 7.5e-3 |

**GPU** (RTX 4070 Ti SUPER; 1 world eager, ≥256 worlds CUDA graph — `lbfgs` eager):

| solver | iters | 1 world (solves/s) | 256 (solves/s) | 4096 (solves/s) | `\|Δpos\|` mean | `\|Δpos\|` max |
|---|--:|--:|--:|--:|--:|--:|
| dls   | 1 | 2,068 | 877,917 | 9,677,403 | 2.1e-4 | 7.4e-4 |
| lm    | 5 |   263 | 151,055 | 1,641,489 | 2.2e-4 | 8.0e-4 |
| lbfgs | 5 |    87 |  21,739 |   344,379 | 3.5e-3 | 1.0e-2 |

```bash
uv run python benchmarks/bench_solvers.py --nworld 1 --steps 200                     # CPU 1 world
uv run python benchmarks/bench_solvers.py --nworld 256 --graph --device cuda:0        # GPU batched
uv run python benchmarks/bench_solvers.py --nworld 4096 --graph --device cuda:0
```

### LM throughput sweep (`panda`, GPU, CUDA graph, iters=5)

| worlds | solves/s | µs/solve |
|-------:|---------:|---------:|
| 1      |     1,353 |    739.0 |
| 64     |    65,961 |     15.2 |
| 1024   |   545,998 |     1.83 |
| 4096   | 1,650,739 |     0.61 |
| 16384  | 2,123,132 |     0.47 |

```bash
uv run python benchmarks/bench_ik.py panda --solver lm --graph --batches 1 64 1024 4096 16384
```

_Numbers will shift with GPU model, batch, task stack, and warp/mujoco-warp versions. Re-run on
your target hardware._
