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

## Solver backends — throughput vs tracking accuracy

Three interchangeable backends minimise the same weighted task cost behind one
`solve_and_integrate` API: **`dls`** (one damped Gauss-Newton step/tick, default),
**`lm`** (Levenberg-Marquardt), **`lbfgs`** (limited-memory BFGS). Default inner
iterations per call: `dls`=1, `lm`=2, `lbfgs`=5 (LM is a full Newton step and
converges in ~1–2 iterations; L-BFGS starts from steepest descent and needs a few
to ramp up). `|Δpos|` = world-0 tracked-frame distance to its target [m]; lower =
tighter tracking, and is device-independent (float32, same math on CPU/GPU).

Each scene runs under every backend below (`dls`/`lm`/`lbfgs`). `solves/s` = worlds
× steps ÷ wall time; GPU batched columns use CUDA-graph capture (`dls`/`lm`; `lbfgs`
is eager). Two regimes:

- **`panda`** (fixed base, gentle target) — DLS already converges each tick, so LM
  only *matches* its accuracy (at ~3× the per-call cost) and L-BFGS trails.
- **`g1`** (floating base, larger target motion) — LM tracks **~3× tighter** than
  DLS (`|Δpos|` 9.6e-3 → 3.2e-3) because it takes full undamped Newton steps; on
  GPU at 4096 worlds that costs only ~1.3× the throughput. L-BFGS underperforms on
  this stiff floating-base problem (its line search stalls; more `--iters` doesn't
  help). Optimizer backends also add robustness where a plain GN step overshoots —
  ill-conditioned Jacobians or unreachable targets (an out-of-reach target winds an
  undamped DLS arm through many revolutions while LM settles at the closest pose).

CPU = Apple Silicon (eager); GPU = RTX 4070 Ti SUPER (1 world eager, batched CUDA graph).

### `panda` (fixed base, `FrameTask` + `PostureTask`, nv=9)

| solver | iters | CPU 1w | CPU 256w | GPU 1w | GPU 4096w | `\|Δpos\|` mean | `\|Δpos\|` max |
|---|--:|--:|--:|--:|--:|--:|--:|
| dls   | 1 | 2,861 | 154,987 | 2,016 | 9,610,689 | 2.6e-4 | 8.4e-4 |
| lm    | 2 |   884 |  48,418 |   623 | 3,672,702 | 2.4e-4 | 8.2e-4 |
| lbfgs | 5 |   118 |   8,148 |    84 |   344,084 | 3.3e-3 | 8.2e-3 |

### `g1` (floating base, pelvis `FrameTask` + `PostureTask` + `ComTask`, nv=49)

| solver | iters | CPU 1w | CPU 256w | GPU 1w | GPU 4096w | `\|Δpos\|` mean | `\|Δpos\|` max |
|---|--:|--:|--:|--:|--:|--:|--:|
| dls   | 1 | 2,034 | 11,224 |   678 | 371,657 | 9.6e-3 | 1.9e-2 |
| lm    | 2 |   589 |  4,116 |   333 | 278,111 | **3.2e-3** | **4.2e-3** |
| lbfgs | 5 |    98 |  1,662 |    75 | 223,299 | 4.1e-2 | 6.0e-2 |

```bash
# any scene x any backend, throughput + tracking accuracy
uv run python benchmarks/bench_solvers.py panda --nworld 256                    # CPU
uv run python benchmarks/bench_solvers.py g1 --nworld 4096 --graph --device cuda:0
```

### LM throughput sweep (`panda`, GPU, CUDA graph, iters=2)

| worlds | solves/s | µs/solve |
|-------:|---------:|---------:|
| 1      |     2,998 |  333.6 |
| 64     |   149,294 |   6.70 |
| 1024   | 1,290,000 |   0.78 |
| 4096   | 3,870,652 |   0.26 |
| 16384  | 5,047,604 |   0.20 |

```bash
uv run python benchmarks/bench_ik.py panda --solver lm --graph --batches 1 64 1024 4096 16384
```

_Numbers will shift with GPU model, batch, task stack, and warp/mujoco-warp versions. Re-run on
your target hardware._
