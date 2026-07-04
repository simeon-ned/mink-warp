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

_Numbers will shift with GPU model, batch, task stack, and warp/mujoco-warp versions. Re-run on
your target hardware._
