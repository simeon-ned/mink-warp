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

Each scene (`panda`, `g1`) runs under every backend (`dls`/`lm`/`lbfgs`) at two
target speeds (`--motion gentle|aggressive`). `solves/s` = worlds × steps ÷ wall
time; GPU batched columns use CUDA-graph capture (`dls`/`lm`; `lbfgs` is eager).

The distinction is **how fast the target moves per control tick**:

- **Gentle** (default) — the target creeps, so a single Gauss-Newton step per tick
  already converges. Every backend tracks equally; LM/L-BFGS only cost throughput.
- **Aggressive** — the target moves ~10× faster, so one linear step lags and LM's
  re-linearization tracks visibly tighter (up to ~8× on `g1`). This is where the
  optimizer backends earn their per-call cost.

Across both, the optimizer backends also add robustness where a plain GN step
overshoots — ill-conditioned Jacobians or unreachable targets (an out-of-reach
target winds an undamped DLS arm through many revolutions while LM settles at the
closest pose). L-BFGS trails on the stiff floating-base `g1` (its line search
stalls; more `--iters` doesn't help), and suits well-conditioned redundant arms.

CPU = Apple Silicon (eager); GPU = RTX 4070 Ti SUPER (1 world eager, batched CUDA graph).

### Gentle motion (default trajectory)

Target creeps ~3e-3 m/tick (`panda`) — one GN step already converges, so every
backend tracks equally; the optimizer backends only cost throughput here.

`panda` (fixed base, `FrameTask` + `PostureTask`, nv=9):

| solver | iters | CPU 1w | CPU 256w | GPU 1w | GPU 4096w | `\|Δpos\|` mean | `\|Δpos\|` max |
|---|--:|--:|--:|--:|--:|--:|--:|
| dls   | 1 | 2,861 | 154,987 | 2,016 | 9,610,689 | 2.6e-4 | 8.4e-4 |
| lm    | 2 |   884 |  48,418 |   623 | 3,672,702 | 2.4e-4 | 8.2e-4 |
| lbfgs | 5 |   118 |   8,148 |    84 |   344,084 | 3.3e-3 | 8.2e-3 |

`g1` (floating base, pelvis `FrameTask` + `PostureTask` + `ComTask`, nv=49):

| solver | iters | CPU 1w | CPU 256w | GPU 1w | GPU 4096w | `\|Δpos\|` mean | `\|Δpos\|` max |
|---|--:|--:|--:|--:|--:|--:|--:|
| dls   | 1 | 2,034 | 11,224 |   678 | 371,657 | 9.6e-3 | 1.9e-2 |
| lm    | 2 |   589 |  4,116 |   333 | 278,111 | **3.2e-3** | **4.2e-3** |
| lbfgs | 5 |    98 |  1,662 |    75 | 223,299 | 4.1e-2 | 6.0e-2 |

### Aggressive motion (`--motion aggressive`)

Target moves ~10× faster (`panda` ~3e-2 m/tick, `g1` ~1.4e-2 m/tick). Now one GN
step lags and LM's re-linearization pays off — dramatically on `g1`, where LM at
iters=2 tracks **~8× tighter than DLS** for ~1.3× the GPU throughput at 4096
worlds. L-BFGS still trails on these stiff/fast problems. Throughput is unchanged
from gentle (same work per tick); only `|Δpos|` differs, so only it is shown.

`panda`:

| solver | iters | GPU 4096w (solves/s) | `\|Δpos\|` mean | `\|Δpos\|` max |
|---|--:|--:|--:|--:|
| dls   | 1 | 9,666,182 | 1.53e-2 | 7.49e-2 |
| lm    | 2 | 3,718,200 | **1.37e-2** | **5.19e-2** |
| lbfgs | 5 |   341,341 | 3.90e-2 | 9.95e-2 |

`g1`:

| solver | iters | GPU 4096w (solves/s) | `\|Δpos\|` mean | `\|Δpos\|` max |
|---|--:|--:|--:|--:|
| dls   | 1 | 371,722 | 3.77e-2 | 1.12e-1 |
| lm    | 2 | 278,219 | **4.87e-3** | **9.25e-3** |
| lbfgs | 5 | 223,687 | 7.75e-2 | 1.20e-1 |

```bash
# any scene x any backend x motion, throughput + tracking accuracy
uv run python benchmarks/bench_solvers.py g1 --nworld 256                                    # CPU, gentle
uv run python benchmarks/bench_solvers.py g1 --motion aggressive --nworld 4096 --graph --device cuda:0
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

## Constrained solver — hard joint limits (box-ADMM)

The `constrained` backend enforces hard joint **limits** (mink's
`ConfigurationLimit` / `VelocityLimit`) by solving, per world, the same QP as
mink — `min ½ ΔqᵀHΔq + cᵀΔq s.t. lo ≤ Δq ≤ hi` — with OSQP-style box-ADMM:
factor `M = H + ρI` once with the existing tile Cholesky, then `admm_iters`
cached-solve + box-clip + dual-update steps, returning the projected step, which
lies **inside the box at every iteration**. So limits are never violated, even
at `admm_iters=1` or when the target drives the arm hard into a bound — unlike
the soft `JointLimitTask` penalty. `ρ = ρ_scale·√(min·max diag H)` self-scales
across scenes; default `admm_iters=30`, `ρ_scale=1.0`.

When a target is ramped straight past the joint limits, the unconstrained `dls`
step follows it into the forbidden zone while the constrained solver clamps at
the limit — its per-step overshoot is exactly 0:

![Constrained IK never surpasses joint limits](figures/limits_demo.png)

**Accuracy vs mink** (`daqp` + `ConfigurationLimit`, world 0, lockstep): `|Δv|`
reaches the float32 parity floor (~6e-4, same as the unconstrained DLS floor) by
`admm_iters≈30`. Feasibility is exact at *any* iteration count.

| admm_iters | `\|Δv\|` mean | `\|Δv\|` max | max limit violation |
|--:|--:|--:|--:|
| 20 | 1.33e-3 | 2.90e-3 | **0** |
| 30 | 6.12e-4 | 2.53e-3 | **0** |
| 40 | 5.92e-4 | 2.53e-3 | **0** |

**Throughput vs `dls`** (GPU RTX 4070 Ti SUPER, CUDA graph, aggressive target
`--amp-scale 3` that pushes joints into their limits). `max viol` = worst joint-
limit overshoot over the run: `dls` (no limits) blows through by ~1–2 rad; the
constrained solver holds it at **0** for a small throughput cost that shrinks
with nv (the one-off factor amortizes over the batch and the K cheap solves are
relatively free on the larger model).

| scene | worlds | dls solves/s | constrained solves/s | overhead | dls max viol | constrained max viol |
|---|--:|--:|--:|--:|--:|--:|
| panda (nv=9)  | 256  |  1,038,419 |   959,042 | 1.08× | 0.99 rad | **0** |
| panda (nv=9)  | 1024 |  3,891,933 | 3,466,640 | 1.12× | 0.99 rad | **0** |
| panda (nv=9)  | 4096 | 11,843,821 | 8,909,478 | 1.33× | 0.99 rad | **0** |
| g1 (nv=49)    | 256  |     24,800 |    24,530 | 1.01× | 2.24 rad | **0** |
| g1 (nv=49)    | 1024 |     96,851 |    95,761 | 1.01× | 2.24 rad | **0** |
| g1 (nv=49)    | 4096 |    374,662 |   357,958 | 1.05× | 2.24 rad | **0** |

![Constrained solver performance](figures/constrained_performance.png)

Throughput tracks `dls` (factor-once ADMM), the per-solve overhead shrinks with
model size (1.08–1.33× on panda nv=9, 1.01–1.05× on g1 nv=49), and accuracy vs
mink `daqp` reaches the float32 parity floor (~6e-4) by `admm_iters≈30`.

```bash
# accuracy sweep vs mink (parity scene), and throughput + violation vs dls
uv run python benchmarks/bench_constrained.py panda --check --iters 20 30 40 --device cuda:0
uv run python benchmarks/bench_constrained.py panda --solvers dls constrained --nworld 4096 --amp-scale 3 --graph --device cuda:0
uv run python benchmarks/bench_constrained.py g1    --solvers dls constrained --nworld 4096 --amp-scale 3 --graph --device cuda:0
```

## Constrained solver — general inequalities (`G Δq ≤ h`)

The `constrained` backend also handles **arbitrary linear inequalities** a box
cannot express (an oriented half-space, coupled-joint bounds, collision rows),
via a factor-once reduced OSQP-ADMM: eliminate the constraint image `z = G Δq`
to the SPD normal matrix `M = H + σI + ρ GᵀG`, factor it once with the tile
Cholesky, then `admm_iters` of {cached solve → project `z` onto `(-∞, h]` → dual
update}. It returns `Δq`, whose feasibility tightens with `admm_iters`
(**asymptotic**), where the box path is exact at every step.

The path is auto-selected. Box-only limits keep the fast exact box kernel; an
inequality-only limit (`LinearInequalityLimit`) — or any box limit with
`use_inequalities=True` — switches to the general kernel, where box limits emit
their `[P; −P]` rows too. So the common case pays nothing, and the general path
is opt-in.

**Parity** (`tests/test_inequality_constraints.py`): the configuration limit
re-expressed as `G=[P; −P]` inequalities matches both the box path and mink's
`daqp` QP to ≈3e-6; an arbitrary half-space matches mink's `quadprog` (active /
inactive / two-row / padded-inert / batched-mixed); a trajectory that never
binds tracks identically to the unconstrained `dls`.

**Throughput vs `dls`** (RTX 4070 Ti SUPER, CUDA graph, `--amp-scale 3`,
`admm_iters=30`). `constrained` is the exact box path; `constrained-ineq` is the
same configuration limit driven through the general `G Δq ≤ h` kernel:

| scene | worlds | dls solves/s | box solves/s (viol) | general-ineq solves/s (viol) | box / ineq overhead |
|---|--:|--:|--:|--:|--:|
| panda (nv=9)  | 256  |    959,684 |   910,551 (**0**) |   824,590 (**0**)       | 1.05× / 1.16× |
| panda (nv=9)  | 1024 |  3,681,986 | 3,182,800 (**0**) | 2,761,159 (**0**)       | 1.16× / 1.33× |
| panda (nv=9)  | 4096 | 11,787,208 | 8,910,193 (**0**) | 6,337,345 (**0**)       | 1.32× / 1.86× |
| g1 (nv=49)    | 256  |     24,608 |    24,517 (**0**) |    22,989 (8.6e-5)      | 1.00× / 1.07× |
| g1 (nv=49)    | 1024 |     96,682 |    95,346 (**0**) |    80,452 (8.6e-5)      | 1.01× / 1.20× |
| g1 (nv=49)    | 4096 |    372,401 |   355,117 (**0**) |   218,947 (8.6e-5)      | 1.02× / 1.05× |

The general kernel costs more than the box kernel — dense `GᵀG` plus two tile
matmuls per iteration — so its overhead grows with the batch and the number of
rows (1.07–1.86× `dls`, vs the box path's 1.00–1.32×). Its feasibility is
asymptotic: on `panda` the ±unit configuration rows converge exactly by
`admm_iters=30` (violation **0**); on `g1` (more, coupled limited dofs) a
negligible ~9e-5 rad residual remains at 30 — raise `admm_iters` to shrink it, or
use the exact box path when the constraint is a per-dof interval.

```bash
# general inequality path (same config limit as [P;-P] rows) vs box vs dls
uv run python benchmarks/bench_constrained.py panda --solvers dls constrained constrained-ineq --nworld 4096 --amp-scale 3 --graph --device cuda:0
uv run python benchmarks/bench_constrained.py g1    --solvers dls constrained constrained-ineq --nworld 4096 --amp-scale 3 --graph --device cuda:0
```

_Numbers will shift with GPU model, batch, task stack, and warp/mujoco-warp versions. Re-run on
your target hardware._

---

## Closed-kinematics + collision-avoidance tasks (host-assembly hot path)

`EqualityConstraintTask` (Cassie closed four-bar chains) and
`CollisionAvoidanceLimit` (dual iiwa self-collision) build their rows from host
MuJoCo (`mj_fwdPosition`, `mj_geomDistance`, `mj_jac`) — one world at a time.
On GPU the QP solve is offloaded, so this host assembly, not the solve,
dominates the step (its share `hot %` climbs toward 100% as the batch grows).
`bench_tasks.py` measures throughput, the host-assembly share, the device
broadphase skip-rate (`bp-skip`), and mink parity.

**Accuracy is unchanged by every optimization here** (bit-identical rows), so the
mink parity floor is exactly the feature branch's: equality `|Δe| 3.7e-9`,
`|ΔJ| 2.3e-8`; collision `|ΔG| 1.0e-7`, `|Δh| 1.5e-6`.

Optimizations (all parity-preserving):
- **Equality:** `mj_forward → mj_fwdPosition` (equality `efc_pos`/`efc_J` are
  position-only) + drop a redundant `efc_J` copy.
- **Collision:** a **device broadphase** kernel runs the geom-pair proximity
  tests in parallel over every `(world, pair)` off the batched `geom_xpos`;
  worlds with no nearby pair skip the host narrowphase entirely. Plus a
  **slice scatter** that uploads only the collision block into `G/h` at its row
  offset instead of round-tripping the whole padded QP buffer each step.

Throughput on an RTX 4070 Ti SUPER (eager, `--collision-motion dense`):

| scene | worlds | baseline solves/s | optimized solves/s | speedup | host share (base → opt) |
|---|--:|--:|--:|--:|--:|
| cassie (nv=32, equality)   | 256  |  9,448 | 11,208 | 1.19× | 44% → 34% |
| cassie (nv=32, equality)   | 1024 | 16,110 | 22,891 | 1.42× | 74% → 64% |
| cassie (nv=32, equality)   | 4096 | 20,297 | 31,193 | **1.54×** | 92% → 87% |
| dual iiwa (nv=14, collision) | 256  | 18,925 | 31,223 | 1.65× | 89% → 83% |
| dual iiwa (nv=14, collision) | 1024 | 20,543 | 35,493 | 1.73× | 95% → 92% |
| dual iiwa (nv=14, collision) | 4096 | 20,214 | 37,101 | **1.84×** | 95% → 95% |

The speedup grows with the batch: the bigger the batch, the more the host
assembly dominates, so shrinking it moves the whole step. Baseline = the feature
branch's own `mj_forward` + full-buffer-round-trip code.

**Device broadphase, data-dependent.** The `dense` trajectory sweeps the arms
through each other, so ~⅓ of the (phase-offset) worlds are near every step
(`bp-skip ≈ 67%`). When collisions are sparse (`--collision-motion sparse`, arms
kept on their own sides) the prefilter drops **every** world it can
(`bp-skip = 100%`) and the host narrowphase all but vanishes:

| worlds | dense host `hot µs` | sparse host `hot µs` | sparse solves/s |
|---|--:|--:|--:|
| 256  |  6,791 |   379 | 152,010 |
| 1024 | 26,524 | 1,196 | 323,665 |
| 4096 | 105,261 | 3,857 | 509,283 |

At 4096 worlds the parallel pair checks cut host assembly ~27× (105 ms → 3.9 ms)
when few worlds actually collide — the batched-GPU broadphase paying off exactly
where a serial host loop would waste the most.

```bash
uv run python benchmarks/bench_tasks.py --check                                  # mink parity
uv run python benchmarks/bench_tasks.py --profile --device cuda:0 --nworld 256 1024 4096
uv run python benchmarks/bench_tasks.py dual_iiwa --profile --collision-motion sparse --device cuda:0
```
