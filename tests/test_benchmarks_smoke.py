"""Smoke tests for the benchmark suite: modules import and run a few steps.

These keep benchmarks/ from bit-rotting as the API evolves. They are cheap
(tiny batch, few steps) and run on whatever warp device is available.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_BENCH = Path(__file__).resolve().parent.parent / "benchmarks"
sys.path.insert(0, str(_BENCH))

import common  # noqa: E402
from scenes import DT, SCENES  # noqa: E402


def test_common_summarize_and_throughput():
    st = common.summarize([1.0, 2.0, 3.0, 4.0])
    assert st["n"] == 4
    assert st["min"] == 1.0 and st["max"] == 4.0
    assert st["median"] == pytest.approx(2.5)
    assert common.throughput(1e-3, 100) == pytest.approx(1e5)


@pytest.mark.parametrize("key", list(SCENES))
def test_scene_setup_and_step(key):
    scene = SCENES[key]
    s = scene.setup_mw(2)
    scene.update_mw(s, DT)
    v = s["solver"].solve(s["tasks"], DT, damping=s["damping"])
    arr = v.numpy()
    assert arr.shape == (2, s["configuration"].nv)
    assert np.isfinite(arr).all()


@pytest.mark.parametrize("kind", ["dls", "lm", "lbfgs"])
def test_bench_ik_run_batch_each_solver(kind):
    import bench_ik  # noqa: E402

    st = bench_ik.run_batch("panda", nworld=2, steps=3, warmup=1,
                            use_graph=False, device=None,
                            solver_kind=kind, iters=2)
    assert st["solver"] == kind
    assert st["nworld"] == 2
    assert st["solves_per_s"] > 0.0
    assert np.isfinite(st["mean"])


def test_panda_parity_small():
    pytest.importorskip("mink")
    import qpsolvers

    if "daqp" not in qpsolvers.available_solvers and not qpsolvers.available_solvers:
        pytest.skip("no qpsolvers backend")
    import bench_parity  # noqa: E402

    solver = bench_parity._pick_solver(None)
    r = bench_parity.run("panda", steps=10, seed=0, solver=solver)
    vmax = float(np.max(np.abs(r["v_raw"])))
    # float32 mink-warp vs float64 mink: agreement to a few e-3.
    assert vmax < 5e-3
