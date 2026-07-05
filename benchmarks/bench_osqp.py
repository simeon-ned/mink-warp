"""Validate the constrained solver's ADMM against reference **OSQP** on a set of
OSQP-example QPs (https://osqp.org/docs/examples/).

The constrained IK solver's inner solve is OSQP-style operator splitting. This
harness lifts that inner solver out of the IK context and runs it on generic
convex QPs — the same problem classes OSQP documents — to answer two questions:

1. **Is our solver correct?** Its converged solution must equal OSQP's (and an
   independent `daqp` cross-check) to solver tolerance, and be feasible.
2. **How does it converge / compute vs OSQP?** With *matched* hyperparameters
   (no Ruiz scaling, no adaptive rho, same rho / sigma / alpha) our fixed-trip
   kernel and OSQP run the *same* ADMM, so their iterates should track — a direct
   check that we implement OSQP's algorithm. Against *default* OSQP (adaptive rho
   + scaling + polish) OSQP converges in fewer iterations; we trade that for a
   branchless fixed-trip kernel that solves a whole *batch* of QPs in one GPU
   launch (the throughput comparison).

Problem forms map onto the two solve paths:
* **box** ``min ½ xᵀH x + cᵀx s.t. lo ≤ x ≤ hi`` -> box-ADMM kernel.
* **inequality** ``min ½ xᵀH x + cᵀx s.t. G x ≤ h`` -> general OSQP-ADMM kernel.
  Two-sided ``l ≤ A x ≤ u`` is stacked as ``[A; -A] x ≤ [u; -l]``.

Run (osqp / scipy / matplotlib are benchmark-only, pulled in on the fly)::

    uv run --with osqp --with matplotlib python benchmarks/bench_osqp.py --plot
    uv run --with osqp python benchmarks/bench_osqp.py --problem random_qp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from time import perf_counter_ns

import numpy as np
import warp as wp

sys.path.insert(0, str(Path(__file__).parent.parent / "tests"))
from helpers_admm import box_qp_admm, ineq_qp_admm  # noqa: E402

from mink_warp.kernels.constrained import (  # noqa: E402
    get_admm_box_kernel,
    get_admm_ineq_kernel,
    launch_admm_box_solve,
    launch_admm_ineq_solve,
)

FIG_DIR = Path(__file__).parent / "figures"


# --------------------------------------------------------------------------- #
# Problems (OSQP examples), each in our dense form + the OSQP (P,q,A,l,u) form.
# --------------------------------------------------------------------------- #
class Problem:
    def __init__(self, name, kind, H, c, *, box=None, G=None, h=None, osqp_form=None):
        self.name = name
        self.kind = kind  # "box" | "ineq"
        self.H = np.asarray(H, float)
        self.c = np.asarray(c, float)
        self.box = box  # (lo, hi)
        self.G = None if G is None else np.asarray(G, float)
        self.h = None if h is None else np.asarray(h, float)
        self.osqp_form = osqp_form  # (P, q, A, l, u) dense
        self.n = self.H.shape[0]

    def objective(self, x):
        return 0.5 * x @ self.H @ x + self.c @ x

    def primal_residual(self, x):
        if self.kind == "box":
            lo, hi = self.box
            return float(max(0.0, np.max(lo - x), np.max(x - hi)))
        return float(max(0.0, np.max(self.G @ x - self.h)))

    def rho(self):
        """Geometric-mean-of-positive-diagonal penalty (robust form of the IK
        heuristic, which assumes H has a strictly positive diagonal)."""
        d = np.diag(self.H)
        dp = d[d > 1e-9]
        if dp.size == 0:
            return 1.0
        return float(np.sqrt(max(dp.min(), 1e-6) * dp.max()))


def bounded_least_squares(seed=0, m=30, n=12):
    """OSQP "Least squares": min ½‖Ax−b‖² s.t. −1 ≤ x ≤ 1  (box)."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    b = rng.standard_normal(m)
    H = A.T @ A + 1e-3 * np.eye(n)
    c = -A.T @ b
    lo, hi = -np.ones(n), np.ones(n)
    osqp_form = (H, c, np.eye(n), lo, hi)
    return Problem("bounded_least_squares", "box", H, c, box=(lo, hi), osqp_form=osqp_form)


def random_qp(seed=1, n=12, m=20):
    """OSQP "random QP": min ½xᵀHx + cᵀx s.t. Gx ≤ h, rows placed active."""
    rng = np.random.default_rng(seed)
    W = rng.standard_normal((n, n))
    H = W @ W.T + 0.3 * np.eye(n)
    c = rng.standard_normal(n)
    G = rng.standard_normal((m, n))
    x_unc = np.linalg.solve(H, -c)
    h = G @ x_unc - rng.uniform(0.05, 0.4, m)  # push several rows active
    osqp_form = (H, c, G, np.full(m, -np.inf), h)
    return Problem("random_qp", "ineq", H, c, G=G, h=h, osqp_form=osqp_form)


def lasso(seed=2, m=40, n=12, lam=0.3):
    """OSQP "Lasso": min ½‖Ax−b‖² + λ‖x‖₁, as a QP in z=[x;t] with −t≤x≤t.

    Variables z=[x; t] (2n). Objective ½ xᵀ(AᵀA)x − (Aᵀb)ᵀx + λ 1ᵀt. Inequality
    rows x−t ≤ 0 and −x−t ≤ 0. Stresses the general path on a rank-deficient H
    (the t-block is zero) — where the σI SPD floor and rho matter."""
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((m, n))
    x_true = np.zeros(n)
    x_true[rng.choice(n, size=max(1, n // 4), replace=False)] = rng.standard_normal(
        max(1, n // 4)
    )
    b = A @ x_true + 0.1 * rng.standard_normal(m)
    AtA = A.T @ A
    H = np.zeros((2 * n, 2 * n))
    H[:n, :n] = AtA
    c = np.concatenate([-A.T @ b, lam * np.ones(n)])
    Id = np.eye(n)
    G = np.block([[Id, -Id], [-Id, -Id]])  # x - t <= 0 ; -x - t <= 0
    h = np.zeros(2 * n)
    osqp_form = (H, c, G, np.full(2 * n, -np.inf), h)
    return Problem("lasso", "ineq", H, c, G=G, h=h, osqp_form=osqp_form)


PROBLEMS = {p().name: p for p in [bounded_least_squares, random_qp, lasso]}


# --------------------------------------------------------------------------- #
# Reference solvers
# --------------------------------------------------------------------------- #
def solve_osqp(prob, *, max_iter, matched=False, rho=None, sigma=1e-6, alpha=1.6,
               eps=1e-9):
    import osqp
    import scipy.sparse as sp

    P, q, A, lb, ub = prob.osqp_form
    mo = osqp.OSQP()
    kw = dict(
        P=sp.csc_matrix(P), q=q, A=sp.csc_matrix(A), l=lb, u=ub,
        verbose=False, max_iter=max_iter, eps_abs=eps, eps_rel=eps,
    )
    if matched:
        kw.update(
            rho=rho, sigma=sigma, alpha=alpha, scaling=0, adaptive_rho=0,
            polish=False, check_termination=max_iter + 1,  # run exactly max_iter
        )
    mo.setup(**kw)
    r = mo.solve()
    return np.asarray(r.x, float), r.info.iter, r.info.status


def solve_daqp(prob):
    try:
        import daqp
    except Exception:
        return None
    if prob.kind == "box":
        lo, hi = prob.box
        A = np.eye(prob.n)
        bu, bl = hi.copy(), lo.copy()
    else:
        A = prob.G
        bu = prob.h.copy()
        bl = np.full(prob.G.shape[0], -1e30)
    sense = np.zeros(A.shape[0], dtype=np.int32)
    x, _f, ef, _i = daqp.solve(
        np.ascontiguousarray(prob.H), np.ascontiguousarray(prob.c),
        np.ascontiguousarray(A, float), np.ascontiguousarray(bu),
        np.ascontiguousarray(bl), sense,
    )
    return np.asarray(x, float) if ef == 1 else None


# --------------------------------------------------------------------------- #
# Our solver — per-iteration trace (mirrors the shipped kernel math exactly) and
# the actual device kernel.
# --------------------------------------------------------------------------- #
def our_trace(prob, *, iters, rho, sigma=1e-6, alpha=1.6):
    """Per-iteration [x] iterates from the same ADMM the kernel runs."""
    H, c = prob.H, prob.c
    n = prob.n
    xs = []
    def make_solve(L):
        def sol(r):
            return np.linalg.solve(L.T, np.linalg.solve(L, r))
        return sol

    if prob.kind == "box":
        lo, hi = prob.box
        M = H + rho * np.eye(n)
        L = np.linalg.cholesky(M)
        sol = make_solve(L)
        b = -c
        x = sol(b)
        z = np.clip(x, lo, hi)
        u = np.zeros(n)
        xs.append(z.copy())
        for _ in range(iters):
            x = sol(rho * (z - u) + b)
            xh = alpha * x + (1 - alpha) * z
            z = np.clip(xh + u, lo, hi)
            u = u + xh - z
            xs.append(z.copy())
    else:
        G, h = prob.G, prob.h
        M = H + sigma * np.eye(n) + rho * (G.T @ G)
        L = np.linalg.cholesky(M)
        sol = make_solve(L)
        x = sol(-c)
        z = np.minimum(G @ x, h)
        y = np.zeros(G.shape[0])
        xs.append(x.copy())
        for _ in range(iters):
            xt = sol(sigma * x - c + G.T @ (rho * z - y))
            zt = G @ xt
            x = alpha * xt + (1 - alpha) * x
            zh = alpha * zt + (1 - alpha) * z
            zn = np.minimum(zh + y / rho, h)
            y = y + rho * (zh - zn)
            z = zn
            xs.append(x.copy())
    return xs


def our_kernel(prob, *, iters, rho, sigma=1e-6, alpha=1.6, nworld=1, device="cpu"):
    """Run the SHIPPED device kernel; returns (x for world 0, seconds/solve)."""
    n = prob.n
    with wp.ScopedDevice(device):
        H = wp.array(np.tile(prob.H[None], (nworld, 1, 1)), dtype=float)
        b = wp.array(np.tile((-prob.c)[None], (nworld, 1)), dtype=float)
        rho_a = wp.array(np.full(nworld, rho, np.float32), dtype=float)
        dq = wp.zeros((nworld, n), dtype=float)
        if prob.kind == "box":
            lo, hi = prob.box
            lo_a = wp.array(np.tile(lo[None], (nworld, 1)), dtype=float)
            hi_a = wp.array(np.tile(hi[None], (nworld, 1)), dtype=float)
            k = get_admm_box_kernel(n, iters)

            def run():
                launch_admm_box_solve(
                    k, nworld=nworld, H=H, b=b, lo=lo_a, hi=hi_a, rho=rho_a,
                    alpha=alpha, dq=dq)
        else:
            m = prob.G.shape[0]
            G = wp.array(np.tile(prob.G[None], (nworld, 1, 1)), dtype=float)
            h = wp.array(np.tile(prob.h[None], (nworld, 1)), dtype=float)
            k = get_admm_ineq_kernel(n, m, iters)

            def run():
                launch_admm_ineq_solve(
                    k, nworld=nworld, H=H, b=b, G=G, h=h, rho=rho_a, sigma=sigma,
                    alpha=alpha, dq=dq)
        run()
        wp.synchronize_device(device)  # warmup / compile
        reps = 20
        t0 = perf_counter_ns()
        for _ in range(reps):
            run()
        wp.synchronize_device(device)
        sec = (perf_counter_ns() - t0) * 1e-9 / reps
    return dq.numpy()[0].copy(), sec


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def evaluate(prob, *, kmax=400, device="cpu"):
    rho = prob.rho()
    # Reference optimum (OSQP high precision) + independent daqp cross-check.
    x_star, osqp_iters, status = solve_osqp(prob, max_iter=20000, eps=1e-10)
    obj_star = prob.objective(x_star)
    x_daqp = solve_daqp(prob)

    # Our convergence trace (numpy, exact kernel math) + faithfulness vs helper.
    xs = our_trace(prob, iters=kmax, rho=rho)
    if prob.kind == "box":
        x_final_helper = box_qp_admm(prob.H, prob.c, *prob.box, rho=rho, iters=kmax,
                                     alpha=1.6)
    else:
        x_final_helper = ineq_qp_admm(prob.H, prob.c, prob.G, prob.h, rho=rho,
                                      sigma=1e-6, iters=kmax, alpha=1.6)
    assert np.allclose(xs[-1], x_final_helper, atol=1e-6), "trace != shipped helper"

    # Device kernel at sampled K -> must land on the trace + match OSQP.
    kernel_pts = {}
    for K in [10, 30, 100, kmax]:
        xk, _ = our_kernel(prob, iters=K, rho=rho, device=device)
        kernel_pts[K] = xk

    # OSQP traces: matched settings (same ADMM) + default (adaptive+scaling).
    ks = _iter_grid(kmax)
    dist_ours = [np.abs(xs[k] - x_star).max() for k in ks]
    res_ours = [prob.primal_residual(xs[k]) for k in ks]
    dist_match, dist_def = [], []
    for k in ks:
        xm, _, _ = solve_osqp(prob, max_iter=max(k, 1), matched=True, rho=rho)
        dist_match.append(np.abs(xm - x_star).max())
        xd, _, _ = solve_osqp(prob, max_iter=max(k, 1))
        dist_def.append(np.abs(xd - x_star).max())

    return dict(
        prob=prob, rho=rho, x_star=x_star, obj_star=obj_star, status=status,
        osqp_iters=osqp_iters, x_daqp=x_daqp, xs=xs, kernel_pts=kernel_pts,
        ks=ks, dist_ours=dist_ours, res_ours=res_ours,
        dist_match=dist_match, dist_def=dist_def,
    )


def _iter_grid(kmax):
    g = sorted(set([1, 2, 3, 5, 8, 12, 20, 30, 50, 80, 120, 200, 300, kmax]))
    return [k for k in g if k <= kmax]


def print_report(ev):
    p = ev["prob"]
    xk = ev["kernel_pts"][max(ev["kernel_pts"])]
    dx_osqp = np.abs(xk - ev["x_star"]).max()
    dx_daqp = None if ev["x_daqp"] is None else np.abs(ev["x_star"] - ev["x_daqp"]).max()
    print(f"\n=== {p.name}  (n={p.n}, kind={p.kind}, rho={ev['rho']:.3g}) ===")
    print(f"  OSQP: status={ev['status']}, iters={ev['osqp_iters']}, obj*={ev['obj_star']:.6g}")
    if dx_daqp is not None:
        print(f"  OSQP vs daqp (independent)         |dx|_inf = {dx_daqp:.2e}")
    print(f"  our kernel (K={max(ev['kernel_pts'])}) vs OSQP*    |dx|_inf = {dx_osqp:.2e}")
    print(f"  our obj gap |obj-obj*|              = {abs(p.objective(xk)-ev['obj_star']):.2e}")
    print(f"  our feasibility (primal residual)  = {p.primal_residual(xk):.2e}")
    print(f"  {'K':>5} {'our |x-x*|':>12} {'OSQP-match':>12} {'OSQP-default':>12}")
    for k, do, dm, dd in zip(ev["ks"], ev["dist_ours"], ev["dist_match"], ev["dist_def"]):
        print(f"  {k:>5} {do:>12.2e} {dm:>12.2e} {dd:>12.2e}")


def throughput_report(prob, *, iters=100, device="cpu"):
    """Batched: our kernel solves nworld QPs in one launch; OSQP loops."""
    rho = prob.rho()
    print(f"\n--- throughput ({prob.name}, K={iters}, {device}) ---")
    for nworld in [1, 64, 256, 1024]:
        _, sec = our_kernel(prob, iters=iters, rho=rho, nworld=nworld, device=device)
        print(f"  ours  nworld={nworld:>5}: {sec*1e6:8.1f} us/batch  "
              f"{nworld/sec:12.0f} solves/s")
    # OSQP single-solve time (loop cost = nworld x this).
    import osqp
    import scipy.sparse as sp
    P, q, A, lb, ub = prob.osqp_form
    mo = osqp.OSQP()
    mo.setup(P=sp.csc_matrix(P), q=q, A=sp.csc_matrix(A), l=lb, u=ub, verbose=False)
    mo.solve()  # warm
    t0 = perf_counter_ns()
    reps = 50
    for _ in range(reps):
        mo.solve()
    sec = (perf_counter_ns() - t0) * 1e-9 / reps
    print(f"  osqp  single-solve: {sec*1e6:8.1f} us      {1/sec:12.0f} solves/s (serial)")


def plot(evs, out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ncol = len(evs)
    fig, axes = plt.subplots(2, ncol, figsize=(5.2 * ncol, 7.2), squeeze=False)
    for j, ev in enumerate(evs):
        p = ev["prob"]
        ax = axes[0][j]
        ax.semilogy(ev["ks"], np.maximum(ev["dist_ours"], 1e-16), "-o", ms=3,
                    color="#2ca02c", label="ours (fixed-trip ADMM)")
        ax.semilogy(ev["ks"], np.maximum(ev["dist_match"], 1e-16), "--", lw=1.6,
                    color="#1f77b4", label="OSQP (matched settings)")
        ax.semilogy(ev["ks"], np.maximum(ev["dist_def"], 1e-16), ":", lw=1.8,
                    color="#d62728", label="OSQP (default: adaptive ρ + scaling)")
        for K, xk in ev["kernel_pts"].items():
            ax.semilogy([K], [max(np.abs(xk - ev["x_star"]).max(), 1e-16)], "k*",
                        ms=9, zorder=5)
        ax.set_title(f"{p.name}\n(n={p.n}, {p.kind})", fontsize=10)
        ax.set_ylabel(r"$\|x_k - x^\star\|_\infty$" if j == 0 else "")
        ax.grid(alpha=0.25, which="both")
        if j == 0:
            ax.legend(fontsize=7.5, loc="upper right")
        ax.plot([], [], "k*", ms=9, label="device kernel")

        ax2 = axes[1][j]
        ax2.semilogy(ev["ks"], np.maximum(ev["res_ours"], 1e-16), "-o", ms=3,
                     color="#2ca02c")
        ax2.set_ylabel("primal residual  max(0, Gx−h)" if j == 0 else "")
        ax2.set_xlabel("ADMM iteration K")
        ax2.grid(alpha=0.25, which="both")
    axes[0][ncol - 1].plot([], [], "k*", ms=9, label="device kernel (★)")
    fig.suptitle("Constrained-solver ADMM vs reference OSQP — convergence to the "
                 "shared optimum $x^\\star$", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    print(f"\nsaved {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--problem", nargs="+", default=list(PROBLEMS),
                    choices=list(PROBLEMS))
    ap.add_argument("--kmax", type=int, default=400)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--throughput", action="store_true")
    args = ap.parse_args()

    evs = []
    for name in args.problem:
        ev = evaluate(PROBLEMS[name](), kmax=args.kmax, device=args.device)
        print_report(ev)
        evs.append(ev)
    if args.throughput:
        for ev in evs:
            throughput_report(ev["prob"], device=args.device)
    if args.plot:
        plot(evs, FIG_DIR / "osqp_convergence.png")


if __name__ == "__main__":
    main()
