"""Constrained (box-QP) IK backend enforcing hard joint limits.

Solves, per world, the same QP as mink::

    min_dq  1/2 dq^T H dq + c^T dq   s.t.   lo <= dq <= hi

where ``H, c`` are assembled from the task stack exactly as :class:`DLSSolver`
does, and ``[lo, hi]`` is the intersection of the supplied hard limits
(:class:`~mink_warp.limits.ConfigurationLimit` /
:class:`~mink_warp.limits.VelocityLimit`). The box is solved by OSQP-style
box-ADMM (factor ``M = H + rho I`` once with tile Cholesky, then a fixed number
of cached-solve + box-clip + dual-update iterations), returning the projected
step ``dq`` which lies inside ``[lo, hi]`` at *every* iteration — so joint /
velocity limits are never violated, even when the target drives the arm hard
into a bound and even if the inner loop is truncated.

Tile Cholesky is cuSolverDx / GPU-only, so the ADMM solve runs on CUDA only.
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from ..configuration import Configuration
from ..kernels.constrained import (
    compute_rho_mean_diag,
    get_admm_box_kernel,
    init_box,
    launch_admm_box_solve,
)
from ..kernels.solver import (
    accumulate_normal_equations,
    add_damping_diag,
    neg_vec,
    scale_velocity,
    zero_normal_equations,
)
from ..limits import ConfigurationLimit, Limit
from ..tasks.task import Task
from .base import Solver


class ConstrainedSolver(Solver):
    """Batched box-constrained IK solver (hard joint limits via ADMM).

    Args:
        configuration: Batched configuration to advance.
        limits: Hard limits to enforce. ``None`` defaults to a single
            :class:`ConfigurationLimit` (mink's ``limits=None`` behaviour); pass
            ``[]`` to disable limits (then this reduces to a regularized DLS solve).
        admm_iters: Inner ADMM iterations per solve (feasibility holds at any
            count; more tightens agreement with the true QP optimum).
        rho_scale, rho_min, rho_max: Control the per-world ADMM penalty
            ``rho = clamp(rho_scale*mean(diag H), rho_min, rho_max)``.
        alpha: ADMM over-relaxation in [1, 2) (1.6 accelerates convergence).
        damping: Levenberg-Marquardt damping added to ``H``'s diagonal (matches
            mink's ``damping``, applied on top of per-task ``mu``).
    """

    name = "constrained"

    def __init__(
        self,
        configuration: Configuration,
        limits: Sequence[Limit] | None = None,
        *,
        admm_iters: int = 20,
        rho_scale: float = 1.0,
        rho_min: float = 1e-6,
        rho_max: float = 1e6,
        alpha: float = 1.6,
        damping: float = 1e-12,
    ):
        super().__init__(configuration)
        if admm_iters < 1:
            raise ValueError(f"admm_iters must be >= 1, got {admm_iters}")
        if limits is None:
            limits = [ConfigurationLimit(configuration.model)]
        self.limits = list(limits)
        self.admm_iters = int(admm_iters)
        self.default_damping = damping
        self.rho_scale = float(rho_scale)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.alpha = float(alpha)

        nworld = configuration.nworld
        nv = configuration.nv
        with wp.ScopedDevice(configuration.device):
            self.H = wp.zeros((nworld, nv, nv), dtype=float)
            self.c = wp.zeros((nworld, nv), dtype=float)
            self.mu_total = wp.zeros(nworld, dtype=float)
            self.rhs = wp.zeros((nworld, nv), dtype=float)  # b = -c = W^T e
            self.lo = wp.zeros((nworld, nv), dtype=float)
            self.hi = wp.zeros((nworld, nv), dtype=float)
            self.rho = wp.zeros(nworld, dtype=float)
            self.dq = wp.zeros((nworld, nv), dtype=float)
            self.v = wp.zeros((nworld, nv), dtype=float)
        self._admm = get_admm_box_kernel(nv, self.admm_iters)
        self._graph = None
        self._graph_key: tuple | None = None

    def solve(
        self,
        tasks: Sequence[Task],
        dt: float,
        damping: float | None = None,
    ) -> wp.array:
        """Compute the box-feasible velocity for one differential step."""
        self._check_dt(dt)
        self._solve_device(tasks, dt, self._damping(damping))
        return self.v

    def solve_and_integrate(
        self,
        tasks: Sequence[Task],
        dt: float,
        *,
        iterations: int = 1,
        use_graph: bool = False,
        damping: float | None = None,
    ) -> wp.array:
        """Solve and integrate ``iterations`` times; returns last velocity.

        Each outer iteration re-linearizes ``H`` and recomputes the box at the
        current configuration, so the returned step stays feasible throughout.
        """
        self._check_dt(dt)
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        damping = self._damping(damping)
        if (
            use_graph
            and iterations == 1
            and wp.get_device(self.configuration.device).is_cuda
        ):
            self._ensure_graph(tasks, dt, damping)
            if self._graph is not None:
                wp.capture_launch(self._graph)
                return self.v

        v = self.v
        for _ in range(iterations):
            v = self.solve(tasks, dt, damping=damping)
            self.configuration.integrate_inplace(v, dt)
        return v

    def _damping(self, damping: float | None) -> float:
        return self.default_damping if damping is None else damping

    def invalidate_graph(self) -> None:
        self._graph = None
        self._graph_key = None

    def _ensure_graph(
        self,
        tasks: Sequence[Task],
        dt: float,
        damping: float,
    ) -> None:
        key = (tuple(tasks), dt, damping)
        if self._graph is not None and self._graph_key == key:
            return

        self.configuration.set_integration_dt(dt)
        q_snapshot = wp.clone(self.configuration.q)
        # Warm up kernels + limit device buffers before capture.
        self._solve_device(tasks, dt, damping)
        self.configuration.integrate_inplace(self.v, dt=None)

        with wp.ScopedCapture() as capture:
            self._solve_device(tasks, dt, damping)
            self.configuration.integrate_inplace(self.v, dt=None)
        self._graph = capture.graph
        self._graph_key = key
        # Undo warmup + capture advances so replay starts from a pristine config.
        with wp.ScopedDevice(self.configuration.device):
            wp.copy(self.configuration.q, q_snapshot)
            self.configuration.update()

    def _assemble(
        self,
        tasks: Sequence[Task],
        dt: float,
        damping: float,
    ) -> None:
        """Assemble the per-world box QP (H, b=-c, lo, hi, rho) on device.

        Everything up to but excluding the (GPU-only) tile-Cholesky ADMM solve,
        so it runs on any device and can be inspected in CPU tests.
        """
        cfg = self.configuration
        nv = cfg.nv
        nworld = cfg.nworld
        with wp.ScopedDevice(cfg.device):
            # --- Objective H, b (=-c=W^T e), identical to the DLS assembly. ---
            wp.launch(
                zero_normal_equations,
                dim=nworld,
                inputs=[self.H, self.c, self.mu_total, nv],
            )
            for task in tasks:
                W, e, mu = task.compute_residual(cfg)
                k = int(W.shape[1])
                wp.launch(
                    accumulate_normal_equations,
                    dim=nworld,
                    inputs=[W, e, mu, k, nv, self.H, self.c, self.mu_total],
                )
            wp.launch(
                add_damping_diag,
                dim=(nworld, nv),
                inputs=[self.H, self.mu_total, float(damping), nv],
            )
            wp.launch(
                neg_vec,
                dim=nworld,
                inputs=[self.c, nv],
                outputs=[self.rhs],
            )
            # --- Box [lo, hi] from the hard limits. ---
            wp.launch(init_box, dim=(nworld, nv), outputs=[self.lo, self.hi])
            for limit in self.limits:
                limit.apply_box(cfg, dt, self.lo, self.hi)
            # --- Per-world ADMM penalty. ---
            wp.launch(
                compute_rho_mean_diag,
                dim=nworld,
                inputs=[
                    self.H,
                    nv,
                    self.rho_scale,
                    self.rho_min,
                    self.rho_max,
                ],
                outputs=[self.rho],
            )

    def _solve_device(
        self,
        tasks: Sequence[Task],
        dt: float,
        damping: float,
    ) -> None:
        cfg = self.configuration
        nv = cfg.nv
        nworld = cfg.nworld
        self._assemble(tasks, dt, damping)
        with wp.ScopedDevice(cfg.device):
            # --- Box-ADMM solve -> feasible dq (GPU-only tile Cholesky). ---
            launch_admm_box_solve(
                self._admm,
                nworld=nworld,
                H=self.H,
                b=self.rhs,
                lo=self.lo,
                hi=self.hi,
                rho=self.rho,
                alpha=self.alpha,
                dq=self.dq,
            )
            wp.launch(
                scale_velocity,
                dim=nworld,
                inputs=[self.dq, float(dt), nv],
                outputs=[self.v],
            )
