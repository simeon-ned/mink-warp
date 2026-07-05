"""Damped least-squares (Gauss-Newton) IK backend.

One damped Gauss-Newton step per iteration:
``(W^T W + lambda I) dq = W^T e``, ``v = dq / dt`` — Mink's differential-IK step.
This is the original mink-warp solver, now behind the shared
:class:`~mink_warp.solvers.base.Solver` interface. Optional CUDA-graph capture
for a fixed task set.
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from ..configuration import Configuration
from ..kernels.solver import (
    accumulate_normal_equations,
    add_damping_diag,
    get_cholesky_solve_kernel,
    launch_cholesky_solve,
    neg_vec,
    scale_velocity,
    zero_normal_equations,
)
from ..tasks.task import Task
from .base import Solver


class DLSSolver(Solver):
    """Reusable batched damped-least-squares IK solver.

    Solves :math:`(W^T W + \\lambda I)\\Delta q = -W^T e` on device and returns
    :math:`v = \\Delta q / dt`. Optional CUDA graph capture for fixed task sets.

    Note: CUDA graphs cannot include host->device copies. ``dt`` is written to a
    device buffer before capture; integrate is out-of-place (in-place qpos
    aliasing is not graph-safe).
    """

    name = "dls"

    def __init__(self, configuration: Configuration, damping: float = 1e-12):
        super().__init__(configuration)
        self.default_damping = damping
        nworld = configuration.nworld
        nv = configuration.nv
        with wp.ScopedDevice(configuration.device):
            self.H = wp.zeros((nworld, nv, nv), dtype=float)
            self.c = wp.zeros((nworld, nv), dtype=float)
            self.mu_total = wp.zeros(nworld, dtype=float)
            self.rhs = wp.zeros((nworld, nv), dtype=float)
            self.dq = wp.zeros((nworld, nv), dtype=float)
            self.v = wp.zeros((nworld, nv), dtype=float)
        # Newton-style tile Cholesky, specialized for this model's nv.
        self._cholesky_solve = get_cholesky_solve_kernel(nv)
        self._graph = None
        self._graph_tasks: tuple[Task, ...] | None = None
        self._graph_dt: float | None = None
        self._graph_damping: float | None = None

    def solve(
        self,
        tasks: Sequence[Task],
        dt: float,
        damping: float | None = None,
    ) -> wp.array:
        """Compute the velocity for one differential step (no integration)."""
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
        """Solve and integrate ``iterations`` times; returns last velocity."""
        self._check_dt(dt)
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        damping = self._damping(damping)
        if (use_graph and iterations == 1
                and wp.get_device(self.configuration.device).is_cuda
                and all(getattr(t, "supports_cuda_graph", True) for t in tasks)):
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
        """Drop a captured CUDA graph (e.g. after changing the task list)."""
        self._graph = None
        self._graph_tasks = None
        self._graph_dt = None
        self._graph_damping = None

    def _ensure_graph(
        self,
        tasks: Sequence[Task],
        dt: float,
        damping: float,
    ) -> None:
        task_key = tuple(tasks)
        if (
            self._graph is not None
            and self._graph_tasks == task_key
            and self._graph_dt == dt
            and self._graph_damping == damping
        ):
            return

        # Host->device dt write must happen outside the graph.
        self.configuration.set_integration_dt(dt)

        # Snapshot the pristine config: the warmup below integrates one real step.
        q_snapshot = wp.clone(self.configuration.q)
        # Warm up kernels and allocate all task buffers before capture.
        self._solve_device(tasks, dt, damping)
        self.configuration.integrate_inplace(self.v, dt=None)

        with wp.ScopedCapture() as capture:
            self._solve_device(tasks, dt, damping)
            # dt already on device; do not host-assign inside the graph.
            self.configuration.integrate_inplace(self.v, dt=None)
        self._graph = capture.graph
        self._graph_tasks = task_key
        self._graph_dt = dt
        self._graph_damping = damping
        # Undo the warmup + capture advances so capture_launch replays from the
        # pristine configuration (otherwise the first graphed call double-steps).
        with wp.ScopedDevice(self.configuration.device):
            wp.copy(self.configuration.q, q_snapshot)
            self.configuration.update()

    def _solve_device(
        self,
        tasks: Sequence[Task],
        dt: float,
        damping: float,
    ) -> None:
        cfg = self.configuration
        nv = cfg.nv
        nworld = cfg.nworld
        with wp.ScopedDevice(cfg.device):
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
            launch_cholesky_solve(
                self._cholesky_solve,
                nworld=nworld,
                H=self.H,
                rhs=self.rhs,
                dq=self.dq,
            )
            wp.launch(
                scale_velocity,
                dim=nworld,
                inputs=[self.dq, float(dt), nv],
                outputs=[self.v],
            )
