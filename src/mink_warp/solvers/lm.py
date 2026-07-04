"""Levenberg-Marquardt IK backend (Newton-style, batched).

Iterated Gauss-Newton with adaptive damping and a trust-region accept/reject:
each iteration forms ``H = W^T W``, ``g = W^T r`` over the tasks, solves
``(H + lambda I) delta = -g`` per world with a tile Cholesky, proposes
``q <- q (+) delta``, then keeps or reverts the step by the gain ratio
``rho = actual / predicted`` and updates lambda (Nielsen). Ports the structure of
``newton/_src/sim/ik/ik_lm_optimizer.py`` onto mink-warp's task API.
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from ..configuration import Configuration
from ..kernels.lm import (
    accumulate_cost,
    accumulate_lm,
    add_lm_lambda,
    lm_accept,
    lm_pred_reduction,
    zero_cost,
    zero_lm,
)
from ..kernels.solver import (
    get_cholesky_solve_kernel,
    launch_cholesky_solve,
    neg_vec,
    scale_velocity,
)
from ..tasks.task import Task
from .base import Solver


class LMSolver(Solver):
    """Batched Levenberg-Marquardt IK solver.

    Args:
        lambda0: initial damping (reset at the start of every call).
        lambda_min / lambda_max: clamp range for the adaptive damping.
        eps: predicted-reduction floor below which a step is treated as a
            reject (avoids division blow-up near convergence).
    """

    name = "lm"

    def __init__(
        self,
        configuration: Configuration,
        lambda0: float = 1e-2,
        lambda_min: float = 1e-9,
        lambda_max: float = 1e9,
        eps: float = 1e-16,
    ):
        super().__init__(configuration)
        self.lambda0 = float(lambda0)
        self.lambda_min = float(lambda_min)
        self.lambda_max = float(lambda_max)
        self.eps = float(eps)

        nworld = configuration.nworld
        nv = configuration.nv
        nq = configuration.nq
        with wp.ScopedDevice(configuration.device):
            self.H = wp.zeros((nworld, nv, nv), dtype=float)
            self.g = wp.zeros((nworld, nv), dtype=float)
            self.rhs = wp.zeros((nworld, nv), dtype=float)
            self.delta = wp.zeros((nworld, nv), dtype=float)
            self.dq_total = wp.zeros((nworld, nv), dtype=float)
            self.v = wp.zeros((nworld, nv), dtype=float)
            self.C_old = wp.zeros(nworld, dtype=float)
            self.C_new = wp.zeros(nworld, dtype=float)
            self.pred = wp.zeros(nworld, dtype=float)
            self.lam = wp.zeros(nworld, dtype=float)
            self.nu = wp.zeros(nworld, dtype=float)
            self.qpos_old = wp.zeros((nworld, nq), dtype=float)
        self._cholesky_solve = get_cholesky_solve_kernel(nv)
        self._graph = None
        self._graph_key: tuple | None = None

    def solve_and_integrate(
        self,
        tasks: Sequence[Task],
        dt: float,
        *,
        iterations: int = 10,
        use_graph: bool = False,
        **_ignored,
    ) -> wp.array:
        self._check_dt(dt)
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        self._reset()

        if use_graph and wp.get_device().is_cuda:
            self._ensure_graph(tasks, dt, iterations)
            if self._graph is not None:
                wp.capture_launch(self._graph)
                return self.v

        for _ in range(iterations):
            self._iteration(tasks, unit_dt=1.0)
        self._finalize_velocity(dt)
        return self.v

    # Internal.

    def _reset(self) -> None:
        with wp.ScopedDevice(self.configuration.device):
            self.lam.fill_(self.lambda0)
            self.nu.fill_(2.0)
            self.dq_total.zero_()

    def _iteration(self, tasks: Sequence[Task], unit_dt: float | None) -> None:
        cfg = self.configuration
        nv = cfg.nv
        nq = cfg.nq
        nworld = cfg.nworld
        with wp.ScopedDevice(cfg.device):
            # Assemble H, g, C_old at the current configuration.
            wp.launch(zero_lm, dim=nworld, inputs=[self.H, self.g, self.C_old, nv])
            for task in tasks:
                error, jac, cost = task.error_jacobian_cost(cfg)
                k = int(error.shape[1])
                wp.launch(
                    accumulate_lm,
                    dim=nworld,
                    inputs=[error, jac, cost, k, nv, self.H, self.g, self.C_old],
                )
            # Save q, damp, solve (H + lambda I) delta = -g.
            wp.copy(self.qpos_old, cfg.q)
            wp.launch(add_lm_lambda, dim=(nworld, nv), inputs=[self.H, self.lam, nv])
            wp.launch(neg_vec, dim=nworld, inputs=[self.g, nv], outputs=[self.rhs])
            launch_cholesky_solve(
                self._cholesky_solve, nworld=nworld, H=self.H,
                rhs=self.rhs, dq=self.delta,
            )
            wp.launch(
                lm_pred_reduction, dim=nworld,
                inputs=[self.delta, self.g, self.lam, nv], outputs=[self.pred],
            )
            # Propose q <- q (+) delta (unit step), evaluate trial cost.
            cfg.integrate_inplace(self.delta, dt=unit_dt)
            wp.launch(zero_cost, dim=nworld, inputs=[self.C_new])
            for task in tasks:
                error, cost = task.error_cost(cfg)
                k = int(error.shape[1])
                wp.launch(
                    accumulate_cost, dim=nworld,
                    inputs=[error, cost, k, self.C_new],
                )
            # Trust-region accept/reject + Nielsen lambda update (per world).
            wp.launch(
                lm_accept, dim=nworld,
                inputs=[
                    self.pred, self.C_old, self.C_new, self.qpos_old, self.delta,
                    self.lambda_min, self.lambda_max, self.eps, nq, nv,
                    cfg.q, self.lam, self.nu, self.dq_total,
                ],
            )
            # Refresh FK for the worlds that reverted.
            cfg.update()

    def _finalize_velocity(self, dt: float) -> None:
        cfg = self.configuration
        with wp.ScopedDevice(cfg.device):
            wp.launch(
                scale_velocity, dim=cfg.nworld,
                inputs=[self.dq_total, float(dt), cfg.nv], outputs=[self.v],
            )

    def invalidate_graph(self) -> None:
        self._graph = None
        self._graph_key = None

    def _ensure_graph(
        self, tasks: Sequence[Task], dt: float, iterations: int
    ) -> None:
        key = (tuple(tasks), dt, iterations)
        if self._graph is not None and self._graph_key == key:
            return
        cfg = self.configuration
        # Unit dt for the delta integrates lives on device; set outside capture.
        cfg.set_integration_dt(1.0)
        # Warm up all kernels / buffers before capture.
        for _ in range(iterations):
            self._iteration(tasks, unit_dt=None)
        self._finalize_velocity(dt)
        with wp.ScopedCapture() as capture:
            for _ in range(iterations):
                self._iteration(tasks, unit_dt=None)
            self._finalize_velocity(dt)
        self._graph = capture.graph
        self._graph_key = key
