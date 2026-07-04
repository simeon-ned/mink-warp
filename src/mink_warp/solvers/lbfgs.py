"""L-BFGS IK backend (quasi-Newton, batched).

Same weighted least-squares cost as :class:`LMSolver`, but never forms the
Hessian. Each iteration builds a descent direction with the two-loop recursion
over a short ``(s, y)`` history, runs a parallel line search over a fixed set of
step sizes (best-cost per world), then updates the curvature history. Eager only
(the per-candidate step sizes make CUDA-graph capture impractical).
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from ..configuration import Configuration
from ..kernels.lbfgs import (
    accumulate_grad,
    apply_step,
    lbfgs_gamma,
    lbfgs_two_loop,
    ls_init,
    ls_update,
    push_history,
    shift_history,
    zero_grad,
)
from ..kernels.lm import accumulate_cost, zero_cost
from ..kernels.solver import scale_velocity
from ..tasks.task import Task
from .base import Solver

#: Default parallel line-search step sizes (largest first).
DEFAULT_LINE_SEARCH = (1.0, 0.5, 0.25, 0.1, 0.05)


class LBFGSSolver(Solver):
    """Batched limited-memory BFGS IK solver.

    Args:
        history: number ``m`` of ``(s, y)`` pairs retained.
        line_search: candidate step sizes evaluated in parallel each iteration.
        eps: sufficient-decrease floor for the line search.
        curvature_eps: relative curvature floor ``s.y > curvature_eps * ||y||^2``
            below which a pair is skipped (kept out of the history).
    """

    name = "lbfgs"

    def __init__(
        self,
        configuration: Configuration,
        history: int = 6,
        line_search: Sequence[float] = DEFAULT_LINE_SEARCH,
        eps: float = 1e-16,
        curvature_eps: float = 1e-8,
    ):
        super().__init__(configuration)
        if history < 1:
            raise ValueError(f"history must be >= 1, got {history}")
        self.m = int(history)
        self.alphas = tuple(float(a) for a in line_search)
        if not self.alphas:
            raise ValueError("line_search must contain at least one step size")
        self.eps = float(eps)
        self.curvature_eps = float(curvature_eps)

        nworld = configuration.nworld
        nv = configuration.nv
        nq = configuration.nq
        m = self.m
        with wp.ScopedDevice(configuration.device):
            self.g = wp.zeros((nworld, nv), dtype=float)
            self.g_new = wp.zeros((nworld, nv), dtype=float)
            self.p = wp.zeros((nworld, nv), dtype=float)
            self.q_work = wp.zeros((nworld, nv), dtype=float)
            self.step = wp.zeros((nworld, nv), dtype=float)
            self.dq_total = wp.zeros((nworld, nv), dtype=float)
            self.v = wp.zeros((nworld, nv), dtype=float)
            self.s_hist = wp.zeros((nworld, m, nv), dtype=float)
            self.y_hist = wp.zeros((nworld, m, nv), dtype=float)
            self.rho_hist = wp.zeros((nworld, m), dtype=float)
            self.alpha_scr = wp.zeros((nworld, m), dtype=float)
            self.gamma = wp.zeros(nworld, dtype=float)
            self.C_old = wp.zeros(nworld, dtype=float)
            self.C_new = wp.zeros(nworld, dtype=float)
            self.C_cand = wp.zeros(nworld, dtype=float)
            self.C_best = wp.zeros(nworld, dtype=float)
            self.best_alpha = wp.zeros(nworld, dtype=float)
            self.qpos_base = wp.zeros((nworld, nq), dtype=float)

    def solve_and_integrate(
        self,
        tasks: Sequence[Task],
        dt: float,
        *,
        iterations: int = 10,
        use_graph: bool = False,  # noqa: ARG002 - eager only
        **_ignored,
    ) -> wp.array:
        self._check_dt(dt)
        if iterations < 1:
            raise ValueError(f"iterations must be >= 1, got {iterations}")
        cfg = self.configuration
        nv = cfg.nv
        nworld = cfg.nworld

        with wp.ScopedDevice(cfg.device):
            self.dq_total.zero_()
            self.s_hist.zero_()
            self.y_hist.zero_()
            self.rho_hist.zero_()
            # Gradient / cost at the starting configuration.
            self._grad(tasks, self.g, self.C_old)

            count = 0
            for _ in range(iterations):
                wp.launch(lbfgs_gamma, dim=nworld,
                          inputs=[self.s_hist, self.y_hist, count, nv],
                          outputs=[self.gamma])
                wp.launch(lbfgs_two_loop, dim=nworld,
                          inputs=[self.g, self.s_hist, self.y_hist, self.rho_hist,
                                  self.gamma, count, nv],
                          outputs=[self.alpha_scr, self.q_work, self.p])
                self._line_search(tasks)
                # Apply the chosen step and accumulate the net motion.
                wp.copy(cfg.q, self.qpos_base)
                wp.launch(apply_step, dim=nworld,
                          inputs=[self.p, self.best_alpha, nv],
                          outputs=[self.step, self.dq_total])
                cfg.integrate_inplace(self.step, dt=1.0)
                # Gradient / cost at the new configuration.
                self._grad(tasks, self.g_new, self.C_new)
                # Curvature-guarded history update.
                if count == self.m:
                    wp.launch(shift_history, dim=nworld,
                              inputs=[self.m, nv],
                              outputs=[self.s_hist, self.y_hist, self.rho_hist])
                    slot = self.m - 1
                else:
                    slot = count
                    count += 1
                wp.launch(push_history, dim=nworld,
                          inputs=[self.step, self.g, self.g_new, slot,
                                  self.curvature_eps, nv],
                          outputs=[self.s_hist, self.y_hist, self.rho_hist])
                wp.copy(self.g, self.g_new)
                wp.copy(self.C_old, self.C_new)

            wp.launch(scale_velocity, dim=nworld,
                      inputs=[self.dq_total, float(dt), nv], outputs=[self.v])
        return self.v

    # Internal.

    def _grad(self, tasks: Sequence[Task], g_buf: wp.array, C_buf: wp.array) -> None:
        cfg = self.configuration
        nv = cfg.nv
        wp.launch(zero_grad, dim=cfg.nworld, inputs=[g_buf, C_buf, nv])
        for task in tasks:
            error, jac, cost = task.error_jacobian_cost(cfg)
            k = int(error.shape[1])
            wp.launch(accumulate_grad, dim=cfg.nworld,
                      inputs=[error, jac, cost, k, nv, g_buf, C_buf])

    def _line_search(self, tasks: Sequence[Task]) -> None:
        cfg = self.configuration
        nworld = cfg.nworld
        wp.copy(self.qpos_base, cfg.q)
        wp.launch(ls_init, dim=nworld, outputs=[self.C_best, self.best_alpha])
        for a in self.alphas:
            wp.copy(cfg.q, self.qpos_base)
            cfg.integrate_inplace(self.p, dt=a)  # q_base (+) a * p
            wp.launch(zero_cost, dim=nworld, inputs=[self.C_cand])
            for task in tasks:
                error, cost = task.error_cost(cfg)
                k = int(error.shape[1])
                wp.launch(accumulate_cost, dim=nworld,
                          inputs=[error, cost, k, self.C_cand])
            wp.launch(ls_update, dim=nworld,
                      inputs=[self.C_cand, self.C_old, float(a), self.eps],
                      outputs=[self.C_best, self.best_alpha])
