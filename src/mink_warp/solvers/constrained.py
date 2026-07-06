"""Constrained (box-QP) IK backend enforcing hard joint limits.

Solves, per world, the same QP as mink::

    min_dq  1/2 dq^T H dq + c^T dq   s.t.   lo <= dq <= hi

where ``H, c`` are assembled from the task stack exactly as :class:`DLSSolver`
does. Two solve paths, auto-selected from the supplied limits:

* **box** (default, fast) — when every limit is a per-dof interval
  (:class:`~mink_warp.limits.ConfigurationLimit` /
  :class:`~mink_warp.limits.VelocityLimit`), their ``[lo, hi]`` is intersected
  and solved by OSQP-style box-ADMM (factor ``M = H + rho I`` once, then fixed
  cached-solve + box-clip + dual-update iterations). The returned step lies
  inside ``[lo, hi]`` at *every* iteration, so joint / velocity limits are never
  violated even when the target drives the arm hard into a bound or the loop is
  truncated.

* **general inequality** — when any limit contributes dense rows a box cannot
  express (:class:`~mink_warp.limits.LinearInequalityLimit`, or any box limit
  when ``use_inequalities=True``), the stacked ``G dq <= h`` is solved by the
  reduced Schur-normal OSQP-ADMM (factor ``M = H + sigma I + rho G^T G`` once,
  project the constraint image onto ``(-inf, h]`` each step). Feasibility is
  reached asymptotically (tightening with ``admm_iters``) rather than exactly.

Both paths run one block per world with tile Cholesky (which also runs on CPU
under Warp's LLVM backend, so the whole solver is testable without a GPU).
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from ..configuration import Configuration
from ..kernels.constrained import (
    compute_rho,
    get_admm_box_kernel,
    get_admm_ineq_kernel,
    init_box,
    init_ineq,
    launch_admm_box_solve,
    launch_admm_ineq_solve,
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
    r"""Batched constrained IK via ADMM on the task normal equations.

    Per world, solves:

    .. math::

        \min_{\Delta q}\ \tfrac{1}{2} \Delta q^\top H \Delta q + c^\top \Delta q
        \quad \text{s.t.}\ \ell \leq \Delta q \leq u,\ G \Delta q \leq h

    where :math:`H, c` match :class:`~mink_warp.DLSSolver`. Box limits are
    enforced exactly each ADMM iteration; general rows converge with
    ``admm_iters``.

    Args:
        configuration: Batched configuration to advance.
        limits: Hard limits. ``None`` → default :class:`~mink_warp.ConfigurationLimit`;
            ``[]`` → unconstrained regularized DLS inside the solver.
        admm_iters: Inner ADMM iterations per solve.
        rho_scale, rho_min, rho_max: ADMM penalty
            :math:`\rho = \mathrm{clamp}(\rho_{\mathrm{scale}} \sqrt{h_{\min} h_{\max}}, \ldots)`.
        alpha: Over-relaxation in :math:`[1, 2)`.
        damping: Tikhonov :math:`\lambda` on :math:`H` (matches ``solve_ik``).
        sigma: SPD floor for the general-inequality path.
        use_inequalities: Force dense :math:`G \Delta q \leq h` even for box
            limits. Auto-enabled for inequality-only limits such as
            :class:`~mink_warp.CollisionAvoidanceLimit`.
    """

    name = "constrained"
    supports_limits = True

    def __init__(
        self,
        configuration: Configuration,
        limits: Sequence[Limit] | None = None,
        *,
        admm_iters: int = 30,
        rho_scale: float = 1.0,
        rho_min: float = 1e-6,
        rho_max: float = 1e6,
        alpha: float = 1.6,
        damping: float = 1e-12,
        sigma: float = 1e-6,
        use_inequalities: bool = False,
    ):
        super().__init__(configuration)
        if admm_iters < 1:
            raise ValueError(f"admm_iters must be >= 1, got {admm_iters}")
        if rho_min <= 0.0:
            raise ValueError(
                f"rho_min must be > 0 (SPD safeguard for the H+rho*I Cholesky "
                f"factor), got {rho_min}"
            )
        if sigma <= 0.0:
            raise ValueError(
                f"sigma must be > 0 (SPD safeguard for the H+sigma*I+rho*G^T G "
                f"factor in the general-inequality path), got {sigma}"
            )
        if limits is None:
            limits = [ConfigurationLimit(configuration.model)]
        self.limits = list(limits)
        self.admm_iters = int(admm_iters)
        self.default_damping = damping
        self.rho_scale = float(rho_scale)
        self.rho_min = float(rho_min)
        self.rho_max = float(rho_max)
        self.alpha = float(alpha)
        self.sigma = float(sigma)

        # Choose the solve path. Any inequality-only limit forces the general
        # path; box-only limits use it only when explicitly requested. A path
        # needs at least one dense row, else it degenerates to the box path.
        self.n_ineq = sum(int(getattr(lim, "n_inequalities", 0)) for lim in self.limits)
        all_box = all(getattr(lim, "box_capable", True) for lim in self.limits)
        # An inequality-only limit that yields no rows can't fall back to the box
        # path (it has no box form) — fail loud rather than crash later in
        # apply_box with a misleading message.
        if not all_box and self.n_ineq == 0:
            raise ValueError(
                "an inequality-only limit (box_capable=False) contributes 0 "
                "rows; it has no box form to fall back on."
            )
        self._use_ineq = (use_inequalities or not all_box) and self.n_ineq > 0

        nworld = configuration.nworld
        nv = configuration.nv
        with wp.ScopedDevice(configuration.device):
            self.H = wp.zeros((nworld, nv, nv), dtype=float)
            self.c = wp.zeros((nworld, nv), dtype=float)
            self.mu_total = wp.zeros(nworld, dtype=float)
            self.rhs = wp.zeros((nworld, nv), dtype=float)  # b = -c = W^T e
            self.rho = wp.zeros(nworld, dtype=float)
            self.dq = wp.zeros((nworld, nv), dtype=float)
            self.v = wp.zeros((nworld, nv), dtype=float)
            if self._use_ineq:
                self.G = wp.zeros((nworld, self.n_ineq, nv), dtype=float)
                self.h = wp.zeros((nworld, self.n_ineq), dtype=float)
            else:
                self.lo = wp.zeros((nworld, nv), dtype=float)
                self.hi = wp.zeros((nworld, nv), dtype=float)
        if self._use_ineq:
            self._admm = get_admm_ineq_kernel(nv, self.n_ineq, self.admm_iters)
        else:
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
            and all(getattr(t, "supports_cuda_graph", True) for t in tasks)
            and all(getattr(lim, "supports_cuda_graph", True) for lim in self.limits)
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
            # --- Constraint block from the hard limits. ---
            if self._use_ineq:
                # Dense G dq <= h: reset to inert rows, then each limit scatters.
                wp.launch(init_ineq, dim=(nworld, self.n_ineq), outputs=[self.G, self.h])
                offset = 0
                for limit in self.limits:
                    rows = int(getattr(limit, "n_inequalities", 0))
                    if rows > 0:
                        limit.scatter_inequalities(cfg, dt, offset, self.G, self.h)
                        offset += rows
            else:
                # Box [lo, hi]: reset to (-inf, +inf), then each limit tightens.
                wp.launch(init_box, dim=(nworld, nv), outputs=[self.lo, self.hi])
                for limit in self.limits:
                    limit.apply_box(cfg, dt, self.lo, self.hi)
            # --- Per-world ADMM penalty. ---
            wp.launch(
                compute_rho,
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
            # --- ADMM solve -> dq (tile Cholesky). ---
            if self._use_ineq:
                launch_admm_ineq_solve(
                    self._admm,
                    nworld=nworld,
                    H=self.H,
                    b=self.rhs,
                    G=self.G,
                    h=self.h,
                    rho=self.rho,
                    sigma=self.sigma,
                    alpha=self.alpha,
                    dq=self.dq,
                )
            else:
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
