"""Build and solve the differential inverse kinematics problem."""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from .configuration import Configuration
from .solver_kernels import (
    accumulate_normal_equations,
    add_damping_diag,
    cholesky_solve_batched,
    neg_vec,
    scale_velocity,
    zero_normal_equations,
)
from .tasks import BaseTask


class IKSolver:
    """Reusable batched DLS IK solver.

    Owns device buffers for the normal equations
    :math:`(H + \\lambda I)\\Delta q = -c` and returns velocities
    :math:`v = \\Delta q / dt`, matching Mink's differential-IK output
    (without hard QP inequalities).

    Inspired by Newton's ``IKSolver`` buffer ownership; the math follows
    Mink's residual stacking (``H = W^T W``).
    """

    def __init__(self, configuration: Configuration):
        self.configuration = configuration
        nworld = configuration.nworld
        nv = configuration.nv
        device = configuration.device
        with wp.ScopedDevice(device):
            self.H = wp.zeros((nworld, nv, nv), dtype=float)
            self.c = wp.zeros((nworld, nv), dtype=float)
            self.mu_total = wp.zeros(nworld, dtype=float)
            self.rhs = wp.zeros((nworld, nv), dtype=float)
            self.dq = wp.zeros((nworld, nv), dtype=float)
            self.v = wp.zeros((nworld, nv), dtype=float)

    def solve(
        self,
        tasks: Sequence[BaseTask],
        dt: float,
        damping: float = 1e-12,
    ) -> wp.array:
        """Solve one differential-IK step.

        Args:
            tasks: Kinematic tasks (device residuals).
            dt: Integration timestep [s].
            damping: Global Levenberg-Marquardt damping on all dofs.

        Returns:
            Velocity ``v`` on device, shape ``(nworld, nv)``.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")

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
                residual = task.compute_qp_residual(cfg)
                if residual is None:
                    raise NotImplementedError(
                        f"{type(task).__name__} does not expose a QP residual; "
                        "only least-squares tasks are supported in v0."
                    )
                W, e, mu = residual
                k = int(W.shape[1])
                # In-place accumulate: H += WᵀW, c += -eᵀW (Mink form).
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

            # Device solve: H dq = -c (Cholesky; overwrites H with L).
            wp.launch(
                neg_vec,
                dim=nworld,
                inputs=[self.c, nv],
                outputs=[self.rhs],
            )
            wp.launch(
                cholesky_solve_batched,
                dim=nworld,
                inputs=[self.H, self.rhs, nv],
                outputs=[self.dq],
            )

            wp.launch(
                scale_velocity,
                dim=nworld,
                inputs=[self.dq, float(dt), nv],
                outputs=[self.v],
            )
        return self.v

    def step(
        self,
        tasks: Sequence[BaseTask],
        dt: float,
        damping: float = 1e-12,
        iterations: int = 1,
    ) -> wp.array:
        """Solve and integrate ``iterations`` times; returns velocity of last step."""
        v = self.v
        for _ in range(iterations):
            v = self.solve(tasks, dt, damping=damping)
            self.configuration.integrate_inplace(v, dt)
        return v


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    damping: float = 1e-12,
    *,
    solver: IKSolver | None = None,
) -> wp.array:
    """Solve the differential inverse kinematics problem.

    Unconstrained least-squares form of Mink's IK (no hard limits / equalities):

    .. math::

        (W^T W + \\lambda I)\\,\\Delta q = -W^T e,\\quad v = \\Delta q / dt

    Args:
        configuration: Batched robot configuration.
        tasks: Kinematic tasks.
        dt: Integration timestep [s].
        damping: Global LM damping.
        solver: Optional reusable :class:`IKSolver` (avoids reallocating buffers).

    Returns:
        Velocity on device, shape ``(nworld, nv)``.
    """
    if solver is None:
        solver = IKSolver(configuration)
    elif solver.configuration is not configuration:
        raise ValueError("IKSolver was created for a different Configuration")
    return solver.solve(tasks, dt, damping=damping)


def solve_ik_iterations(
    configuration: Configuration,
    tasks: Sequence[BaseTask],
    dt: float,
    iterations: int = 10,
    damping: float = 1e-2,
    *,
    solver: IKSolver | None = None,
) -> wp.array:
    """Run ``iterations`` DLS steps with integration; returns final ``q``.

    Useful for open-loop reach problems (Newton-style multi-iter solve).
    """
    if solver is None:
        solver = IKSolver(configuration)
    solver.step(tasks, dt, damping=damping, iterations=iterations)
    return configuration.q
