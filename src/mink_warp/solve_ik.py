"""Batched differential inverse kinematics — thin API over the solver backends.

``IKSolver`` is the damped-least-squares backend (Mink's differential step) and
stays the default. See :mod:`mink_warp.solvers` for the Levenberg-Marquardt and
L-BFGS backends, all sharing the :class:`~mink_warp.solvers.base.Solver` API.
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from .configuration import Configuration
from .solvers import DLSSolver, LBFGSSolver, LMSolver, Solver, make_solver
from .tasks.task import Task

# Backward-compatible default: the original mink-warp solver is damped LS.
IKSolver = DLSSolver

__all__ = [
    "IKSolver",
    "DLSSolver",
    "LMSolver",
    "LBFGSSolver",
    "Solver",
    "make_solver",
    "solve_ik",
    "solve_ik_iterations",
]


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    dt: float,
    damping: float = 1e-12,
    *,
    solver: Solver | None = None,
) -> wp.array:
    """Solve one differential-IK step; returns velocity ``(nworld, nv)``.

    With the default (or any :class:`DLSSolver`) this is a pure velocity solve
    that does **not** mutate the configuration. Optimizer backends (LM / L-BFGS)
    advance the configuration and return the equivalent tangent velocity.
    """
    if solver is None:
        solver = DLSSolver(configuration)
    elif solver.configuration is not configuration:
        raise ValueError("solver was created for a different Configuration")
    if isinstance(solver, DLSSolver):
        return solver.solve(tasks, dt, damping=damping)
    return solver.solve_and_integrate(tasks, dt)


def solve_ik_iterations(
    configuration: Configuration,
    tasks: Sequence[Task],
    dt: float,
    iterations: int = 10,
    damping: float = 1e-2,
    *,
    solver: Solver | None = None,
) -> wp.array:
    """Run ``iterations`` solve+integrate steps; returns final ``q``."""
    if solver is None:
        solver = DLSSolver(configuration)
    solver.solve_and_integrate(
        tasks, dt, damping=damping, iterations=iterations
    )
    return configuration.q
