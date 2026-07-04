"""Batched differential inverse kinematics — thin API over the solver backends.

``IKSolver`` is the damped-least-squares backend (Mink's differential step) and
stays the default. See :mod:`mink_warp.solvers` for the Levenberg-Marquardt and
L-BFGS backends, all sharing the :class:`~mink_warp.solvers.base.Solver` API.
"""

from __future__ import annotations

from collections.abc import Sequence

import warp as wp

from .configuration import Configuration
from .limits import Limit
from .solvers import (
    ConstrainedSolver,
    DLSSolver,
    LBFGSSolver,
    LMSolver,
    Solver,
    make_solver,
)
from .tasks.task import Task

# Backward-compatible default: the original mink-warp solver is damped LS.
IKSolver = DLSSolver

# Sentinel: "limits not passed" -> keep the historical unconstrained behaviour.
_UNSET = object()

__all__ = [
    "IKSolver",
    "DLSSolver",
    "LMSolver",
    "LBFGSSolver",
    "ConstrainedSolver",
    "Solver",
    "make_solver",
    "solve_ik",
    "solve_ik_iterations",
]


def solve_ik(
    configuration: Configuration,
    tasks: Sequence[Task],
    dt: float,
    damping: float | None = None,
    *,
    solver: Solver | None = None,
    limits: Sequence[Limit] | None = _UNSET,  # type: ignore[assignment]
) -> wp.array:
    """Solve one differential-IK step; returns velocity ``(nworld, nv)``.

    With the default (or any :class:`DLSSolver`) this is a pure velocity solve
    that does **not** mutate the configuration. Optimizer backends (LM / L-BFGS)
    advance the configuration and return the equivalent tangent velocity.

    ``damping=None`` uses the solver's own default (a supplied ``DLSSolver``
    keeps its configured damping); pass a float to override it for this call.

    ``limits`` enforces hard joint limits via a :class:`ConstrainedSolver`
    (mink-shaped): ``None`` uses the default :class:`ConfigurationLimit`, a list
    supplies specific limits, ``[]`` disables them. If ``limits`` is omitted the
    historical unconstrained behaviour is kept. Ignored when ``solver`` is given.
    """
    if solver is None:
        if limits is _UNSET:
            solver = DLSSolver(configuration)
        else:
            cs_kwargs = {} if damping is None else {"damping": damping}
            solver = ConstrainedSolver(
                configuration,
                limits=None if limits is None else list(limits),
                **cs_kwargs,
            )
    elif solver.configuration is not configuration:
        raise ValueError("solver was created for a different Configuration")
    if isinstance(solver, (LMSolver, LBFGSSolver)):
        return solver.solve_and_integrate(tasks, dt)
    return solver.solve(tasks, dt, damping=damping)


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
    elif solver.configuration is not configuration:
        raise ValueError("solver was created for a different Configuration")
    solver.solve_and_integrate(
        tasks, dt, damping=damping, iterations=iterations
    )
    return solver.configuration.q
