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
    r"""Compute joint velocity tangent to the current configuration.

    Differential IK minimizes a stacked task objective. Unconstrained
    (:class:`DLSSolver`) solves the normal equations

    .. math::

        (H + \lambda I)\, v = -c, \qquad v = \frac{\Delta q}{\mathrm{d}t}

    where :math:`H, c` come from :meth:`~mink_warp.Task.compute_residual` and
    :math:`\lambda` is Tikhonov ``damping`` in
    :math:`[\mathrm{cost}]^2 / [\mathrm{tangent}]`.

    With hard limits, :class:`ConstrainedSolver` solves (per world):

    .. math::

        \begin{aligned}
        \min_{\Delta q}\ & \tfrac{1}{2} \Delta q^\top H \Delta q + c^\top \Delta q \\
        \text{s.t.}\ & \ell \leq \Delta q \leq u \quad \text{(box limits)} \\
        & G \Delta q \leq h \quad \text{(general inequalities)}
        \end{aligned}

    Args:
        configuration: Batched configuration; FK must be current.
        tasks: Soft objectives to satisfy at weighted best.
        dt: Integration timestep :math:`\mathrm{d}t` in [s].
        damping: Tikhonov weight :math:`\lambda` on :math:`H` (solver default
            when ``None``).
        solver: Backend instance. ``None`` auto-builds :class:`DLSSolver` or
            :class:`ConstrainedSolver` from ``limits``.
        limits: Hard limits (Mink-shaped). ``None`` → default
            :class:`ConfigurationLimit`; ``[]`` → none; omitted → unconstrained.

    Returns:
        Velocity :math:`v` with shape ``(nworld, nv)``.
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
    else:
        if solver.configuration is not configuration:
            raise ValueError("solver was created for a different Configuration")
        if limits is not _UNSET:
            if not getattr(solver, "supports_limits", False):
                raise ValueError(
                    f"{type(solver).__name__} does not support limits (it is a "
                    f"cost-only backend); pass a ConstrainedSolver, or omit the "
                    f"solver= argument to build one automatically from limits=."
                )
            raise ValueError(
                "limits= is honoured only when the solver is auto-built "
                "(solver=None); an explicit ConstrainedSolver already has its "
                "limits fixed at construction. Drop limits= here, or configure "
                "them on the solver you pass in."
            )
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
