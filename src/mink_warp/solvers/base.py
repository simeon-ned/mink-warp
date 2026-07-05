"""Common solver interface shared by every batched IK backend.

Every solver minimises the same per-world weighted least-squares cost

    C(q) = 1/2 * sum_tasks || cost_i * error_i(q) ||^2

and exposes the identical entry point :meth:`Solver.solve_and_integrate`, so the
backend (:class:`DLSSolver`, :class:`LMSolver`, :class:`LBFGSSolver`) is
interchangeable in a control loop or benchmark.
"""

from __future__ import annotations

import abc
from collections.abc import Sequence

import warp as wp

from ..configuration import Configuration
from ..tasks.task import Task


class Solver(abc.ABC):
    """Batched IK solver operating on a shared :class:`Configuration`.

    Contract: :meth:`solve_and_integrate` advances the configuration toward the
    task targets and returns a representative tangent velocity ``(nworld, nv)``.

    For multi-iteration backends (LM / L-BFGS) the returned ``v`` is the sum of
    the per-iteration tangent steps divided by ``dt``. Re-integrating ``v`` over
    ``dt`` reproduces the optimized configuration exactly for Euclidean joints
    (hinge / slide); for free / ball joints it agrees to second order (the true
    update is a composition of manifold exponentials, ``v`` their tangent sum).
    The configuration itself is always left at the exact optimized state.
    A solver is bound to its configuration's ``nworld`` / ``nv`` and cannot be
    reused with a differently sized one.
    """

    #: Registry key / human label.
    name: str = "solver"

    #: Whether this backend enforces hard limits. Only :class:`ConstrainedSolver`
    #: sets this True; the cost-only backends (DLS / LM / L-BFGS) cannot honour a
    #: ``limits=`` argument and callers must not silently assume they do.
    supports_limits: bool = False

    def __init__(self, configuration: Configuration):
        self.configuration = configuration

    @abc.abstractmethod
    def solve_and_integrate(
        self,
        tasks: Sequence[Task],
        dt: float,
        *,
        iterations: int = 1,
        use_graph: bool = False,
        **kwargs,
    ) -> wp.array:
        """Advance ``configuration`` and return the tangent velocity."""

    def step(
        self,
        tasks: Sequence[Task],
        dt: float,
        *,
        iterations: int = 1,
        **kwargs,
    ) -> wp.array:
        """Alias for :meth:`solve_and_integrate` (no graph capture)."""
        return self.solve_and_integrate(
            tasks, dt, iterations=iterations, **kwargs
        )

    def invalidate_graph(self) -> None:
        """Drop any captured CUDA graph. Backends without a graph no-op."""

    @staticmethod
    def _check_dt(dt: float) -> None:
        if dt <= 0.0:
            raise ValueError(f"dt must be > 0, got {dt}")
