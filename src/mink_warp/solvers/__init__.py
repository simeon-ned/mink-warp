"""Interchangeable batched IK solver backends.

All backends share the :class:`Solver` interface and minimise the same weighted
least-squares task cost, so they can be swapped in a control loop or benchmark:

    solver = make_solver(configuration, "lm")      # or "dls" / "lbfgs"
    solver.solve_and_integrate(tasks, dt, iterations=10)
"""

from __future__ import annotations

from ..configuration import Configuration
from .base import Solver
from .constrained import ConstrainedSolver
from .dls import DLSSolver
from .lbfgs import LBFGSSolver
from .lm import LMSolver

#: Registry of solver kinds by name.
SOLVERS: dict[str, type[Solver]] = {
    DLSSolver.name: DLSSolver,
    LMSolver.name: LMSolver,
    LBFGSSolver.name: LBFGSSolver,
    ConstrainedSolver.name: ConstrainedSolver,
}


def make_solver(
    configuration: Configuration, kind: str = "dls", **kwargs
) -> Solver:
    """Construct a solver backend by name.

    ``"dls"`` / ``"lm"`` / ``"lbfgs"`` / ``"constrained"`` (the last enforces
    hard joint limits; pass ``limits=`` to override the default ConfigurationLimit).
    """
    try:
        cls = SOLVERS[kind]
    except KeyError:
        raise ValueError(
            f"unknown solver {kind!r}; choose from {sorted(SOLVERS)}"
        ) from None
    return cls(configuration, **kwargs)


__all__ = [
    "Solver",
    "DLSSolver",
    "LMSolver",
    "LBFGSSolver",
    "ConstrainedSolver",
    "SOLVERS",
    "make_solver",
]
