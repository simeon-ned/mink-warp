"""Base class for hard kinematic limits enforced by the constrained solver.

A limit contributes a per-dof box on the tangent step ``dq``. Each limit
intersects its bounds into a shared per-world ``[lo, hi]`` buffer via
:meth:`Limit.apply_box` (``lo = max(lo, ...)``, ``hi = min(hi, ...)``), so
multiple single-dof limits compose by tightening — matching mink's stacked
``G dq <= h`` inequalities, which for the built-in limits are all single-dof rows.
"""

from __future__ import annotations

import abc

import warp as wp

from ..configuration import Configuration


class Limit(abc.ABC):
    """Abstract hard limit. Subclasses intersect a box into ``[lo, hi]``."""

    @abc.abstractmethod
    def apply_box(
        self,
        configuration: Configuration,
        dt: float,
        lo: wp.array,
        hi: wp.array,
    ) -> None:
        """Intersect this limit's bounds into the shared per-world box.

        Args:
            configuration: Current batched configuration.
            dt: Integration timestep [s].
            lo: Per-world lower bound on ``dq``, shape ``(nworld, nv)``, updated
                in place with ``lo = max(lo, this_lower)``.
            hi: Per-world upper bound on ``dq``, shape ``(nworld, nv)``, updated
                in place with ``hi = min(hi, this_upper)``.
        """
        raise NotImplementedError
