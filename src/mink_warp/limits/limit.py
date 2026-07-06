"""Base class for hard kinematic limits enforced by the constrained solver.

A limit constrains the tangent step ``dq`` and can expose either or both of two
forms, matching mink's stacked ``G dq <= h`` inequalities:

* **box** — a per-dof interval, intersected into a shared per-world ``[lo, hi]``
  buffer via :meth:`Limit.apply_box` (``lo = max(lo, ...)``,
  ``hi = min(hi, ...)``). Single-dof limits compose by tightening. This is the
  fast default path (solved exactly at every ADMM step by the box kernel).

* **dense inequality** — general rows ``G dq <= h`` scattered into a shared
  padded ``(G, h)`` buffer via :meth:`Limit.scatter_inequalities`. Used by the
  general-inequality solve path, and the *only* form for constraints a per-dof
  box cannot express (an arbitrary half-space, collision avoidance, ...).

A limit advertises ``n_inequalities`` (how many dense rows it contributes; ``0``
means it has no dense form) and ``box_capable`` (whether :meth:`apply_box`
works). The built-in joint/velocity limits support both; a general half-space
limit is inequality-only.
"""

from __future__ import annotations

import abc

import warp as wp

from ..configuration import Configuration


class Limit(abc.ABC):
    r"""Abstract hard kinematic limit.

    Limits constrain the tangent step :math:`\Delta q \in T_q(\mathcal{C})`
    and are enforced by :class:`~mink_warp.ConstrainedSolver`. Each limit may
    expose:

    * **box** — per-dof bounds :math:`\ell \leq \Delta q \leq u`
    * **dense rows** — :math:`G(q)\, \Delta q \leq h(q)`
    """

    #: Number of dense ``G dq <= h`` rows this limit contributes (0 = box-only).
    n_inequalities: int = 0
    #: Whether :meth:`apply_box` is implemented (False for inequality-only limits).
    box_capable: bool = True
    #: False when inequality assembly reads device state on the host each step.
    supports_cuda_graph: bool = True

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
        raise NotImplementedError(
            f"{type(self).__name__} is inequality-only (box_capable=False); "
            f"it has no per-dof box form."
        )

    def scatter_inequalities(
        self,
        configuration: Configuration,
        dt: float,
        row_offset: int,
        G: wp.array,
        h: wp.array,
    ) -> None:
        r"""Write dense inequality rows into shared buffers.

        Row :math:`i` encodes :math:`G_i \Delta q \leq h_i`. Buffers have shape
        ``(nworld, m, nv)`` and ``(nworld, m)``; only rows owned by this limit
        are overwritten.

        Args:
            configuration: Current batched configuration.
            dt: Timestep :math:`\mathrm{d}t` in [s].
            row_offset: First row index for this limit.
            G: Shared inequality matrix.
            h: Shared bound vector.
        """
        raise NotImplementedError(
            f"{type(self).__name__} declares n_inequalities="
            f"{self.n_inequalities} but does not implement scatter_inequalities."
        )
