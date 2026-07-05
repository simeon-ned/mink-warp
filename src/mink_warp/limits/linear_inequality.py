"""Arbitrary constant linear inequality limit ``G dq <= h``.

The general-inequality escape hatch: any half-space (or stack of them) on the
tangent step that a per-dof box cannot express — an oriented plane in dof space,
a hand-written collision half-space, a coupled-joint bound. ``G`` and ``h`` are
constant (configuration-independent) and broadcast to every world, so this is
inequality-only (``box_capable = False``); it is enforced by the constrained
solver's general OSQP-ADMM path.

For configuration-dependent rows (e.g. collision avoidance, whose normals move
with ``q``) subclass :class:`~mink_warp.limits.Limit` and implement
``scatter_inequalities`` directly.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..kernels.constrained import linear_ineq_scatter
from .limit import Limit


class LinearInequalityLimit(Limit):
    """Constant dense inequality ``G dq <= h`` applied to every world."""

    box_capable = False

    def __init__(self, G: npt.ArrayLike, h: npt.ArrayLike):
        G_np = np.atleast_2d(np.asarray(G, dtype=np.float32))
        h_np = np.atleast_1d(np.asarray(h, dtype=np.float32))
        if G_np.ndim != 2:
            raise ValueError(f"G must be 2-D (m, nv); got shape {G_np.shape}")
        if G_np.shape[0] == 0:
            raise ValueError(
                "LinearInequalityLimit needs at least one row; an inequality "
                "limit with no inequalities constrains nothing."
            )
        if h_np.shape != (G_np.shape[0],):
            raise ValueError(
                f"h must have shape ({G_np.shape[0]},) to match G's rows; "
                f"got {h_np.shape}"
            )
        self._G_np = G_np
        self._h_np = h_np
        self.nv = int(G_np.shape[1])
        self.n_inequalities = int(G_np.shape[0])
        self._dev: dict[str, tuple[wp.array, wp.array]] = {}

    def _ensure_dev(self, device: str):
        cached = self._dev.get(device)
        if cached is not None:
            return cached
        with wp.ScopedDevice(device):
            arrs = (
                wp.array(self._G_np, dtype=float),
                wp.array(self._h_np, dtype=float),
            )
        self._dev[device] = arrs
        return arrs

    def scatter_inequalities(
        self,
        configuration: Configuration,
        dt: float,
        row_offset: int,
        G: wp.array,
        h: wp.array,
    ) -> None:
        del dt  # Constant rows are timestep-independent.
        if self.n_inequalities == 0:
            return
        if self.nv != configuration.nv:
            raise ValueError(
                f"LinearInequalityLimit G has nv={self.nv} but the configuration "
                f"has nv={configuration.nv}."
            )
        device = configuration.device
        Gc, hc = self._ensure_dev(device)
        with wp.ScopedDevice(device):
            wp.launch(
                linear_ineq_scatter,
                dim=(configuration.nworld, self.n_inequalities),
                inputs=[Gc, hc, int(row_offset)],
                outputs=[G, h],
            )
