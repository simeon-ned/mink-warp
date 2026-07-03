"""Optional host↔device helpers. Hot path should use ``wp.array`` only."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import warp as wp


def to_wp(
    data: npt.ArrayLike | wp.array,
    *,
    dtype=float,
    device: str | None = None,
) -> wp.array:
    """Upload ``data`` to a Warp array, or return it unchanged if already one.

    Use only at API boundaries (script init, tests). Prefer keeping ``wp.array``
    end-to-end in the solve loop.
    """
    if isinstance(data, wp.array):
        return data
    arr = np.ascontiguousarray(data)
    if device is not None:
        with wp.ScopedDevice(device):
            return wp.array(arr, dtype=dtype)
    return wp.array(arr, dtype=dtype)
