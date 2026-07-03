"""All kinematic tasks derive from the :class:`Task` base class."""

from __future__ import annotations

import abc
from typing import NamedTuple

import numpy as np
import warp as wp

from ..configuration import Configuration
from ..exceptions import InvalidDamping, InvalidGain


class Objective(NamedTuple):
    r"""Quadratic objective :math:`\frac{1}{2} \Delta q^T H \Delta q + c^T \Delta q`.

    Device arrays: ``H`` is ``(nworld, nv, nv)``, ``c`` is ``(nworld, nv)``.
    """

    H: wp.array
    c: wp.array


class BaseTask(abc.ABC):
    """Base class for all tasks."""

    @abc.abstractmethod
    def compute_qp_objective(self, configuration: Configuration) -> Objective:
        raise NotImplementedError

    def compute_qp_residual(
        self, configuration: Configuration
    ) -> tuple[wp.array, wp.array, wp.array] | None:
        r"""Weighted least-squares residual on device, or ``None``.

        Returns ``(weighted_jacobian, weighted_error, mu)`` with shapes
        ``(nworld, k, nv)``, ``(nworld, k)``, ``(nworld,)``.
        """
        return None


class Task(BaseTask):
    r"""Abstract base class for kinematic tasks.

    Device-native: ``compute_error`` / ``compute_jacobian`` return ``wp.array``.
    """

    def __init__(
        self,
        cost: np.ndarray,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        if not 0.0 <= gain <= 1.0:
            raise InvalidGain("`gain` must be in the range [0, 1]")
        if lm_damping < 0.0:
            raise InvalidDamping("`lm_damping` must be >= 0")

        self.cost = np.asarray(cost, dtype=np.float64)
        self.gain = gain
        self.lm_damping = lm_damping

    @abc.abstractmethod
    def compute_error(self, configuration: Configuration) -> wp.array:
        """Task error on device, shape ``(nworld, k)``."""
        raise NotImplementedError

    @abc.abstractmethod
    def compute_jacobian(self, configuration: Configuration) -> wp.array:
        """Task Jacobian on device, shape ``(nworld, k, nv)``."""
        raise NotImplementedError
