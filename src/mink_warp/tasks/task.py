r"""Kinematic task base classes.

In mink-warp, all tasks derive from :class:`Task`. Each task defines an error
:math:`e(q) \in \mathbb{R}^k` and Jacobian :math:`J(q) \in \mathbb{R}^{k \times n_v}`
evaluated in parallel for ``nworld`` configurations. The stacking formalism
matches Mink / Pink; see also `task-based inverse kinematics
<https://scaron.info/robot-locomotion/inverse-kinematics.html>`_.
"""

from __future__ import annotations

import abc

import numpy as np
import numpy.typing as npt
import warp as wp

from ..configuration import Configuration
from ..exceptions import InvalidDamping, InvalidGain, TargetNotSet
from ..interop import to_wp
from ..kernels.residual import weighted_residual


class Task(abc.ABC):
    r"""Abstract base class for kinematic tasks.

    Attributes:
        cost: Weight vector (same dimension as the task error). Units depend on
            the task (e.g. :math:`[\mathrm{cost}] / [\mathrm{m}]` for position).
        gain: Task gain :math:`\alpha \in [0, 1]` for low-pass filtering.
            Defaults to ``1.0`` (dead-beat).
        lm_damping: Unitless Levenberg–Marquardt scale (active when the error
            is large). Helps under infeasible targets.
    """

    k: int = 0
    #: False when :meth:`_eval` reads device state on the host (e.g. ``q.numpy()``);
    #: such tasks cannot participate in CUDA graph capture.
    supports_cuda_graph: bool = True

    def __init__(
        self,
        cost: npt.ArrayLike,
        gain: float = 1.0,
        lm_damping: float = 0.0,
    ):
        if not 0.0 <= gain <= 1.0:
            raise InvalidGain("`gain` must be in the range [0, 1]")
        if lm_damping < 0.0:
            raise InvalidDamping("`lm_damping` must be >= 0")
        self.cost = np.atleast_1d(np.asarray(cost, dtype=np.float64))
        self.gain = gain
        self.lm_damping = lm_damping
        self._error: wp.array | None = None
        self._jacobian: wp.array | None = None
        self._cost_dev: wp.array | None = None
        self._weighted_jac: wp.array | None = None
        self._weighted_err: wp.array | None = None
        self._mu: wp.array | None = None
        self._nworld: int | None = None
        self._device: str | None = None
        self._nv: int | None = None

    @abc.abstractmethod
    def _eval(self, configuration: Configuration) -> None:
        """Write ``self._error`` and ``self._jacobian`` on device."""

    def _alloc_extra_buffers(self, configuration: Configuration) -> None:
        """Allocate task-specific buffers (targets, etc.)."""

    def _ensure_buffers(self, configuration: Configuration) -> None:
        nworld = configuration.nworld
        device = configuration.device
        nv = configuration.nv
        if (
            self._nworld == nworld
            and self._device == device
            and self._nv == nv
            and self._error is not None
        ):
            return
        with wp.ScopedDevice(device):
            self._error = wp.zeros((nworld, self.k), dtype=float)
            self._jacobian = wp.zeros((nworld, self.k, nv), dtype=float)
            self._cost_dev = wp.array(self.cost.astype(np.float32), dtype=float)
            self._weighted_jac = wp.zeros((nworld, self.k, nv), dtype=float)
            self._weighted_err = wp.zeros((nworld, self.k), dtype=float)
            self._mu = wp.zeros(nworld, dtype=float)
            self._alloc_extra_buffers(configuration)
        self._nworld = nworld
        self._device = device
        self._nv = nv

    def compute_error(self, configuration: Configuration) -> wp.array:
        self._ensure_buffers(configuration)
        self._eval(configuration)
        assert self._error is not None
        return self._error

    def compute_jacobian(self, configuration: Configuration) -> wp.array:
        self._ensure_buffers(configuration)
        self._eval(configuration)
        assert self._jacobian is not None
        return self._jacobian

    def _ensure_cost_dev(self, configuration: Configuration) -> wp.array:
        if self._cost_dev is None:
            with wp.ScopedDevice(configuration.device):
                self._cost_dev = wp.array(self.cost.astype(np.float32), dtype=float)
        return self._cost_dev

    def error_jacobian_cost(
        self, configuration: Configuration
    ) -> tuple[wp.array, wp.array, wp.array]:
        """Raw ``(error, jacobian, cost)`` after a single ``_eval``.

        Used by the optimizer solvers (LM / L-BFGS), which own their damping and
        therefore need the unweighted, non-negated residual — not the
        ``gain``/``lm_damping``-shaped :meth:`compute_residual`.
        """
        self._ensure_buffers(configuration)
        self._eval(configuration)
        assert self._error is not None
        assert self._jacobian is not None
        return self._error, self._jacobian, self._ensure_cost_dev(configuration)

    def error_cost(
        self, configuration: Configuration
    ) -> tuple[wp.array, wp.array]:
        """Raw ``(error, cost)`` after a single ``_eval`` (trial-cost evaluation)."""
        self._ensure_buffers(configuration)
        self._eval(configuration)
        assert self._error is not None
        return self._error, self._ensure_cost_dev(configuration)

    def compute_residual(
        self, configuration: Configuration
    ) -> tuple[wp.array, wp.array, wp.array]:
        r"""Weighted residual ``(W, e, mu)`` for the IK normal equations.

        Tasks are stacked into a least-squares objective equivalent to Mink's QP
        cost:

        .. math::

            \frac{1}{2} \| W J \Delta q + \alpha e \|_2^2
            = \frac{1}{2} \Delta q^\top H \Delta q + c^\top \Delta q

        with :math:`H = \sum_i W_i^\top W_i + \mu I` and
        :math:`c = \sum_i -W_i^\top (\alpha e_i)`. Here :math:`W` is a diagonal
        weight matrix from ``cost``, :math:`\alpha` is ``gain``, and
        :math:`\mu` is the per-task LM term from ``lm_damping``.

        First-order task dynamics (per task, before stacking):

        .. math::

            J(q)\, \Delta q = -\alpha\, e(q)

        Args:
            configuration: Batched robot configuration :math:`q` with shape
                ``(nworld, nq)``.

        Returns:
            Weighted Jacobian :math:`WJ`, weighted error :math:`-\alpha W e`,
            and scalar LM damping :math:`\mu` per world.
        """
        self._ensure_buffers(configuration)
        self._eval(configuration)
        assert self._error is not None
        assert self._jacobian is not None
        assert self._weighted_jac is not None
        assert self._weighted_err is not None
        assert self._mu is not None
        with wp.ScopedDevice(configuration.device):
            # Refresh the device cost buffer if a cost setter nulled it
            # (same path the LM / L-BFGS accessors take).
            self._ensure_cost_dev(configuration)
            wp.launch(
                weighted_residual,
                dim=configuration.nworld,
                inputs=[
                    self._error,
                    self._jacobian,
                    self._cost_dev,
                    float(self.gain),
                    float(self.lm_damping),
                    self.k,
                    configuration.nv,
                ],
                outputs=[self._weighted_jac, self._weighted_err, self._mu],
            )
        return self._weighted_jac, self._weighted_err, self._mu

    # Mink-compatible alias.
    compute_qp_residual = compute_residual


class TargetedTask(Task):
    r"""Task with a batched device target buffer.

    Targets are ``wp.array`` with shape ``(nworld, target_width)`` or a single
    row broadcast to all worlds. Host uploads use :func:`~mink_warp.to_wp` at
    boundaries.
    """

    target_width: int = 0

    def __init__(self, cost, gain=1.0, lm_damping=0.0):
        super().__init__(cost, gain, lm_damping)
        self._target: wp.array | None = None
        self._pending: wp.array | None = None
        self._target_set = False

    def _alloc_extra_buffers(self, configuration: Configuration) -> None:
        self._target = wp.zeros(
            (configuration.nworld, self.target_width), dtype=float
        )

    def _set_pending(
        self,
        target: wp.array | npt.ArrayLike,
        *,
        configuration: Configuration | None = None,
    ) -> None:
        if isinstance(target, wp.array):
            self._pending = target
        else:
            arr = np.asarray(target, dtype=np.float32)
            self._pending = to_wp(arr, dtype=float)
        self._target_set = True
        if configuration is not None:
            self._ensure_buffers(configuration)
            self._flush_pending(configuration.nworld, configuration.device)
        elif self._target is not None and self._nworld is not None:
            self._flush_pending(self._nworld, self._device)

    def _flush_pending(self, nworld: int, device: str | None) -> None:
        assert self._target is not None and device is not None
        src = self._pending
        if src is None:
            if not self._target_set:
                raise TargetNotSet(self.__class__.__name__)
            return
        w = self.target_width
        with wp.ScopedDevice(device):
            if src.shape == (w,):
                self._broadcast_target(src, nworld)
            elif src.shape == (nworld, w):
                wp.copy(self._target, src)
            else:
                raise TargetNotSet(
                    f"{self.__class__.__name__}: target shape {src.shape} "
                    f"incompatible with nworld={nworld}, width={w}"
                )
        self._pending = None

    def _broadcast_target(self, src: wp.array, nworld: int) -> None:
        """Broadcast a single target row to all worlds. Override if needed."""
        assert self._target is not None
        # Generic: tile via numpy upload (rare path). Prefer override with kernel.
        row = src.numpy()
        tiled = np.broadcast_to(row, (nworld, self.target_width)).copy()
        self._target.assign(tiled)

    def _require_target(self, configuration: Configuration) -> wp.array:
        if not self._target_set:
            raise TargetNotSet(self.__class__.__name__)
        self._ensure_buffers(configuration)
        self._flush_pending(configuration.nworld, configuration.device)
        assert self._target is not None
        return self._target
