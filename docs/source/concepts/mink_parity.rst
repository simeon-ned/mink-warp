.. _mink-parity:

Mink API parity
===============

mink-warp mirrors `Mink <https://github.com/kevinzakka/mink>`_ [Mink]_ where it helps
porting examples and tests. The implementation differs: Mink solves a QP on CPU
with ``qpsolvers``; mink-warp assembles normal equations on GPU and uses Warp
linear solvers.

What matches
------------

.. list-table::
   :header-rows: 1
   :widths: 32 68

   * - Mink
     - mink-warp
   * - ``Configuration``
     - ``Configuration(model, nworld=…)`` — batched; ``q`` is ``wp.array``
   * - ``FrameTask``, ``PostureTask``, ``ComTask``, ``DampingTask``
     - Same names; body-frame Jacobian convention
   * - ``ConfigurationLimit`` / joint-limit task
     - ``JointLimitTask`` / ``ConfigurationLimitTask`` (soft) + ``ConfigurationLimit`` (hard)
   * - ``VelocityLimit``
     - :class:`~mink_warp.limits.VelocityLimit` (hard, box + dense rows)
   * - Custom ``Limit.compute_qp_inequalities`` (Mink)
     - :class:`~mink_warp.limits.Limit.scatter_inequalities` + ``LinearInequalityLimit``; subclass for ``q``-dependent rows
   * - ``SE3``, ``SO3``
     - Host Lie helpers for targets; device ops in ``lie/wp_ops``
   * - ``solve_ik(configuration, tasks, dt)``
     - Same call shape; returns ``wp.array`` velocity
   * - Residual form :math:`H = W^T W`, :math:`c = -W^T e`
     - Same stacking in ``compute_residual``

What differs
------------

**Batching.** Every buffer is leading-dimension ``nworld``. Targets are
``wp.array (nworld, …)`` or broadcast from a single pose.

**Device types.** Hot-path arrays are ``wp.array``. Use ``.numpy()``,
``to_wp()``, or ``*_numpy`` / ``*_se3`` helpers at boundaries.

**Solvers.** Mink selects a QP backend (``"daqp"``, etc.). mink-warp uses
:class:`~mink_warp.solvers.DLSSolver` by default; LM / L-BFGS / constrained
backends are GPU-native (see :doc:`../workflows/solvers`).

**Limits.** Mink enforces hard limits inside the QP (``G Δq ≤ h``). mink-warp offers:

- Soft: :class:`~mink_warp.tasks.JointLimitTask` (least-squares penalty)
- Hard: :class:`~mink_warp.limits.ConfigurationLimit`, :class:`~mink_warp.limits.VelocityLimit`,
  :class:`~mink_warp.limits.LinearInequalityLimit` via
  :class:`~mink_warp.solvers.ConstrainedSolver` or ``solve_ik(..., limits=…)``

  Box limits use a fast box-ADMM path; general rows use reduced OSQP-ADMM
  (see :doc:`../workflows/constrained`). ``solve_ik(..., limits=None)`` matches
  Mink's default ``ConfigurationLimit``.

**Integration.** Both use MuJoCo's position integrator semantics; mink-warp
routes through ``mjwarp`` and uses out-of-place ``qpos`` writes for CUDA graphs.

Porting checklist
-----------------

1. Replace ``Configuration(model)`` with ``Configuration(model, nworld=B)``.
2. Upload targets once: ``task.set_target(wp.array(...))`` or set from configuration.
3. Replace ``solve_ik(..., "daqp")`` with ``solve_ik(...)`` (unconstrained) or
   ``solve_ik(..., limits=None)`` (Mink default joint limit).
4. Keep ``integrate_inplace`` in the loop unless using ``solve_and_integrate``.
5. Run parity tests: ``uv run pytest tests/ -k mink`` (requires ``mink`` extra).

Related
-------

- `Mink documentation <https://kevinzakka.github.io/mink/>`_
- :doc:`../workflows/constrained`
- :doc:`../tutorial/tasks_and_limits`
- :doc:`../benchmarks`
