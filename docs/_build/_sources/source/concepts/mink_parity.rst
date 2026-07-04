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

**Limits.** Mink enforces hard limits inside the QP. mink-warp offers:

- Soft: :class:`~mink_warp.tasks.JointLimitTask` (least-squares penalty)
- Hard: :class:`~mink_warp.limits.ConfigurationLimit` via
  :class:`~mink_warp.solvers.ConstrainedSolver` or ``solve_ik(..., limits=…)``

**Integration.** Both use MuJoCo's position integrator semantics; mink-warp
routes through ``mjwarp`` and uses out-of-place ``qpos`` writes for CUDA graphs.

Porting checklist
-----------------

1. Replace ``Configuration(model)`` with ``Configuration(model, nworld=B)``.
2. Upload targets once: ``task.set_target(wp.array(...))`` or set from configuration.
3. Replace ``solve_ik(..., "daqp")`` with ``solve_ik(...)`` or a explicit
   :class:`~mink_warp.solvers.DLSSolver`.
4. Keep ``integrate_inplace`` in the loop unless using ``solve_and_integrate``.
5. Run parity tests: ``uv run pytest tests/ -k mink`` (requires ``mink`` extra).

Related
-------

- `Mink documentation <https://kevinzakka.github.io/mink/>`_
- :doc:`../tutorial/tasks_and_limits`
- :doc:`../benchmarks`
