.. _architecture:

Architecture overview
=====================

How a batched IK step flows through mink-warp. For the Mink comparison, see
:doc:`mink_parity`.

Pipeline
--------

.. code-block:: text

   mujoco.MjModel
        ↓
   Configuration(nworld)     # mjwarp model/data, device qpos
        ↓
   Task.compute_residual() # W, e, mu per task (device)
        ↓
   Solver backend          # assemble H, c; linear / ADMM solve
        ↓
   integrate_inplace(v)    # mjwarp position step (out-of-place qpos)

Everything above the dashed host boundary stays on device in the hot path.

Configuration
-------------

:class:`~mink_warp.Configuration` owns:

- ``wp_model`` / ``wp_data`` from MuJoCo Warp (batched ``nworld``)
- ``q`` as ``wp.array (nworld, nq)``
- FK via ``mjwarp.kinematics`` + ``com_pos``
- Body-frame Jacobians via ``mjwarp.jac`` + a frame transform kernel
- Integration via ``integrate_qpos`` (graph-safe out-of-place writes)

Tasks
-----

Tasks inherit from :class:`~mink_warp.tasks.task.Task` or
:class:`~mink_warp.tasks.task.TargetedTask`. Each implements
:meth:`~mink_warp.tasks.task.Task.compute_residual`, returning weighted
Jacobian rows ``W``, error ``e``, and optional Levenberg–Marquardt damping
``mu`` — the same stacking convention as Mink [FrameTaskJacobian]_.

Solvers
-------

Solver backends share :class:`~mink_warp.solvers.base.Solver`:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Backend
     - One step
     - Notes
   * - ``DLSSolver`` (``IKSolver`` alias)
     - :math:`v = \Delta q / dt` from damped normal equations
     - Default; optional CUDA graph
   * - ``LMSolver``
     - LM step on configuration
     - Newton-style tiled Cholesky
   * - ``LBFGSSolver``
     - Quasi-Newton step
     - Multi-iter ``solve_and_integrate``
   * - ``ConstrainedSolver``
     - Box-QP / ADMM with hard limits
     - GPU-only; mink-shaped limits API

Kernels
-------

Low-level Warp code lives under ``mink_warp/kernels/`` (frame Jacobians,
residual stacking, tile Cholesky, ADMM box projection). Public code should
call tasks and solvers, not kernels directly.

Related
-------

- :doc:`mink_parity`
- :doc:`../workflows/solvers`
- :doc:`../api/index`
