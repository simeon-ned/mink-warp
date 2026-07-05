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

Tasks inherit from :class:`~mink_warp.Task` or
:class:`~mink_warp.tasks.TargetedTask`. Each implements
:meth:`~mink_warp.Task.compute_residual`, returning weighted
Jacobian rows ``W``, error ``e``, and optional Levenberg–Marquardt damping
``mu`` — the same stacking convention as Mink [FrameTaskJacobian]_.

Solvers
-------

Solver backends share :class:`~mink_warp.Solver`:

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Backend
     - One step
     - Notes
   * - :class:`~mink_warp.DLSSolver` (``IKSolver`` alias)
     - :math:`v = \Delta q / dt` from damped normal equations
     - Default; optional CUDA graph
   * - :class:`~mink_warp.LMSolver`
     - LM step on configuration
     - Newton-style tiled Cholesky
   * - :class:`~mink_warp.LBFGSSolver`
     - Quasi-Newton step
     - Multi-iter ``solve_and_integrate``
   * - :class:`~mink_warp.ConstrainedSolver`
     - Box or general ``G Δq ≤ h`` via ADMM
     - Mink QP equivalent; box path exactly feasible each iter

Kernels
-------

Low-level Warp code lives under ``mink_warp/kernels/`` (frame Jacobians,
residual stacking, tile Cholesky, box / general-inequality ADMM). Public code
should call tasks and solvers, not kernels directly.

Related
-------

- :doc:`mink_parity`
- :doc:`../workflows/constrained`
- :doc:`../workflows/solvers`
- :doc:`../api/index`
