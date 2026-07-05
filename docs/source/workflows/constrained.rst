.. _constrained-ik:

Constrained IK (hard limits)
============================

Mink solves a **QP** each control step: minimise task error subject to hard
inequalities ``G Δq ≤ h``. mink-warp mirrors that via
:class:`~mink_warp.solvers.ConstrainedSolver` — a batched OSQP-style ADMM inner
solve on the same normal equations as :class:`~mink_warp.solvers.DLSSolver`.

Problem
-------

Per world, after stacking tasks into :math:`H, c`:

.. math::

   \min_{\Delta q}\; \tfrac{1}{2}\,\Delta q^\top H \Delta q + c^\top \Delta q
   \quad\text{s.t.}\quad G\,\Delta q \le h

Velocity is :math:`v = \Delta q / dt` as usual.

Two solve paths (auto-selected)
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 38 40

   * - Path
     - When
     - Properties
   * - **Box ADMM**
     - Every limit is a per-dof interval (:class:`~mink_warp.limits.ConfigurationLimit`,
       :class:`~mink_warp.limits.VelocityLimit`) and ``use_inequalities=False``
     - Fast; **exact box feasibility at every ADMM iteration** (even if truncated)
   * - **General inequality ADMM**
     - Any inequality-only limit (e.g. :class:`~mink_warp.limits.LinearInequalityLimit`),
       or ``use_inequalities=True`` (re-expresses box limits as ``[P;-P]`` rows)
     - Reduced Schur-normal OSQP-ADMM; feasibility improves with ``admm_iters``

Both paths factor a SPD matrix once per solve (tile Cholesky) and run a fixed
number of ADMM iterations. They work on **CUDA and CPU** (Warp LLVM backend).

Built-in limits
---------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Limit
     - Form
   * - :class:`~mink_warp.limits.ConfigurationLimit`
     - ``gain*(lower - q) ≤ Δq ≤ gain*(upper - q)`` per hinge/slide joint (Mink parity)
   * - :class:`~mink_warp.limits.VelocityLimit`
     - ``|Δq_i| ≤ dt * v_max`` on bounded dofs
   * - :class:`~mink_warp.limits.LinearInequalityLimit`
     - Constant ``G Δq ≤ h`` (half-spaces, coupled bounds); **inequality-only**

Each limit exposes either a **box** (:meth:`~mink_warp.limits.Limit.apply_box`) or
**dense rows** (:meth:`~mink_warp.limits.Limit.scatter_inequalities`), matching
Mink's stacked ``G Δq ≤ h``. See :doc:`../api/limits`.

Functional API (Mink-shaped)
----------------------------

Omit ``limits`` for unconstrained DLS (historical default):

.. code-block:: python

   v = mw.solve_ik(cfg, tasks, dt)  # no hard limits

Pass ``limits`` to auto-build a :class:`~mink_warp.ConstrainedSolver`:

.. code-block:: python

   # Mink's limits=None → default ConfigurationLimit
   v = mw.solve_ik(cfg, tasks, dt, limits=None)

   # Explicit stack
   v = mw.solve_ik(
       cfg, tasks, dt,
       limits=[mw.ConfigurationLimit(model), mw.VelocityLimit(model, 3.0)],
   )

   # Unconstrained despite passing limits= (empty list)
   v = mw.solve_ik(cfg, tasks, dt, limits=[])

``limits=`` is only honoured when ``solver=None``. Passing both ``solver=`` and
``limits=`` raises — configure limits on the solver you construct, or drop
``solver=``.

Explicit solver
---------------

.. code-block:: python

   solver = mw.ConstrainedSolver(
       cfg,
       limits=[mw.ConfigurationLimit(model, gain=0.95)],
       admm_iters=30,
       damping=1e-12,
   )
   v = solver.solve(tasks, dt=0.01)

Tuning knobs: ``admm_iters``, ``rho_scale`` / ``rho_min`` / ``rho_max``, ``alpha``
(over-relaxation), ``sigma`` (general path SPD floor), ``use_inequalities``.

General inequalities
----------------------

Use :class:`~mink_warp.LinearInequalityLimit` for constant half-spaces Mink's
per-dof box cannot express:

.. code-block:: python

   import numpy as np

   # e.g. n @ dq <= h0  (one row)
   n = np.zeros(model.nv, dtype=np.float32)
   n[2] = 1.0
   lim = mw.LinearInequalityLimit(n[None, :], [0.05])

   solver = mw.ConstrainedSolver(cfg, limits=[lim], admm_iters=200)
   v = solver.solve(tasks, dt=0.01)

Presence of an inequality-only limit **auto-selects** the general path. For
parity/debugging you can force box limits through the dense path:

.. code-block:: python

   mw.ConstrainedSolver(
       cfg,
       limits=[mw.ConfigurationLimit(model)],
       use_inequalities=True,
       admm_iters=80,
   )

Custom limits
-------------

For **configuration-dependent** rows (collision normals that move with ``q``),
subclass :class:`~mink_warp.limits.Limit` and implement
:meth:`~mink_warp.limits.Limit.scatter_inequalities` (set ``box_capable = False``
if there is no per-dof box form).

Box vs general — when to care
-------------------------------

- **Control loops with joint + velocity caps:** default box path; raise
  ``admm_iters`` only if you need closer agreement with the true QP optimum.
- **Half-spaces / coupled dof bounds:** ``LinearInequalityLimit`` or a custom
  ``Limit``; budget more ``admm_iters`` — feasibility is asymptotic.
- **Mink parity tests:** ``ConfigurationLimit`` box path matches mink ``daqp``;
  ``use_inequalities=True`` re-expresses the same rows as dense ``G Δq ≤ h``.

Validation
----------

- ``tests/test_constrained_solver.py`` — box path vs mink
- ``tests/test_inequality_constraints.py`` — general path, mixed limits
- ``tests/test_admm_reference.py`` — NumPy ADMM oracle
- ``benchmarks/bench_constrained.py`` — throughput + violation metrics
- ``benchmarks/bench_osqp.py`` — inner ADMM vs OSQP on standard QPs

Related
-------

- :doc:`solvers` — all backends
- :doc:`../tutorial/tasks_and_limits` — soft vs hard limits
- :doc:`../api/limits`
- :doc:`../benchmarks`
