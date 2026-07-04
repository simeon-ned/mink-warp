.. _solvers:

Solver backends
===============

All backends implement :class:`~mink_warp.solvers.base.Solver` and minimise the
same weighted least-squares task cost. Pick based on step semantics and limits.

Quick reference
---------------

.. list-table::
   :header-rows: 1
   :widths: 18 22 60

   * - Name
     - ``make_solver``
     - Use when
   * - DLS
     - ``"dls"``
     - Real-time velocity IK (Mink-style differential step); default
   * - LM
     - ``"lm"``
     - Few Newton-style steps on ``q`` per control tick
   * - L-BFGS
     - ``"lbfgs"``
     - Multi-iteration reach with quasi-Newton
   * - Constrained
     - ``"constrained"``
     - Hard joint / velocity limits (box-QP + ADMM, CUDA)

Damped least squares (default)
------------------------------

.. code-block:: python

   solver = mw.make_solver(cfg, "dls", damping=1e-12)
   v = solver.solve(tasks, dt=0.01)
   cfg.integrate_inplace(v, dt)

:class:`~mink_warp.solvers.DLSSolver` is also exposed as ``mw.IKSolver``.

Levenberg–Marquardt
-------------------

.. code-block:: python

   solver = mw.LMSolver(cfg)
   solver.solve_and_integrate(tasks, dt=0.01, iterations=5)

Advances ``q`` on device; returns equivalent tangent velocity.

L-BFGS
------

.. code-block:: python

   solver = mw.LBFGSSolver(cfg, history=10)
   solver.solve_and_integrate(tasks, dt=0.01, iterations=20)

Hard limits (constrained)
-------------------------

.. code-block:: python

   from mink_warp.limits import ConfigurationLimit, VelocityLimit

   limits = [ConfigurationLimit(model), VelocityLimit(model, 3.0)]
   solver = mw.ConstrainedSolver(cfg, limits=limits)
   v = solver.solve(tasks, dt=0.01)

Or via the functional API:

.. code-block:: python

   v = mw.solve_ik(cfg, tasks, dt=0.01, limits=limits)

``limits=None`` uses a default :class:`~mink_warp.limits.ConfigurationLimit`;
``limits=[]`` disables hard limits (unconstrained DLS).

Functional shortcuts
--------------------

.. code-block:: python

   v = mw.solve_ik(cfg, tasks, dt)                    # one DLS step
   q = mw.solve_ik_iterations(cfg, tasks, dt, iterations=10)

See :doc:`../api/solvers` for the full API.

Related
-------

- :doc:`cuda_graphs` — graph capture with ``DLSSolver``
- :doc:`../tutorial/tasks_and_limits`
