.. _solvers:

Solver backends
===============

All backends implement :class:`~mink_warp.Solver` and minimise the
same weighted least-squares task cost. Pick based on step semantics and whether
you need **hard limits**.

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
     - Hard limits via box / general-inequality ADMM (Mink QP equivalent)

Only :class:`~mink_warp.ConstrainedSolver` has ``supports_limits = True``.
Cost-only backends (DLS / LM / L-BFGS) ignore ``limits=`` if passed explicitly.

Damped least squares (default)
------------------------------

.. code-block:: python

   solver = mw.make_solver(cfg, "dls", damping=1e-12)
   v = solver.solve(tasks, dt=0.01)
   cfg.integrate_inplace(v, dt)

:class:`~mink_warp.DLSSolver` is also exposed as ``mink_warp.IKSolver``.

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

Mink-compatible shortcut — auto-builds :class:`~mink_warp.ConstrainedSolver`:

.. code-block:: python

   from mink_warp.limits import ConfigurationLimit, VelocityLimit

   v = mw.solve_ik(cfg, tasks, dt=0.01, limits=None)  # default joint limit
   v = mw.solve_ik(
       cfg, tasks, dt=0.01,
       limits=[ConfigurationLimit(model), VelocityLimit(model, 3.0)],
   )

Explicit solver (tuning ``admm_iters``, ``use_inequalities``, etc.):

.. code-block:: python

   solver = mw.ConstrainedSolver(cfg, limits=[ConfigurationLimit(model)])
   v = solver.solve(tasks, dt=0.01)

``limits=[]`` disables hard limits (regularized DLS inside the constrained backend).

Full guide: :doc:`constrained`.

Functional shortcuts
--------------------

.. code-block:: python

   v = mw.solve_ik(cfg, tasks, dt)                    # unconstrained DLS
   q = mw.solve_ik_iterations(cfg, tasks, dt, iterations=10)

See :doc:`../api/solvers` for the full API.

Related
-------

- :doc:`constrained` — box vs general inequality ADMM, ``LinearInequalityLimit``
- :doc:`cuda_graphs` — graph capture with ``DLSSolver``
- :doc:`../tutorial/tasks_and_limits`
