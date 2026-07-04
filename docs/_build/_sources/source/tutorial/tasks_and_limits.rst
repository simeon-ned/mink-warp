.. _tasks-and-limits:

Tasks and limits
================

mink-warp uses the same **tasks vs limits** split as Mink:

.. grid:: 2

   .. grid-item-card:: **Tasks** (soft objectives)

      Weighted least-squares terms stacked into :math:`W, e`. Can conflict;
      the solver minimises combined error.

      - :class:`~mink_warp.FrameTask` — frame pose
      - :class:`~mink_warp.PostureTask` — nominal configuration
      - :class:`~mink_warp.ComTask` — center of mass
      - :class:`~mink_warp.DampingTask` — velocity regularization
      - :class:`~mink_warp.JointLimitTask` — soft limit penalty

   .. grid-item-card:: **Limits** (hard constraints)

      Enforced by :class:`~mink_warp.solvers.ConstrainedSolver` (box-QP).
      The returned step never violates the intersection of active limits.

      - :class:`~mink_warp.limits.ConfigurationLimit` — joint bounds
      - :class:`~mink_warp.limits.VelocityLimit` — per-step velocity cap

Frame task
----------

.. code-block:: python

   task = mw.FrameTask(
       "attachment_site", "site",
       position_cost=1.0,
       orientation_cost=1.0,
       gain=0.8,
       lm_damping=1.0,
   )
   task.set_target_from_configuration(cfg)

Posture and damping
-------------------

Regularize redundant dofs and avoid drift:

.. code-block:: python

   posture = mw.PostureTask(model, cost=1e-2)
   posture.set_target_from_configuration(cfg)
   damping = mw.DampingTask(model, cost=1e-3)

Center of mass
--------------

.. code-block:: python

   com = mw.ComTask(cost=np.array([1.0, 1.0, 0.1]))
   com.set_target_from_configuration(cfg)

Soft vs hard joint limits
-------------------------

**Soft** (always-on penalty, unconstrained DLS):

.. code-block:: python

   soft = mw.JointLimitTask(model, cost=10.0)

**Hard** (never violate bounds):

.. code-block:: python

   v = mw.solve_ik(cfg, tasks, dt, limits=[mw.ConfigurationLimit(model)])

Typical stack
-------------

.. code-block:: python

   tasks = [frame, posture, mw.DampingTask(model, cost=1e-3)]
   limits = [mw.ConfigurationLimit(model)]
   solver = mw.ConstrainedSolver(cfg, limits=limits)
   solver.solve_and_integrate(tasks, dt=0.01)

See :doc:`../workflows/solvers` for backend details and :doc:`../api/tasks` for
the full task API.
