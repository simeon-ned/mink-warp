.. _tasks-and-limits:

Tasks and limits
================

Differential inverse kinematics computes a velocity :math:`v = \Delta q / \mathrm{d}t`
that reduces weighted task errors. Each task defines a residual
:math:`e(q) \in \mathbb{R}^k` driven toward zero and a Jacobian
:math:`J(q)` such that :math:`J \Delta q \approx -\alpha e` at first order
(see :doc:`../concepts/notation`).

mink-warp uses the same **tasks vs limits** split as Mink:

.. grid:: 2

   .. grid-item-card:: **Tasks** (soft objectives)

      Weighted least-squares terms stacked into :math:`W, e`. Can conflict;
      the solver minimises combined error.

      - :class:`~mink_warp.FrameTask` — frame pose
      - :class:`~mink_warp.RelativeFrameTask` — pose relative to another frame
      - :class:`~mink_warp.PostureTask` — nominal configuration
      - :class:`~mink_warp.ComTask` — center of mass
      - :class:`~mink_warp.EqualityConstraintTask` — MuJoCo equality rows (connect, weld, …)
      - :class:`~mink_warp.DampingTask` — velocity regularization
      - :class:`~mink_warp.JointLimitTask` — soft limit penalty

   .. grid-item-card:: **Limits** (hard constraints)

      Enforced by :class:`~mink_warp.ConstrainedSolver` via ADMM on the
      same :math:`H, c` as DLS. Mink's ``G Δq ≤ h`` form; two GPU paths (box /
      general inequality).

      - :class:`~mink_warp.ConfigurationLimit` — joint bounds
      - :class:`~mink_warp.VelocityLimit` — per-step velocity cap
      - :class:`~mink_warp.CollisionAvoidanceLimit` — geom-pair normal velocity
      - :class:`~mink_warp.LinearInequalityLimit` — constant half-spaces

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

Relative frame
--------------

Regulate a frame pose in another frame's coordinates (e.g. hand relative to torso):

.. code-block:: python

   rel = mw.RelativeFrameTask(
       "left_palm", "site",
       "torso_link", "body",
       position_cost=5.0, orientation_cost=0.5,
   )
   rel.set_target_from_configuration(cfg)

Equality constraints
--------------------

Regulate MuJoCo equality rows (closed chains). Uses host ``mj_forward`` per world:

.. code-block:: python

   eq = mw.EqualityConstraintTask(model, cost=500.0, gain=0.5)
   tasks = [frame, posture, eq]

Collision avoidance
-------------------

Configuration-dependent inequalities ``G Δq ≤ h`` (forces the general ADMM path):

.. code-block:: python

   limits = [
       mw.ConfigurationLimit(model),
       mw.CollisionAvoidanceLimit(
           model,
           geom_pairs=[(["wrist_3_link"], ["floor", "wall"])],
       ),
   ]
   v = mw.solve_ik(cfg, tasks, dt, limits=limits)

Soft vs hard joint limits
-------------------------

**Soft** (penalty in the task stack, unconstrained DLS):

.. code-block:: python

   soft = mw.JointLimitTask(model, cost=10.0)

**Hard** (never violate bounds — Mink ``limits=None`` default):

.. code-block:: python

   v = mw.solve_ik(cfg, tasks, dt, limits=None)  # default ConfigurationLimit

   # or explicit
   v = mw.solve_ik(cfg, tasks, dt, limits=[mw.ConfigurationLimit(model)])

Hard velocity cap
-----------------

.. code-block:: python

   limits = [
       mw.ConfigurationLimit(model),
       mw.VelocityLimit(model, 3.0),
   ]
   v = mw.solve_ik(cfg, tasks, dt, limits=limits)

General inequalities
--------------------

For half-spaces or coupled bounds that are not a per-dof box, use
:class:`~mink_warp.LinearInequalityLimit` or subclass
:class:`~mink_warp.Limit`.
See :doc:`../workflows/constrained` for the box vs general ADMM paths and tuning.

Typical stack
-------------

.. code-block:: python

   tasks = [frame, posture, mw.DampingTask(model, cost=1e-3)]
   limits = [mw.ConfigurationLimit(model)]
   solver = mw.ConstrainedSolver(cfg, limits=limits)
   solver.solve_and_integrate(tasks, dt=0.01)

See :doc:`../workflows/solvers`, :doc:`../workflows/constrained`, and
:doc:`../api/tasks` for full API details.

Examples
--------

Runnable mjviser demos (numbered by complexity):

.. list-table::
   :header-rows: 1
   :widths: 12 30 58

   * - #
     - Script
     - Uses
   * - 01
     - ``examples/01_panda_ik.py``
     - :class:`~mink_warp.FrameTask`, :class:`~mink_warp.JointLimitTask` (soft)
   * - 02
     - ``examples/02_constrained_ur5e.py``
     - :class:`~mink_warp.CollisionAvoidanceLimit`, :class:`~mink_warp.VelocityLimit` (hard)
   * - 03
     - ``examples/03_equality_cassie.py``
     - :class:`~mink_warp.EqualityConstraintTask`
   * - 04
     - ``examples/04_self_collision_dual_iiwa.py``
     - Self-collision :class:`~mink_warp.CollisionAvoidanceLimit`
   * - 05
     - ``examples/05_relative_frame_g1.py``
     - :class:`~mink_warp.RelativeFrameTask` + collision

See :doc:`../examples`.
