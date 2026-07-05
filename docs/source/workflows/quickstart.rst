.. _quickstart:

Quickstart: single-world IK
===========================

This page walks through a minimal differential IK loop on the Franka Panda —
the mink-warp equivalent of Mink's getting-started example.

Setup
-----

Load the model and create a batched configuration with ``nworld=1``:

.. code-block:: python

   import mujoco
   import mink_warp as mw

   model = mujoco.MjModel.from_xml_path("franka_emika_panda/mjx_scene.xml")
   cfg = mw.Configuration(model, nworld=1)
   cfg.update_from_keyframe("home")

:class:`~mink_warp.Configuration` wraps MuJoCo Warp state and exposes FK,
Jacobians, and integration on device.

Target pose
-----------

Use host :class:`~mink_warp.SE3` / :class:`~mink_warp.SO3` for targets (same
composition rules as Mink):

.. code-block:: python

   import numpy as np
   from mink_warp import SE3, SO3

   ee = cfg.get_transform_frame_to_world_se3("attachment_site", "site")
   target = (
       SE3.from_translation(np.array([0.0, -0.4, -0.2])) @ ee
       @ SE3.from_rotation(SO3.exp(np.array([0.0, -np.pi / 2, 0.0])))
   )

Tasks
-----

.. code-block:: python

   frame = mw.FrameTask(
       "attachment_site", "site",
       position_cost=1.0, orientation_cost=1.0, gain=0.5,
   )
   frame.set_target(target)
   posture = mw.PostureTask(model, cost=1e-2)
   posture.set_target_from_configuration(cfg)

IK loop
-------

Each step: solve for velocity, then integrate:

.. code-block:: python

   dt = 1.0 / 60.0
   for _ in range(120):
       v = mw.solve_ik(cfg, [frame, posture], dt)
       cfg.integrate_inplace(v, dt)

Or use a reusable solver with integration built in:

.. code-block:: python

   solver = mw.DLSSolver(cfg)
   solver.solve_and_integrate([frame, posture], dt, iterations=120)

Complete example
----------------

.. literalinclude:: ../../../examples/docs/quickstart.py
   :language: python

Next steps
----------

- :doc:`batched_ik` — scale to hundreds of parallel worlds
- :doc:`../tutorial/tasks_and_limits` — posture, CoM, soft vs hard limits
- :doc:`solvers` — LM / L-BFGS / constrained backends
