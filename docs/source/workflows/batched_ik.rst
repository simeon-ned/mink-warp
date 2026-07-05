.. _batched-ik:

Batched IK
==========

The main reason to use mink-warp over Mink is **parallel IK for many worlds**
with one model and one solver launch grid.

Configuration
-------------

.. code-block:: python

   cfg = mw.Configuration(model, nworld=512, device="cuda")
   q0 = np.tile(model.qpos0, (512, 1)).astype(np.float32)
   # Optional: per-world perturbation
   for i in range(512):
       q0[i, 0] += 0.03 * np.sin(i * 2 * np.pi / 512)
   cfg.update(q=q0)

All subsequent FK, Jacobians, and solves use shape ``(nworld, …)``.

Per-world targets
-----------------

Upload a batch of targets once (or only when they change):

.. code-block:: python

   import warp as wp

   targets = wp.array(poses_np, dtype=float)  # (nworld, 7) wxyz_xyz
   frame.set_target(targets)

For a single shared target, pass one ``SE3`` or a length-1 array; tasks broadcast
as needed.

Solver loop
-----------

.. code-block:: python

   solver = mw.DLSSolver(cfg)
   while running:
       solver.solve_and_integrate(tasks, dt=0.01, use_graph=True)
       q_host = cfg.q.numpy()  # boundary copy for visualization

Demos
-----

Full batched examples with mjviser:

.. code-block:: bash

   uv sync --extra examples
   uv run examples/01_panda_ik.py
   uv run examples/05_relative_frame_g1.py

Assets live under ``examples/franka_emika_panda/`` and ``examples/unitree_g1/``.

Performance notes
-----------------

- Prefer keeping ``q``, targets, and task buffers on device across steps.
- Re-upload targets only when they change (see ``05_relative_frame_g1.py``).
- Cholesky tile solve cost is ~flat in ``nworld``; FK + Jacobian assembly often
  dominate at moderate ``nv`` (see :doc:`../benchmarks`).

Related
-------

- :doc:`cuda_graphs`
- :doc:`../examples`
