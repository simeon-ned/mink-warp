MINK-WARP
=========

**mink-warp** is batched differential inverse kinematics on
`MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_, with a
`Mink <https://github.com/kevinzakka/mink>`_-shaped API.

Given a robot's current configuration and a stack of task-space objectives,
mink-warp computes joint velocities (or configuration updates) that reduce
weighted task error — for **many worlds in parallel** on the GPU.

Purpose
-------

mink-warp exists for **real-time, GPU-batched differential IK** in MuJoCo Warp
pipelines — not as a replacement for `Newton IK <https://github.com/google-deepmind/newton>`_.
It targets the same niche as **Mink** (composable tasks, velocity output, control
loops), scaled to ``nworld`` on device. See :doc:`source/concepts/why_mink_warp`
for a full comparison with Newton IK and when to use each.

Key features
------------

- **Mink-shaped API** — ``Configuration``, ``FrameTask``, ``PostureTask``,
  ``solve_ik``, body-frame Jacobians, and host ``SE3`` / ``SO3`` helpers.
- **Device-native hot path** — FK, Jacobians, residual assembly, and linear
  solves run as Warp kernels on ``wp.array`` buffers; NumPy only at boundaries.
- **Batched by design** — ``nworld`` parallel instances share one model and one
  launch grid (Panda grids, multi-agent IK, parameter sweeps).
- **Interchangeable solvers** — damped least squares (default), Levenberg–Marquardt,
  L-BFGS, and **constrained ADMM** for hard limits (box + general ``G Δq ≤ h``,
  Mink QP equivalent).
- **CUDA graph capture** — optional one-step ``solve_and_integrate`` graph for
  fixed task sets in control loops.

Minimal example
---------------

.. code-block:: python

   import mujoco
   import mink_warp as mw

   model = mujoco.MjModel.from_xml_path("robot.xml")
   cfg = mw.Configuration(model, nworld=512, device="cuda")

   frame = mw.FrameTask("ee", "site", position_cost=1.0, orientation_cost=1.0)
   frame.set_target_from_configuration(cfg)
   posture = mw.PostureTask(model, cost=1e-2)
   posture.set_target_from_configuration(cfg)

   solver = mw.DLSSolver(cfg)
   solver.solve_and_integrate([frame, posture], dt=0.01, use_graph=True)

Table of Contents
-----------------

.. toctree::
   :maxdepth: 1
   :caption: Getting Started

   installation
   source/workflows/quickstart
   source/workflows/batched_ik
   source/examples

.. toctree::
   :maxdepth: 1
   :caption: Concepts

   source/concepts/index

.. toctree::
   :maxdepth: 1
   :caption: User Guide

   source/tutorial/index
   source/workflows/solvers
   source/workflows/constrained
   source/workflows/cuda_graphs
   source/benchmarks

.. toctree::
   :maxdepth: 1
   :caption: API Reference

   source/api/index

.. toctree::
   :maxdepth: 1
   :caption: Further Reading

   references
   source/roadmap

License
-------

mink-warp is licensed under Apache-2.0. See the
`LICENSE file <https://github.com/simeon-ned/mink-warp/blob/main/LICENSE>`_.
