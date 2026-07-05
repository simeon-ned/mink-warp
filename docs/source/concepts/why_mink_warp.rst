.. _why-mink-warp:

Why mink-warp (and not Newton IK)?
==================================

mink-warp and `Newton IK <https://github.com/google-deepmind/newton>`_ [Newton]_
both run batched inverse kinematics on the GPU, but they solve **different
problems** for **different callers**. This page states what mink-warp is for and
when Newton is the better choice.

What mink-warp is for
---------------------

mink-warp is a **batched differential IK library** on MuJoCo Warp with a
**Mink-shaped API** [Mink]_.

Typical use cases:

- **Real-time control loops** — one small IK step per tick, output is joint
  *velocity* :math:`v = \Delta q / dt`, then integrate (legged control, teleop,
  retargeting pipelines).
- **Many parallel worlds** — ``nworld`` copies of the same robot + task stack
  (multi-agent sim, parameter grids, batched targets).
- **mjwarp / mjlab stacks** — stay on ``mujoco.MjModel`` + ``mujoco_warp``;
  no separate physics engine to adopt.
- **Porting Mink** — same task names, body-frame Jacobians, ``solve_ik(cfg,
  tasks, dt)``; add batching and move the hot path to device.

It is **not** a full simulator. It does FK, Jacobians, task residuals, linear /
constrained (QP inequality) solves, and integration — nothing else.

What Newton IK is for
---------------------

Newton IK is part of the **Newton physics platform**. Its ``newton.ik.IKSolver``
is a **batch pose-to-configuration optimizer**:

- Objectives are link **positions** and **orientations** (plus optional joint-limit
  penalties), not Mink-style composable tasks.
- Default workflow: run **many LM or L-BFGS iterations** (optionally **multi-seed**
  sampling) until residuals are small — closer to cuRobo / PyRoki-style IK.
- Built on Newton's own ``Model`` and articulation stack, with analytic /
  autodiff / mixed Jacobians for Newton objectives.

Newton is the right tool when you want **global-ish IK from scratch** inside
Newton sim, or you are already committed to the Newton ecosystem end-to-end.

Side-by-side
------------

.. list-table::
   :header-rows: 1
   :widths: 22 39 39

   * -
     - **mink-warp**
     - **Newton IK**
   * - Primary output
     - Joint **velocity** per control step (differential IK)
     - Joint **configuration** after optimization
   * - Default step
     - One damped least-squares step (:math:`\Delta q`, then :math:`v=\Delta q/dt`)
     - Many LM / L-BFGS iterations toward target poses
   * - API shape
     - Mink: ``FrameTask``, ``PostureTask``, ``solve_ik``
     - Newton: ``IKObjectivePosition``, ``IKObjectiveRotation``, ``IKSolver.step``
   * - Model
     - ``mujoco.MjModel`` + MuJoCo Warp
     - Newton ``Model`` / articulation
   * - Multi-seed sampling
     - Not built-in (you batch worlds yourself)
     - First-class (``IKSampler``, parallel seeds)
   * - Scope
     - IK only (lightweight)
     - Full physics + diff sim + IK module
   * - Best fit
     - GPU sim loops, Mink ports, mjlab-adjacent stacks
     - Newton-native sim, batch pose IK benchmarks

Why we did not wrap Newton IK
-----------------------------

1. **Different contract.** Callers of Mink (and most differential-IK control stacks) expect
   *differential* IK: velocity out, integrate every frame. Newton IK's natural
   contract is *optimize* ``q`` until task-space error is small. Bridging the two
   would either hide Newton's strengths or break Mink parity.

2. **Different model path.** mink-warp is intentionally **MjModel → mjwarp** so
   it drops into the same assets and pipelines as Mink and MuJoCo Warp.
   Newton IK wants a Newton ``Model``; that is a fork, not a swap-in.

3. **Task composability.** Mink's task stack (frame + posture + CoM + soft limits,
   weighted residuals, body-frame Jacobians) is the API we mirror. Newton's
   objective list is excellent for pose IK but not a line-for-line substitute for
   Mink controllers.

4. **Dependency weight.** mink-warp depends on MuJoCo + mujoco_warp + warp. Pulling
   in all of Newton for IK alone is heavier than needed for "batched Mink on GPU."

5. **Implementation reuse, not API reuse.** We borrow **patterns** from Newton
   (tile Cholesky via ``wp.launch_tiled``, batched normal equations) where they
   fit mink-warp's solvers — without making Newton a runtime dependency.

When to use which
-----------------

**Use mink-warp if:**

- You run a **fixed-rate control loop** and want one cheap IK step per tick.
- You already have **Mink** or **mjwarp** code and want GPU batching.
- You need **hard limits in a Mink-like** form (``ConfigurationLimit`` +
  ``ConstrainedSolver``) in the same loop as soft tasks.

**Use Newton IK if:**

- You need **batch pose IK** with multi-seed search and high success rates on
  hard reach problems inside Newton.
- Your stack is already **Newton-native** (sim, diff, assets on Newton ``Model``).
- You want Newton's **analytic Jacobian** paths for built-in position/rotation
  objectives at very large batch sizes.

The two can coexist in a lab: Newton for offline batch reach / dataset generation;
mink-warp for online differential control in MuJoCo Warp sim or deploy-adjacent
prototyping.

Related
-------

- :doc:`architecture`
- :doc:`mink_parity`
- :doc:`../workflows/solvers`
- `Newton IK tutorial <https://github.com/google-deepmind/newton/blob/main/docs/tutorials/01_robotics.ipynb>`_
