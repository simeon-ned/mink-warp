.. _examples:

Examples
========

Runnable scripts ship under ``examples/`` (numbered by increasing complexity).
Assets are vendored from Mink.

Visualization (mjviser)
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - Description
   * - ``examples/01_panda_ik.py``
     - Panda EE tracking, :class:`~mink_warp.JointLimitTask` (soft), CUDA graph
   * - ``examples/02_constrained_ur5e.py``
     - Hard limits: :class:`~mink_warp.ConfigurationLimit`,
       :class:`~mink_warp.CollisionAvoidanceLimit`,
       :class:`~mink_warp.VelocityLimit`
   * - ``examples/03_equality_cassie.py``
     - Closed-chain :class:`~mink_warp.EqualityConstraintTask`; feet pinned, COM bob
   * - ``examples/04_self_collision_dual_iiwa.py``
     - Dual Kuka arms; inter-arm :class:`~mink_warp.CollisionAvoidanceLimit`
   * - ``examples/05_relative_frame_g1.py``
     - G1 humanoid: :class:`~mink_warp.RelativeFrameTask`, squat, hand motion, collision

.. code-block:: bash

   uv sync --extra examples
   uv run examples/01_panda_ik.py
   uv run examples/05_relative_frame_g1.py

Walkthrough by complexity
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 12 28 60

   * - #
     - Script
     - Highlights
   * - 01
     - ``01_panda_ik.py``
     - :class:`~mink_warp.FrameTask`, soft limits, ``IKSolver``, mjviser grid
   * - 02
     - ``02_constrained_ur5e.py``
     - :class:`~mink_warp.ConstrainedSolver` + collision — see :doc:`workflows/constrained`
   * - 03
     - ``03_equality_cassie.py``
     - :class:`~mink_warp.EqualityConstraintTask` on Cassie closed chain
   * - 04
     - ``04_self_collision_dual_iiwa.py``
     - Dual Kuka iiwa14; inter-arm collision (Mink ``arm_dual_iiwa``)
   * - 05
     - ``05_relative_frame_g1.py``
     - G1: :class:`~mink_warp.RelativeFrameTask` + collision — see :doc:`tutorial/tasks_and_limits`

Documentation examples
----------------------

Scripts in ``examples/docs/`` are tested by ``tests/test_docs.py`` and included
in the Sphinx tutorials via ``literalinclude``.

Benchmarks
----------

See :doc:`benchmarks` and ``benchmarks/README.md``.
