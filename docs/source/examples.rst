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
     - Panda EE tracking, soft joint limits, CUDA graph
   * - ``examples/02_constrained_ur5e.py``
     - Hard limits: configuration, collision avoidance, velocity cap
   * - ``examples/03_equality_cassie.py``
     - Closed-chain equality constraints, feet pinned, COM bob
   * - ``examples/04_self_collision_dual_iiwa.py``
     - Dual Kuka arms, inter-arm collision avoidance
   * - ``examples/05_relative_frame_g1.py``
     - G1 humanoid: RelativeFrameTask, squat, hand motion, collision

.. code-block:: bash

   uv sync --extra examples
   uv run examples/01_panda_ik.py
   uv run examples/05_relative_frame_g1.py

Documentation examples
----------------------

Scripts in ``examples/docs/`` are tested by ``tests/test_docs.py`` and included
in the Sphinx tutorials via ``literalinclude``.

Benchmarks
----------

See :doc:`benchmarks` and ``benchmarks/README.md``.
