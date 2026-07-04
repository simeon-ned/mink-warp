.. _examples:

Examples
========

Runnable scripts ship under ``examples/``. Assets are vendored from Mink.

Batched visualization (mjviser)
-------------------------------

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Script
     - Description
   * - ``examples/batched_panda_ik.py``
     - 512 Panda arms, circular EE targets, CUDA graph
   * - ``examples/batched_g1_ik.py``
     - Multi-G1 hands/feet/torso targets, periodic resample

.. code-block:: bash

   uv sync --extra examples
   uv run examples/batched_panda_ik.py
   uv run examples/batched_g1_ik.py

Documentation examples
----------------------

Scripts in ``examples/docs/`` are tested by ``tests/test_docs.py`` and included
in the Sphinx tutorials via ``literalinclude``.

Benchmarks
----------

See :doc:`benchmarks` and ``benchmarks/README.md``.
