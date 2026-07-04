.. _benchmarks:

Benchmarks
==========

Throughput and accuracy scripts live in ``benchmarks/``.

Run locally
-----------

.. code-block:: bash

   uv sync --extra dev
   uv run python benchmarks/bench_ik.py
   uv run python benchmarks/bench_parity.py
   uv run python benchmarks/bench_solvers.py

What they measure
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Script
     - Metric
   * - ``bench_ik.py``
     - Solves/sec vs batch size (Panda, G1)
   * - ``bench_parity.py``
     - Agreement with CPU Mink (oracle)
   * - ``bench_solvers.py``
     - DLS / LM / L-BFGS / constrained relative cost
   * - ``bench_constrained.py``
     - Box-ADMM constrained solve throughput

Recorded numbers are in ``benchmarks/RESULTS.md``.

Interpretation
--------------

- Tile Cholesky helps most at large ``nv`` (e.g. G1); FK + Jacobian assembly
  often dominate end-to-end IK time at moderate ``nv``.
- CUDA graphs reduce launch overhead when the task set is fixed.

Related
-------

- :doc:`workflows/batched_ik`
- :doc:`workflows/cuda_graphs`
