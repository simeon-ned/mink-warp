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
   uv run python benchmarks/bench_constrained.py
   uv run --with osqp python benchmarks/bench_osqp.py

What they measure
-----------------

.. list-table::
   :header-rows: 1
   :widths: 30 70

   * - Script
     - Metric
   * - ``bench_ik.py``
     - Solves/sec vs batch size (Panda, G1); ``--solver dls/lm/lbfgs``
   * - ``bench_parity.py``
     - Agreement with CPU Mink (unconstrained DLS oracle)
   * - ``bench_solvers.py``
     - DLS / LM / L-BFGS relative cost and tracking error
   * - ``bench_constrained.py``
     - Constrained solver throughput, joint-limit violation vs DLS; box vs
       ``constrained-ineq`` path; accuracy vs mink ``daqp`` + ``ConfigurationLimit``
   * - ``bench_osqp.py``
     - Inner box / general ADMM vs reference OSQP on standard QP examples

Recorded numbers are in ``benchmarks/RESULTS.md``.

Constrained solver notes
------------------------

- **Box path** (default for ``ConfigurationLimit`` + ``VelocityLimit``): exact
  feasibility each ADMM step; tune ``admm_iters`` for optimality, not safety.
- **General inequality path** (``LinearInequalityLimit``, or
  ``use_inequalities=True``): needs enough ``admm_iters`` for tight feasibility.
- Parity vs Mink uses ``limits=None`` / ``ConfigurationLimit`` on Panda (hinge/slide only).

Related
-------

- :doc:`workflows/constrained`
- :doc:`workflows/batched_ik`
- :doc:`workflows/cuda_graphs`
