.. _concepts:

Concepts
========

These pages explain **how mink-warp is put together** — the mental model before
you dive into solver backends or API details.

Suggested reading order
-----------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Topic
     - Page
   * - **Why this library**
     - :doc:`why_mink_warp` — purpose vs Newton IK and vs Mink
   * - **Hard limits / QP**
     - :doc:`../workflows/constrained` — box vs general ``G Δq ≤ h`` ADMM
   * - **Layer stack**
     - :doc:`architecture` — Configuration → tasks → solver → integrate
   * - **Mink compatibility**
     - :doc:`mink_parity` — what matches Mink and what differs
   * - **Hands-on**
     - :doc:`../workflows/quickstart` — single-world IK loop
   * - **Scaling out**
     - :doc:`../workflows/batched_ik` — ``nworld`` patterns

Where to go next
----------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Need
     - Section
   * - Hard joint / velocity / half-space limits
     - :doc:`../workflows/constrained`
   * - Solver choice (DLS / LM / L-BFGS)
     - :doc:`../workflows/solvers`
   * - Fixed-step control loops
     - :doc:`../workflows/cuda_graphs`
   * - Field-level API
     - :doc:`../api/index`
   * - Throughput numbers
     - :doc:`../benchmarks`

.. toctree::
   :maxdepth: 1
   :hidden:

   why_mink_warp
   architecture
   mink_parity
