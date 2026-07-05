Installation
============

``mink-warp`` requires Python 3.10–3.13, MuJoCo, MuJoCo Warp, and Warp.

- **GPU recommended** for batched throughput and CUDA graphs.
- **CPU supported** for development and tests (including constrained ADMM via Warp LLVM).

From source (development)
-------------------------

.. tab-set::

   .. tab-item:: uv

      .. code-block:: bash

         git clone https://github.com/simeon-ned/mink-warp.git && cd mink-warp
         uv sync --extra dev --extra examples

   .. tab-item:: pip

      .. code-block:: bash

         git clone https://github.com/simeon-ned/mink-warp.git && cd mink-warp
         pip install -e ".[dev,examples]"

Verification
------------

.. code-block:: bash

   uv run python -c "import mink_warp as mw; print(mw.__version__)"

Optional extras
---------------

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Extra
     - Purpose
   * - ``dev``
     - pytest, ruff, mink (parity tests)
   * - ``examples``
     - mjviser + viser for Panda / G1 demos
   * - ``docs``
     - Sphinx site (``make docs``)

Build the documentation locally: see ``docs/BUILDING.md``.

Related projects
----------------

- `Mink <https://github.com/kevinzakka/mink>`_ — single-world CPU differential IK
  (QP-based); API reference for tasks and limits.
- `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_ — batched MuJoCo
  simulation on GPU [MuJoCoWarp]_.
- `Newton <https://github.com/google-deepmind/newton>`_ — GPU physics + batch pose
  IK [Newton]_; mink-warp reuses solver *patterns* but not the Newton API (see
  :doc:`source/concepts/why_mink_warp`).
