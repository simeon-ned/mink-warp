.. _cuda-graphs:

CUDA graphs
===========

For fixed task sets and timesteps, :class:`~mink_warp.solvers.DLSSolver` can
capture a one-step ``solve_and_integrate`` graph to reduce launch overhead.

Requirements
------------

- CUDA device
- Fixed task list (same objects each frame)
- Fixed ``dt`` and damping between captures
- No host→device copies inside the captured region

Usage
-----

.. code-block:: python

   solver = mw.DLSSolver(cfg)
   cfg.set_integration_dt(0.01)  # written before capture

   # Warm-up (allocates task buffers, compiles kernels)
   solver.solve_and_integrate(tasks, dt=0.01, use_graph=True)

   while running:
       solver.solve_and_integrate(tasks, dt=0.01, use_graph=True)

If tasks or ``dt`` change, call :meth:`~mink_warp.solvers.DLSSolver.invalidate_graph`
(or recreate the solver).

Implementation notes
--------------------

- Integration uses **out-of-place** ``qpos`` writes; in-place aliasing breaks graphs.
- ``dt`` is stored in a device buffer via ``set_integration_dt`` before capture.
- LM / L-BFGS / constrained backends do not expose graph capture yet.

Related
-------

- :doc:`batched_ik`
- :doc:`../concepts/architecture`
