Solvers
=======

Base
----

.. autoclass:: mink_warp.Solver
   :members:

Backends
--------

.. autoclass:: mink_warp.DLSSolver
   :members:

.. autoclass:: mink_warp.LMSolver
   :members:

.. autoclass:: mink_warp.LBFGSSolver
   :members:

.. autoclass:: mink_warp.ConstrainedSolver
   :members:

Factory
-------

.. autofunction:: mink_warp.make_solver

Registry
--------

``mink_warp.solvers.SOLVERS`` maps solver name strings to backend classes
(``"dls"``, ``"lm"``, ``"lbfgs"``, ``"constrained"``).
