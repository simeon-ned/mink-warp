Roadmap
=======

Shipped in current tree
-----------------------

- ``RelativeFrameTask``, ``EqualityConstraintTask``, ``ComTask``
- Hard limits: ``ConfigurationLimit``, ``VelocityLimit``,
  ``CollisionAvoidanceLimit``, ``LinearInequalityLimit`` + ``ConstrainedSolver``
- Numbered mjviser demos ``examples/01_…`` – ``05_…``
- Parity tests vs Mink for frame, relative frame, equality, collision, constrained solve

Planned directions (not committed timelines)
--------------------------------------------

- Optional ``mink_warp.viz`` extra — task target overlays (viser / mjviser adapter)
- Additional Mink tasks (look-at, dof freezing, …)
- Batched equality / collision distance queries on MuJoCo Warp when upstream exposes them
- Limited ball-joint support in :class:`~mink_warp.ConfigurationLimit`
- Published docs on GitHub Pages + CI doc build
- PyPI release

Contributions welcome via GitHub issues and pull requests.
