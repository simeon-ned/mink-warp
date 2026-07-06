Utilities
=========

Interop
-------

.. autofunction:: mink_warp.to_wp

Helpers
-------

.. autofunction:: mink_warp.get_freejoint_dims

Constants
---------

Frame lookup tables used by tasks (defined in ``mink_warp.constants``):

- ``SUPPORTED_FRAMES`` — ``("body", "geom", "site")``
- ``FRAME_TO_ENUM`` — maps frame type string to ``mjtObj``
- ``FRAME_TO_POS_ATTR`` — batched position array attribute on ``wp_data``
- ``FRAME_TO_XMAT_ATTR`` — batched orientation array attribute on ``wp_data``

Exceptions
----------

.. autoclass:: mink_warp.MinkWarpError
   :members:

.. autoclass:: mink_warp.UnsupportedFrame
   :members:

.. autoclass:: mink_warp.InvalidFrame
   :members:

.. autoclass:: mink_warp.InvalidKeyframe
   :members:

.. autoclass:: mink_warp.TaskDefinitionError
   :members:

.. autoclass:: mink_warp.TargetNotSet
   :members:

.. autoclass:: mink_warp.InvalidTarget
   :members:

.. autoclass:: mink_warp.InvalidGain
   :members:

.. autoclass:: mink_warp.InvalidDamping
   :members:

.. autoclass:: mink_warp.InvalidConstraint
   :members:

Integration
-----------

.. autofunction:: mink_warp.integrate.integrate_qpos
