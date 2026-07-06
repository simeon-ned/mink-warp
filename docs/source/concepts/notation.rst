.. _notation:

Notation
========

mink-warp follows the same frame and task conventions as Mink and Pink
(Pinocchio). Subscripts are read right-to-left for transforms; superscripts
indicate the frame in which a vector is expressed.

.. list-table::
   :header-rows: 1
   :widths: 70 30

   * - Quantity
     - Notation
   * - Transform from frame :math:`A` to frame :math:`B`
     - :math:`T_{BA} \in SE(3)`
   * - Position of frame :math:`B` origin in frame :math:`A`
     - :math:`{}^A p_B`
   * - World (inertial) frame
     - :math:`0` or :math:`W`
   * - Configuration vector
     - :math:`q \in \mathbb{R}^{n_q}` (batched: ``(nworld, nq)``)
   * - Tangent / velocity
     - :math:`v \in \mathbb{R}^{n_v}`, :math:`\Delta q = v\,\mathrm{d}t`
   * - Task error and Jacobian
     - :math:`e(q) \in \mathbb{R}^k`, :math:`J(q) \in \mathbb{R}^{k \times n_v}`

Composition (read transforms left to right):

.. math::

    T_{CA} = T_{CB}\, T_{BA}

Frame task
----------

For regulated frame :math:`b`, target :math:`t`, and world :math:`0`:

.. math::

    e(q) = \log(T_{bt}), \qquad
    J(q) = -\mathrm{jlog}_6(T_{tb})\, {}_b J_{0b}(q)

See :class:`~mink_warp.FrameTask` and [FrameTaskJacobian]_.

Stacked IK problem
------------------

Tasks contribute to normal equations (equivalent to a weighted least-squares QP):

.. math::

    \min_{\Delta q}\ \tfrac{1}{2} \Delta q^\top H \Delta q + c^\top \Delta q,
    \quad H = \sum_i W_i^\top W_i + \mu I,\quad
    c = \sum_i -W_i^\top (\alpha_i e_i)

Hard limits add :math:`\ell \leq \Delta q \leq u` and/or :math:`G \Delta q \leq h`.
See :doc:`../workflows/constrained` and :func:`~mink_warp.solve_ik`.

The *task function approach* used here was formalized by Samson, Espiau and
Le Borgne [Samson1991]_. Lie-group errors use the logarithm map on
:math:`SE(3)`; see Solà et al. [Sola2018]_ for background. LM damping on large
errors follows Sugihara [Sugihara2011]_.

Further reading
---------------

- `Task-based inverse kinematics <https://scaron.info/robot-locomotion/inverse-kinematics.html>`_ (Pink / Mink lineage [Pink]_)
- `Spatial algebra cheat sheet <https://scaron.info/robot-locomotion/spatial-vector-algebra-cheat-sheet.html>`_
- :doc:`mink_parity` — API mapping from Mink
