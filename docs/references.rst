.. _references:

References
==========

Cross-references elsewhere use keys like ``[Mink]_``. To cite **mink-warp** itself,
use `CITATION.cff <https://github.com/simeon-ned/mink-warp/blob/main/CITATION.cff>`_
(GitHub “Cite this repository”) or the BibTeX below.

Cite mink-warp
--------------

Nedelchev, S., & Domrachev, I. (2026). *mink-warp: Batched differential inverse
kinematics on MuJoCo Warp*.

.. dropdown:: Cite
   :icon: copy

   .. literalinclude:: ../CITATION.bib
      :language: bibtex

.. _publications:

Publications
------------

.. class:: mw-bib-entry

#. Samson, C., Espiau, B., & Le Borgne, M. (1991).
   *Robot Control: The Task Function Approach*.
   Oxford University Press.

   .. dropdown:: Cite
      :icon: copy

      .. code-block:: bibtex

         @book{samson1991robot,
           title     = {Robot Control: The Task Function Approach},
           author    = {Samson, Claude and Espiau, Bernard and Le Borgne, Michel},
           year      = {1991},
           publisher = {Oxford University Press},
         }

#. Sugihara, T. (2011).
   Solvability-unconcerned inverse kinematics by the Levenberg–Marquardt method.
   *IEEE Transactions on Robotics*, 27(5), 984–991.
   `DOI <https://doi.org/10.1109/TRO.2011.2145830>`_

   .. dropdown:: Cite
      :icon: copy

      .. code-block:: bibtex

         @article{sugihara2011solvability,
           title   = {Solvability-unconcerned inverse kinematics by the {Levenberg--Marquardt} method},
           author  = {Sugihara, Tomomichi},
           journal = {IEEE Transactions on Robotics},
           volume  = {27},
           number  = {5},
           pages   = {984--991},
           year    = {2011},
         }

#. Solà, J., Deray, J., & Atchuthan, D. (2018).
   A micro Lie theory for state estimation in robotics.
   *arXiv:1812.01537*.
   `arXiv <https://arxiv.org/abs/1812.01537>`_

   .. dropdown:: Cite
      :icon: copy

      .. code-block:: bibtex

         @article{sola2018micro,
           title   = {A micro {Lie} theory for state estimation in robotics},
           author  = {Sol{\`a}, Joan and Deray, Jeremie and Atchuthan, Dennis},
           journal = {arXiv preprint arXiv:1812.01537},
           year    = {2018},
         }

#. Caron, S. (2023).
   Jacobian of a kinematic task and derivatives on manifolds.
   `Online note <https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html>`_

   .. dropdown:: Cite
      :icon: copy

      .. code-block:: bibtex

         @misc{caron2023jacobian,
           author       = {Caron, St{\'e}phane},
           title        = {Jacobian of a kinematic task and derivatives on manifolds},
           year         = {2023},
           howpublished = {\url{https://scaron.info/robotics/jacobian-of-a-kinematic-task-and-derivatives-on-manifolds.html}},
         }

.. _related-software:

Related software
----------------

.. list-table::
   :header-rows: 1
   :widths: 18 82

   * - Library
     - Summary
   * - `Mink <https://github.com/kevinzakka/mink>`_
     - Differential IK for MuJoCo — API shape mirrored by mink-warp.
   * - `Pink <https://github.com/stephane-caron/pink>`_
     - Task-based IK with Pinocchio (CPU).
   * - `MuJoCo Warp <https://github.com/google-deepmind/mujoco_warp>`_
     - GPU MuJoCo backend for :class:`~mink_warp.Configuration`.
   * - `Newton <https://github.com/newton-physics/newton>`_
     - GPU physics for robot learning (NVIDIA + Google DeepMind). See
       :doc:`source/concepts/why_mink_warp`.

All entries
-----------

Software: :download:`CITATION.bib <../CITATION.bib>` ·
Download :download:`references.bib <references.bib>` (method papers + related software) or expand below.

.. dropdown:: Full BibTeX file
   :icon: book

   .. literalinclude:: references.bib
      :language: bibtex

.. container:: mw-cite-keys

   .. [Samson1991] Samson, Espiau, Le Borgne, 1991.
   .. [Sugihara2011] Sugihara, 2011.
   .. [Sola2018] Solà et al., 2018.
   .. [FrameTaskJacobian] Caron, 2023.
   .. [Mink] Zakka, mink.
   .. [Pink] Caron, Pink.
   .. [MuJoCoWarp] Google DeepMind, MuJoCo Warp.
   .. [Newton] NVIDIA and Google DeepMind, Newton.
