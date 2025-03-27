===================
Solution Methods
===================

The package offers a few different ways to interact with the HPS algorithms, and exposes routines for performing different stages of the HPS method for different discretization types. 



Building the Fast Direct Solver
==================================

.. autofunction:: hahps.build_solver

.. autofunction:: hahps.solve


Subtree-Recomputation Solution Methods
========================================

.. autofunction:: hahps.solve_subtree

.. autofunction:: hahps.upward_pass_subtree

.. autofunction:: hahps.downward_pass_subtree



Individual HPS Algorithm Stages
================================

Local Solve Stage
-----------------

.. automodule:: hahps.local_solve
   :members:

Merge Stage
------------

.. automodule:: hahps.merge
   :members:




Downward Pass
----------------


.. automodule:: hahps.down_pass
   :members:
