.. ha-hps documentation master file, created by
   sphinx-quickstart on Wed Mar 26 11:43:02 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

ha-hps documentation
====================

The ``hahps`` package provides utilites for constructing fast, direct solvers for systems of linear elliptic partial differential equations. It uses `jax <https://docs.jax.dev/en/latest/>`_ for hardware-accelerated linear algebra operations.

Please see our preprint `Hardware Acceleration for HPS Algorithms in Two and Three Dimensions <https://arxiv.org/abs/2503.17535>`_ for details about the algorithms implemented in this package. 
If you find this work useful, please cite our paper::

   @misc{melia2025hahps,
      title={Hardware Acceleration for HPS Algorithms in Two and Three Dimensions}, 
      author={Owen Melia and Daniel Fortunato and Jeremy Hoskins and Rebecca Willett},
      year={2025},
      eprint={2503.17535},
      archivePrefix={arXiv},
      primaryClass={math.NA},
      url={https://arxiv.org/abs/2503.17535}, 
   }





.. toctree::
   :maxdepth: 1
   :caption: Contents:

   Device_and_data
   DiscretizationNode
   Domain
   PDEProblem
   solution_methods
   quadrature

   

