Contributing
=========================

We welcome contributions to the ``jaxhps`` package! Please make contributions through pull requests `on GitHub <https://github.com/meliao/ha-hps>`_. To make development easier, forking the repository then installing the package in editable mode and using the ``dev_requirements.txt`` is recommended: 

.. code:: bash

   cd ha-hps
   pip install dev_requirements.txt
   pip install -e .


Potential contributions include:
---------------------------------
- Any bug fixes or improvements raised in the `issues <https://github.com/meliao/ha-hps/issues>`_.
- Adding a class abstracting different types of boundary conditions, such as Dirichlet, Neumann, Robin, or boundary conditions specified by a boundary integral equation.
- Improving parallelization of the code in the merge step for adaptive discretizations. Currently, the merge step is not parallelized using a jax construct like ``vmap``.