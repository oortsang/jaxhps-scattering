Spatial discretization
=======================

At a high level, we represent the spatial domain with a tree of ``DiscretizationNode2D`` or ``DiscretizationNode3D`` objects. Generally, the user will specify a root Node, and then pass the root along with other information to a constructor of a ``Domain`` object. The ``Domain`` object will build the discretization tree and populate all of the discretization points. 

For instance, in the main page, we saw how to use the ``Domain`` constructor to build a uniform 2D discretization. We can also use a constructor to build an adaptive discretization. The adaptive discretization is built by recursively subdividing nodes until a specified function can be represented to a desired accuracy. We will use a simulated 2D permittivity field as an example. 

.. code:: python

   import hahps

   # Not distributed in the hahps package,
   # see https://github.com/meliao/ha-hps/blob/main/examples/poisson_boltzmann_utils.py
   from examples.poisson_boltzmann_utils import permittivity_2D

   # Create a root node for the domain
   root = hahps.DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)

   # Create a Domain object with adaptive discretization
   domain = hahps.Domain.from_adaptive_discretization(p=16,
                                                      q=14,
                                                      root=root,
                                                      f=permittivity_2D, # Can also specify a list of functions
                                                      tol=1e-06)
   
   print(domain.n_leaves)



DiscretizationNode objects
----------------------------


.. autoclass:: hahps.DiscretizationNode2D
   :members:
   :member-order: bysource

.. autoclass:: hahps.DiscretizationNode3D
   :members:
   :member-order: bysource

.. autofunction:: hahps.get_all_leaves

Domain object
----------------

.. autoclass:: hahps.Domain
   :members:
   :member-order: bysource
