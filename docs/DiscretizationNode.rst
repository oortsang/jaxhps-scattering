Spatial discretization
=======================

At a high level, we represent the spatial domain with a tree of :class:`jaxhps.DiscretizationNode2D` or :class:`jaxhps.DiscretizationNode3D` objects. Generally, the user will specify a root Node, and then pass the root along with other information to a constructor of a :class:`jaxhps.Domain` object. The :class:`jaxhps.Domain` object will build the discretization tree and populate all of the discretization points. 

For instance, in the :doc:`index`, we saw how to use the :class:`jaxhps.Domain` constructor to build a uniform 2D discretization. We can also use a constructor to build an adaptive discretization. The adaptive discretization is built by recursively subdividing nodes until a specified function can be represented to a desired accuracy. We will use a simulated 2D permittivity field as an example. 

.. code:: python

   import jaxhps

   # Not distributed in the jaxhps package,
   # see https://github.com/meliao/jaxhps/blob/main/examples/poisson_boltzmann_utils.py
   from poisson_boltzmann_utils import permittivity_2D

   # Create a root node for the domain
   root = jaxhps.DiscretizationNode2D(xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0)

   # Create a Domain object with adaptive discretization
   domain = jaxhps.Domain.from_adaptive_discretization(p=16,
                                                      q=14,
                                                      root=root,
                                                      f=permittivity_2D, # Can also specify a list of functions
                                                      tol=1e-06)
   
   print(domain.n_leaves)



DiscretizationNode objects
----------------------------


.. autoclass:: jaxhps.DiscretizationNode2D
   :members:
   :member-order: bysource

.. autoclass:: jaxhps.DiscretizationNode3D
   :members:
   :member-order: bysource

.. autofunction:: jaxhps.get_all_leaves

Domain object
----------------

.. autoclass:: jaxhps.Domain
   :members:
   :member-order: bysource
