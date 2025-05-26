PDEProblem
==========

The ``PDEProblem`` class encapsulates the problem definition for a partial differential equation (PDE) in the context of the HPS method. It includes the domain, source term, and coefficients for the differential operator.

One important argument when creating a ``PDEProblem`` is the boolean argument ``use_ItI``. If this argument is set to ``True``, the solver will use the Impedance-to-Impedance version of the HPS method, which was designed for Helmholtz-like problems where DtN matrices may be ill-conditioned. Importantly, when using the ItI version, the implicit assumption is that the PDE is of the form:

.. math::
   \mathcal{L} u(x) &= f(x), \quad \text{in } \Omega, \\
   u_n(x) + i \eta u(x) &= g(x), \quad \text{on } \partial\Omega.

Note this is a Robin boundary condition with parameter :math:`\eta`. So, when using the ItI version of the code, one must remember to specify Robin boundary data when calling :func:`jaxhps.solve` or :func:`jaxhps.down_pass.down_pass_uniform_2D_ItI`. This is in contrast to the DtN version of the HPS method, specified by ``use_ItI=False``, which assumes a Dirichlet boundary condition on the boundary of the domain.

The ``PDEProblem`` class has two utility functions, :func:`jaxhps.PDEProblem.reset` and :func:`jaxhps.PDEProblem.update_coefficients`, which are meant to be used for cases such as inverse problems, where a sequence of PDE on the same domian must be solved iteratively, each time with slightly different coefficients. 

.. note::
   In the future, I would like to make the boundary condition more explicit when initializing the ``PDEProblem``.

.. note::
   When initializing the ``PDEProblem`` class, the ``source`` argument is optional. This is because the :func:`jaxhps.build_solver` routine for 2D uniform problems is implemented to build the solver for an arbitrary source term. In this case, :func:`jaxhps.solve` performs an upward pass to compute a particular solution given a new source term, and then a downward pass to compute the homogeneous solution.

   If ``source`` is specified when initializing the ``PDEProblem`` instance, the :func:`jaxhps.build_solver` routine will only build a solver for that particular source term.

.. autoclass:: jaxhps.PDEProblem
   :members:
   :member-order: bysource