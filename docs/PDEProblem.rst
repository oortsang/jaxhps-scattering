PDEProblem
==========

The ``PDEProblem`` class encapsulates the problem definition for a partial differential equation (PDE) in the context of the HPS method. It includes the domain, source term, and coefficients for the differential operator.

One important argument when creating a ``PDEProblem`` is the boolean argument ``use_ItI``. If this argument is set to ``True``, the solver will use the Impedance-to-Impedance version of the HPS method, which was designed for Helmholtz-like problems where DtN matrices may be ill-conditioned. Importantly, when using the ItI version, the implicit assumption is that the PDE is of the form:

.. math::
   \mathcal{L} u(x) &= f(x), \quad \text{in } \Omega, \\
   u_n(x) + i \eta u(x) &= g(x), \quad \text{on } \partial\Omega.

Note this is a Robin boundary condition with parameter :math:`\eta`. So, when using the ItI version of the code, one must remember to specify Robin boundary data when calling :func:`hahps.solve` or :func:`hahps.down_pass.down_pass_uniform_2D_ItI`. This is in contrast to the DtN version of the HPS method, specified by ``use_ItI=False``, which assumes a Dirichlet boundary condition on the boundary of the domain.

The ``PDEProblem`` class has two utility functions, :func:`hahps.PDEProblem.reset` and :func:`hahps.PDEProblem.update_coefficients`, which are meant to be used for cases such as inverse problems, where a sequence of PDE on the same domian must be solved iteratively, each time with slightly different coefficients. 

.. note::
   In the future, I would like to make the boundary condition more explicit when initializing the ``PDEProblem``.


.. autoclass:: hahps.PDEProblem
   :members:
   :member-order: bysource