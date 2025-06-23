import logging
import jax.numpy as jnp
import jax
import numpy as np
from jaxhps._build_solver import build_solver
from jaxhps._solve import solve
from jaxhps._domain import Domain
from jaxhps._discretization_tree import (
    DiscretizationNode2D,
    DiscretizationNode3D,
)
from jaxhps._discretization_tree_operations_2D import add_four_children
from jaxhps._discretization_tree_operations_3D import add_eight_children

from jaxhps._pdeproblem import PDEProblem


class Test_solve:
    def test_0(self, caplog) -> None:
        """Uniform 2D DtN"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert len(pde_problem.g_tilde_lst) == L

        assert pde_problem.S_lst[-1].shape == (1, n_bdry // 2, n_bdry)
        assert pde_problem.g_tilde_lst[-1].shape == (1, n_bdry // 2)

        # Solve the problem
        bdry_data = jnp.array(np.random.normal(size=n_bdry))

        solns = solve(pde_problem, bdry_data)

        assert solns.shape == (domain.interior_points[..., 0].shape)
        assert not isinstance(solns, jax.core.Tracer)

    def test_1(self, caplog) -> None:
        """Uniform 2D ItI"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
            use_ItI=True,
            eta=1.0,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert len(pde_problem.g_tilde_lst) == L

        # g_tilde has shape n_bdry in the ItI case.
        assert pde_problem.S_lst[-1].shape == (1, n_bdry, n_bdry)
        assert pde_problem.g_tilde_lst[-1].shape == (1, n_bdry)
        # Solve the problem
        bdry_data = jnp.array(np.random.normal(size=n_bdry))

        solns = solve(pde_problem, bdry_data)

        assert solns.shape == (domain.interior_points[..., 0].shape)
        assert not isinstance(solns, jax.core.Tracer)

    def test_2(self, caplog) -> None:
        """Uniform 3D DtN"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert len(pde_problem.g_tilde_lst) == L

        logging.debug(
            "Here are S_lst shapes: %s", [s.shape for s in pde_problem.S_lst]
        )

        assert pde_problem.S_lst[-1].shape == (n_bdry // 2, n_bdry)
        assert pde_problem.g_tilde_lst[-1].shape == (n_bdry // 2,)

        # Solve the problem
        bdry_data = jnp.array(np.random.normal(size=n_bdry))

        solns = solve(pde_problem, bdry_data)

        assert solns.shape == (domain.interior_points[..., 0].shape)
        assert not isinstance(solns, jax.core.Tracer)

    def test_3(self, caplog) -> None:
        """Adaptive 2D DtN"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        add_four_children(root, root=root, q=q)
        add_four_children(root.children[0], root=root, q=q)
        add_four_children(root.children[1], root=root, q=q)
        add_four_children(root.children[0].children[1], root=root, q=q)

        domain = Domain(p=p, q=q, root=root)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        bdry_data_lst = domain.get_adaptive_boundary_data_lst(
            lambda x: jnp.zeros_like(x[..., 0])
        )

        solns = solve(pde_problem, bdry_data_lst)
        assert solns.shape == (domain.interior_points[..., 0].shape)
        assert not isinstance(solns, jax.core.Tracer)

    def test_4(self, caplog) -> None:
        """Adaptive 3D DtN"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4

        # Create a uniform 2D domain
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )

        add_eight_children(root, root=root, q=q)
        add_eight_children(root.children[0], root=root, q=q)
        add_eight_children(root.children[1], root=root, q=q)

        domain = Domain(p=p, q=q, root=root)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        bdry_data_lst = domain.get_adaptive_boundary_data_lst(
            lambda x: jnp.zeros_like(x[..., 0])
        )

        solns = solve(pde_problem, bdry_data_lst)
        assert solns.shape == (domain.interior_points[..., 0].shape)

    def test_5(self, caplog) -> None:
        """Uniform 2D ItI with up and down passes"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            use_ItI=True,
            eta=1.0,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L

        # g_tilde has shape n_bdry in the ItI case.
        assert pde_problem.S_lst[-1].shape == (1, n_bdry, n_bdry)
        # Solve the problem
        bdry_data = jnp.array(np.random.normal(size=n_bdry))

        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        solns = solve(pde_problem, bdry_data, source=source)
        assert not isinstance(solns, jax.core.Tracer)

        assert solns.shape == (domain.interior_points[..., 0].shape)

        source2 = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        bdry_data2 = jnp.array(np.random.normal(size=n_bdry))

        solns2 = solve(pde_problem, bdry_data2, source=source2)
        assert solns2.shape == (domain.interior_points[..., 0].shape)
        assert not isinstance(solns2, jax.core.Tracer)

    def test_6(self, caplog) -> None:
        """Uniform 2D DtN with up and down passes"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L

        # g_tilde has shape n_bdry in the ItI case.
        assert pde_problem.S_lst[-1].shape == (1, n_bdry // 2, n_bdry)
        # Solve the problem
        bdry_data = jnp.array(np.random.normal(size=(n_bdry, 2)))

        source = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        solns = solve(pde_problem, bdry_data, source=source)
        assert not isinstance(solns, jax.core.Tracer)

        assert solns.shape == (domain.interior_points.shape)

        source2 = jnp.array(
            np.random.normal(size=domain.interior_points.shape)
        )
        bdry_data2 = jnp.array(np.random.normal(size=(n_bdry, 2)))

        solns2 = solve(pde_problem, bdry_data2, source=source2)
        assert solns2.shape == (domain.interior_points.shape)
        assert not isinstance(solns2, jax.core.Tracer)

    def test_7(self, caplog) -> None:
        """Uniform 2D DtN with multiple sources"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2
        nsrc = 3

        # Create a uniform 2D domain
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=(domain.n_leaves, domain.p**2, nsrc))
        )

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert pde_problem.S_lst[-1].shape == (1, n_bdry // 2, n_bdry)

        # Solve the problem
        bdry_data = jnp.array(np.random.normal(size=(n_bdry, nsrc)))

        solns = solve(pde_problem, bdry_data, source=source)

        assert solns.shape == source.shape
        assert not isinstance(solns, jax.core.Tracer)

    def test_8(self, caplog) -> None:
        """Uniform 3D DtN with multiple sources"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        L = 2
        nsrc = 2

        # Create a uniform 2D domain
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )

        domain = Domain(p=p, q=q, root=root, L=L)

        d_xx_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )

        d_yy_coeffs = jnp.array(
            np.random.normal(size=domain.interior_points[..., 0].shape)
        )
        source = jnp.array(
            np.random.normal(size=(domain.n_leaves, domain.p**3, nsrc))
        )
        logging.debug("Source shape: %s", source.shape)

        # Create a PDEProblem instance
        pde_problem = PDEProblem(
            domain=domain,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
            source=source,
        )

        # Build the solver
        T = build_solver(pde_problem, return_top_T=True)

        n_bdry = domain.boundary_points.shape[0]
        assert T.shape == (n_bdry, n_bdry)

        assert len(pde_problem.S_lst) == L
        assert len(pde_problem.g_tilde_lst) == L

        logging.debug(
            "Here are S_lst shapes: %s", [s.shape for s in pde_problem.S_lst]
        )

        assert pde_problem.S_lst[-1].shape == (n_bdry // 2, n_bdry)
        assert pde_problem.g_tilde_lst[-1].shape == (n_bdry // 2, nsrc)

        # Solve the problem
        bdry_data = jnp.array(np.random.normal(size=(n_bdry, nsrc)))

        solns = solve(pde_problem, bdry_data)

        assert solns.shape == source.shape
        assert not isinstance(solns, jax.core.Tracer)
