import jax.numpy as jnp
import numpy as np
from hahps._domain import Domain
from hahps._discretization_tree import (
    DiscretizationNode2D,
)

from hahps._pdeproblem import PDEProblem
from hahps._subtree_recomp import (
    upward_pass_subtree,
    downward_pass_subtree,
)


class Test_upward_pass_subtree:
    def test_0(self, caplog) -> None:
        """DtN case"""
        p = 7
        q = 5
        l = 3
        num_leaves = 4**l
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx_coeffs = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
        )

        T_last = upward_pass_subtree(t, subtree_height=2)

        n_bdry = domain.boundary_points.shape[0]
        assert T_last.shape == (n_bdry, n_bdry)

    def test_1(self, caplog) -> None:
        """ItI case"""
        p = 7
        q = 5
        l = 3
        num_leaves = 4**l
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx_coeffs = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
            use_ItI=True,
            eta=1.0,
        )

        T_last = upward_pass_subtree(t, subtree_height=2)

        n_bdry = domain.boundary_points.shape[0]
        assert T_last.shape == (n_bdry, n_bdry)


class Test_downward_pass_subtree:
    def test_0(self, caplog) -> None:
        """DtN case"""
        p = 7
        q = 5
        l = 3
        num_leaves = 4**l
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx_coeffs = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
        )

        n_bdry = domain.boundary_points.shape[0]

        g = jnp.zeros(n_bdry, dtype=jnp.float64)

        solns = downward_pass_subtree(t, g, subtree_height=2)

        assert solns.shape == domain.interior_points[..., 0].shape

    def test_1(self, caplog) -> None:
        """ItI case"""
        p = 7
        q = 5
        l = 3
        num_leaves = 4**l
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx_coeffs = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        source_term = jnp.array(np.random.normal(size=(num_leaves, p**2)))

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
            use_ItI=True,
            eta=1.0,
        )

        n_bdry = domain.boundary_points.shape[0]
        g = jnp.zeros(n_bdry, dtype=jnp.complex128)

        solns = downward_pass_subtree(t, g, subtree_height=2)

        assert solns.shape == domain.interior_points[..., 0].shape
