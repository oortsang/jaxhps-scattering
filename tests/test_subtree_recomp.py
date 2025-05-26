import jax.numpy as jnp
import jax
import numpy as np
from jaxhps._domain import Domain
from jaxhps._discretization_tree import (
    DiscretizationNode2D,
)
from jaxhps._pdeproblem import PDEProblem
from jaxhps._subtree_recomp import (
    upward_pass_subtree,
    downward_pass_subtree,
)

import logging

bool_gpu_available = any(
    "NVIDIA" in device.device_kind for device in jax.devices()
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

    def test_2(self, caplog) -> None:
        """Multi-Source ItI case"""
        p = 7
        q = 5
        l = 3
        nsrc = 3
        num_leaves = 4**l
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx_coeffs = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        source_term = jnp.array(
            np.random.normal(size=(num_leaves, p**2, nsrc))
        )

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
    # @pytest.mark.skipif(
    #     not bool_gpu_available, reason=f"Skipping because GPU is not available. Here's the bool: {bool_gpu_available}"
    # )
    def test_0(self, caplog) -> None:
        """DtN case"""
        caplog.set_level(logging.DEBUG)
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

        T_top = upward_pass_subtree(
            pde_problem=t,
            subtree_height=2,
            compute_device=jax.devices()[0],
            host_device=jax.devices()[0],
        )

        n_bdry = domain.boundary_points.shape[0]

        assert T_top.shape == (n_bdry, n_bdry)

        g = jnp.zeros(n_bdry, dtype=jnp.float64)

        solns = downward_pass_subtree(t, g, subtree_height=2)

        assert solns.shape == domain.interior_points[..., 0].shape

    # @pytest.mark.skipif(
    #     not bool_gpu_available, reason=f"Skipping because GPU is not available. Here's the bool: {bool_gpu_available}"
    # )
    def test_1(self, caplog) -> None:
        """ItI case"""
        caplog.set_level(logging.DEBUG)

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

        T_top = upward_pass_subtree(
            pde_problem=t,
            subtree_height=2,
            compute_device=jax.devices()[0],
            host_device=jax.devices()[0],
        )

        n_bdry = domain.boundary_points.shape[0]

        assert T_top.shape == (n_bdry, n_bdry)

        g = jnp.zeros(n_bdry, dtype=jnp.complex128)

        solns = downward_pass_subtree(t, g, subtree_height=2)

        assert solns.shape == domain.interior_points[..., 0].shape

    # @pytest.mark.skipif(
    #     not bool_gpu_available, reason=f"Skipping because GPU is not available. Here's the bool: {bool_gpu_available}"
    # )
    def test_2(self, caplog) -> None:
        """Multi-Source ItI case"""
        caplog.set_level(logging.DEBUG)
        p = 7
        q = 5
        l = 3
        nsrc = 3
        num_leaves = 4**l
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        d_xx_coeffs = jnp.array(np.random.normal(size=(num_leaves, p**2)))
        source_term = jnp.array(
            np.random.normal(size=(num_leaves, p**2, nsrc))
        )

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
            use_ItI=True,
            eta=1.0,
        )

        T_top = upward_pass_subtree(
            pde_problem=t,
            subtree_height=2,
            compute_device=jax.devices()[0],
            host_device=jax.devices()[0],
        )

        n_bdry = domain.boundary_points.shape[0]
        assert T_top.shape == (n_bdry, n_bdry)

        g = jnp.zeros((n_bdry, nsrc), dtype=jnp.complex128)

        solns = downward_pass_subtree(t, g, subtree_height=2)

        assert solns.shape == (domain.n_leaves, p**2, nsrc)
