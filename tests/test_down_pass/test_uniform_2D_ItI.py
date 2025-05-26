import numpy as np
import jax.numpy as jnp
import logging
from jaxhps.merge._uniform_2D_ItI import (
    merge_stage_uniform_2D_ItI,
)

from jaxhps.local_solve._uniform_2D_ItI import (
    local_solve_stage_uniform_2D_ItI,
)
from jaxhps.down_pass._uniform_2D_ItI import (
    down_pass_uniform_2D_ItI,
    _propogate_down_2D_ItI,
)
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem
import jax


class Test__propogate_down_2D_ItI:
    def test_0(self) -> None:
        """Tests to make sure returns without error."""
        n_child = 8
        n_src = 3
        S_arr = np.random.normal(size=(8 * n_child, 8 * n_child))
        bdry_data = np.random.normal(size=(8 * n_child, n_src))
        f = np.random.normal(size=(8 * n_child, n_src))

        out = _propogate_down_2D_ItI(S_arr, bdry_data, f)
        expected_out_shape = (4, 4 * n_child, n_src)
        assert out.shape == expected_out_shape
        jax.clear_caches()


class Test_down_pass_uniform_2D_ItI:
    def test_0(self) -> None:
        p = 6
        q = 4
        l = 2
        eta = 4.0

        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        domain = Domain(p=p, q=q, root=root, L=l)
        n_leaves = 4**l

        d_xx_coeffs = np.random.normal(size=(n_leaves, p**2))
        source_term = np.random.normal(size=(n_leaves, p**2))
        print("test_0: d_xx_coeffs = ", d_xx_coeffs.shape)
        print("test_0: source_term = ", source_term.shape)

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
            use_ItI=True,
            eta=eta,
        )

        Y_arr, T_arr, v_arr, h_arr = local_solve_stage_uniform_2D_ItI(
            pde_problem=t
        )

        assert Y_arr.shape == (n_leaves, p**2, 4 * q)
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        S_arr_lst, g_tilde_lst = merge_stage_uniform_2D_ItI(
            T_arr=T_arr, h_arr=h_arr, l=l
        )

        boundary_data = jnp.ones_like(t.domain.boundary_points[..., 0])

        leaf_solns = down_pass_uniform_2D_ItI(
            boundary_data,
            S_arr_lst,
            g_tilde_lst,
            Y_arr,
            v_arr,
        )
        assert leaf_solns.shape == (n_leaves, p**2)
        jax.clear_caches()

    def test_1(self, caplog) -> None:
        """Tests early exit when Y_arr and v_arr are set to None"""
        caplog.set_level(logging.DEBUG)
        p = 6
        q = 4
        l = 3
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        n_bdry = domain.boundary_points.shape[0]

        S_lst = []
        S_lst.insert(0, jnp.ones((1, n_bdry, n_bdry)))
        S_lst.insert(0, jnp.ones((4, n_bdry // 2, n_bdry // 2)))

        g_tilde_lst = []
        g_tilde_lst.insert(0, jnp.ones((1, n_bdry)))
        g_tilde_lst.insert(0, jnp.ones((4, n_bdry // 2)))

        bdry_data = jnp.ones((n_bdry))
        out = down_pass_uniform_2D_ItI(
            boundary_data=bdry_data,
            S_lst=S_lst,
            g_tilde_lst=g_tilde_lst,
            Y_arr=None,
            v_arr=None,
        )
        expected_out_shape = (16, n_bdry // 4)
        assert out.shape == expected_out_shape

    def test_2(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        """Tests n_src = 3"""
        p = 6
        q = 4
        l = 2
        eta = 4.0
        n_src = 3

        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        domain = Domain(p=p, q=q, root=root, L=l)
        n_leaves = 4**l

        d_xx_coeffs = np.random.normal(size=(n_leaves, p**2))
        source_term = np.random.normal(size=(n_leaves, p**2, n_src))
        print("test_0: d_xx_coeffs = ", d_xx_coeffs.shape)
        print("test_0: source_term = ", source_term.shape)

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
            use_ItI=True,
            eta=eta,
        )

        Y_arr, T_arr, v_arr, h_arr = local_solve_stage_uniform_2D_ItI(
            pde_problem=t
        )

        assert Y_arr.shape == (n_leaves, p**2, 4 * q)
        # n_leaves, n_bdry, _ = DtN_arr.shape
        # DtN_arr = DtN_arr.reshape((int(n_leaves / 2), 2, n_bdry, n_bdry))
        # v_prime_arr = v_prime_arr.reshape((int(n_leaves / 2), 2, 4 * t.q))

        S_arr_lst, g_tilde_lst = merge_stage_uniform_2D_ItI(
            T_arr=T_arr, h_arr=h_arr, l=l
        )

        boundary_data = jnp.ones((t.domain.boundary_points.shape[0], n_src))

        leaf_solns = down_pass_uniform_2D_ItI(
            boundary_data,
            S_arr_lst,
            g_tilde_lst,
            Y_arr,
            v_arr,
        )
        assert leaf_solns.shape == (n_leaves, p**2, n_src)
        jax.clear_caches()
