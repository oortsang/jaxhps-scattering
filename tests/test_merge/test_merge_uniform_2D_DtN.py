import logging
import numpy as np
import jax.numpy as jnp
import jax

from jaxhps.merge._uniform_2D_DtN import (
    merge_stage_uniform_2D_DtN,
    _uniform_quad_merge_DtN,
)

from jaxhps.local_solve._uniform_2D_DtN import (
    local_solve_stage_uniform_2D_DtN,
)
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem


class Test__uniform_merge_stage_2D_DtN:
    def test_0(self, caplog) -> None:
        """Tests the _uniform_merge_stage_2D_DtN function returns without error."""
        caplog.set_level(logging.DEBUG)

        p = 8
        q = 6
        # L must == 3 in this test because of the hard-coded shapes
        # expected in the output.
        l = 3
        n_leaves = 4**l

        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        # This could be np.random.normal(size=(n_leaves, p**2))
        d_xx_coeffs = jnp.array(np.random.normal(size=(n_leaves, p**2)))
        d_yy_coeffs = jnp.array(np.random.normal(size=(n_leaves, p**2)))
        source_term = jnp.array(np.random.normal(size=(n_leaves, p**2)))

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
        )

        print("test_0: n_leaves = ", n_leaves)
        print("test_0: n_gauss_bdry = ", 4 * q)

        (
            Y_arr,
            T_arr,
            v_arr,
            h_arr,
        ) = local_solve_stage_uniform_2D_DtN(pde_problem=t)

        S_arr_lst, g_tilde_lst = merge_stage_uniform_2D_DtN(
            T_arr=T_arr, h_arr=h_arr, l=l
        )

        assert len(S_arr_lst) == l
        # assert len(DtN_arr_lst) == l
        assert len(g_tilde_lst) == l

        # Check to make sure the arrays haven't been deleted
        print(
            "test_0: S_arr_lst shapes: ", [S_arr.shape for S_arr in S_arr_lst]
        )
        print(
            "test_0: g_tilde_lst shapes: ",
            [v_arr.shape for v_arr in g_tilde_lst],
        )

        for i in range(l):
            assert S_arr_lst[i].shape[-2] == g_tilde_lst[i].shape[-1]

        # Check the shapes of the bottom-level output arrays
        n_quads = (n_leaves // 4) // 4
        assert S_arr_lst[0].shape == (4 * n_quads, 4 * q, 8 * q)
        assert g_tilde_lst[0].shape == (4 * n_quads, 4 * q)

        # Check the shapes of the middle-level output arrays.
        n_bdry = 16 * q
        n_interface = 8 * q
        assert S_arr_lst[1].shape == (4, n_interface, n_bdry)
        assert g_tilde_lst[1].shape == (4, n_interface)

        # Check the shapes of the top-level output arrays.
        n_root_bdry = t.domain.boundary_points.shape[0]
        n_root_interface = n_root_bdry // 2
        assert S_arr_lst[2].shape == (1, n_root_interface, n_root_bdry)
        assert g_tilde_lst[2].shape == (
            1,
            n_root_interface,
        )
        jax.clear_caches()

    def test_1(self, caplog) -> None:
        """Tests the _uniform_merge_stage_2D_DtN function returns without error, when using multiple sources."""
        caplog.set_level(logging.DEBUG)

        p = 8
        q = 6
        # L must == 3 in this test because of the hard-coded shapes
        # expected in the output.
        l = 3
        n_leaves = 4**l
        n_src = 3

        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        domain = Domain(p=p, q=q, root=root, L=l)

        # This could be np.random.normal(size=(n_leaves, p**2))
        d_xx_coeffs = jnp.array(np.random.normal(size=(n_leaves, p**2)))
        d_yy_coeffs = jnp.array(np.random.normal(size=(n_leaves, p**2)))
        source_term = jnp.array(np.random.normal(size=(n_leaves, p**2, n_src)))

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
            D_yy_coefficients=d_yy_coeffs,
        )

        print("test_0: n_leaves = ", n_leaves)
        print("test_0: n_gauss_bdry = ", 4 * q)

        (
            Y_arr,
            T_arr,
            v_arr,
            h_arr,
        ) = local_solve_stage_uniform_2D_DtN(pde_problem=t)

        S_arr_lst, g_tilde_lst = merge_stage_uniform_2D_DtN(
            T_arr=T_arr, h_arr=h_arr, l=l
        )

        assert len(S_arr_lst) == l
        # assert len(DtN_arr_lst) == l
        assert len(g_tilde_lst) == l

        # Check to make sure the arrays haven't been deleted
        print(
            "test_0: S_arr_lst shapes: ", [S_arr.shape for S_arr in S_arr_lst]
        )
        print(
            "test_0: g_tilde_lst shapes: ",
            [v_arr.shape for v_arr in g_tilde_lst],
        )

        for i in range(l):
            assert S_arr_lst[i].shape[-2] == g_tilde_lst[i].shape[-2]

        # Check the shapes of the bottom-level output arrays
        n_quads = (n_leaves // 4) // 4
        assert S_arr_lst[0].shape == (4 * n_quads, 4 * q, 8 * q)
        assert g_tilde_lst[0].shape == (4 * n_quads, 4 * q, n_src)

        # Check the shapes of the middle-level output arrays.
        n_bdry = 16 * q
        n_interface = 8 * q
        assert S_arr_lst[1].shape == (4, n_interface, n_bdry)
        assert g_tilde_lst[1].shape == (4, n_interface, n_src)

        # Check the shapes of the top-level output arrays.
        n_root_bdry = t.domain.boundary_points.shape[0]
        n_root_interface = n_root_bdry // 2
        assert S_arr_lst[2].shape == (1, n_root_interface, n_root_bdry)
        assert g_tilde_lst[2].shape == (1, n_root_interface, n_src)
        jax.clear_caches()


class Test__uniform_quad_merge:
    def test_0(self) -> None:
        n_bdry = 28
        n_bdry_int = n_bdry // 4
        n_bdry_ext = 2 * (n_bdry // 4)
        T_a = np.random.normal(size=(n_bdry, n_bdry))
        T_b = np.random.normal(size=(n_bdry, n_bdry))
        T_c = np.random.normal(size=(n_bdry, n_bdry))
        T_d = np.random.normal(size=(n_bdry, n_bdry))
        v_prime_a = np.random.normal(size=(n_bdry))
        v_prime_b = np.random.normal(size=(n_bdry))
        v_prime_c = np.random.normal(size=(n_bdry))
        v_prime_d = np.random.normal(size=(n_bdry))

        S, T, v_prime_ext, v_int = _uniform_quad_merge_DtN(
            T_a, T_b, T_c, T_d, v_prime_a, v_prime_b, v_prime_c, v_prime_d
        )

        assert S.shape == (4 * n_bdry_int, 4 * n_bdry_ext)
        assert T.shape == (4 * n_bdry_ext, 4 * n_bdry_ext)
        assert v_prime_ext.shape == (4 * n_bdry_ext,)
        assert v_int.shape == (4 * n_bdry_int,)
        jax.clear_caches()

    def test_1(self) -> None:
        n_bdry = 28
        n_bdry_int = n_bdry // 4
        n_bdry_ext = 2 * (n_bdry // 4)
        n_src = 3
        T_a = np.random.normal(size=(n_bdry, n_bdry))
        T_b = np.random.normal(size=(n_bdry, n_bdry))
        T_c = np.random.normal(size=(n_bdry, n_bdry))
        T_d = np.random.normal(size=(n_bdry, n_bdry))
        v_prime_a = np.random.normal(size=(n_bdry, n_src))
        v_prime_b = np.random.normal(size=(n_bdry, n_src))
        v_prime_c = np.random.normal(size=(n_bdry, n_src))
        v_prime_d = np.random.normal(size=(n_bdry, n_src))

        S, T, v_prime_ext, v_int = _uniform_quad_merge_DtN(
            T_a, T_b, T_c, T_d, v_prime_a, v_prime_b, v_prime_c, v_prime_d
        )

        assert S.shape == (4 * n_bdry_int, 4 * n_bdry_ext)
        assert T.shape == (4 * n_bdry_ext, 4 * n_bdry_ext)
        assert v_prime_ext.shape == (4 * n_bdry_ext, n_src)
        assert v_int.shape == (4 * n_bdry_int, n_src)
        jax.clear_caches()
