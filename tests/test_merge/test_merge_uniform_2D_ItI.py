import numpy as np

from jaxhps.merge._uniform_2D_ItI import (
    merge_stage_uniform_2D_ItI,
    _uniform_quad_merge_ItI,
)

from jaxhps.local_solve._uniform_2D_ItI import (
    local_solve_stage_uniform_2D_ItI,
)
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem


class Test_merge_stage_uniform_2D_ItI:
    def test_0(self) -> None:
        """Tests the function returns without error."""
        p = 6
        q = 4
        l = 3
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
        print(
            "test_0: S_arr_lst shapes = ", [S_arr.shape for S_arr in S_arr_lst]
        )

        assert len(S_arr_lst) == l
        assert len(g_tilde_lst) == l

        for i in range(l):
            print("test_0: i=", i)
            print("test_0: S_arr_lst[i].shape = ", S_arr_lst[i].shape)
            print("test_0: g_tilde_lst[i].shape = ", g_tilde_lst[i].shape)
            assert S_arr_lst[i].shape[-2] == g_tilde_lst[i].shape[-1]

        # Check the shapes of the bottom-level output arrays
        n_quads = (n_leaves // 4) // 4
        assert S_arr_lst[0].shape == (4 * n_quads, 8 * q, 8 * q)
        assert g_tilde_lst[0].shape == (4 * n_quads, 8 * q)

        # Check the shapes of the middle-level output arrays.
        n_bdry = 16 * q
        n_interface = 16 * q
        assert S_arr_lst[1].shape == (4, n_interface, n_bdry)
        assert g_tilde_lst[1].shape == (4, n_interface)

        # Check the shapes of the top-level output arrays.
        n_root_bdry = t.domain.boundary_points.shape[0]
        n_root_interface = n_root_bdry
        assert S_arr_lst[2].shape == (1, n_root_interface, n_root_bdry)
        assert g_tilde_lst[2].shape == (1, n_root_interface)

    def test_1(self) -> None:
        """Tests the function returns without error."""
        p = 6
        q = 4
        l = 3
        n_src = 3
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
        print(
            "test_0: S_arr_lst shapes = ", [S_arr.shape for S_arr in S_arr_lst]
        )

        assert len(S_arr_lst) == l
        assert len(g_tilde_lst) == l

        for i in range(l):
            print("test_0: i=", i)
            print("test_0: S_arr_lst[i].shape = ", S_arr_lst[i].shape)
            print("test_0: g_tilde_lst[i].shape = ", g_tilde_lst[i].shape)
            assert S_arr_lst[i].shape[-2] == g_tilde_lst[i].shape[-2]

        # Check the shapes of the bottom-level output arrays
        n_quads = (n_leaves // 4) // 4
        assert S_arr_lst[0].shape == (4 * n_quads, 8 * q, 8 * q)
        assert g_tilde_lst[0].shape == (4 * n_quads, 8 * q, n_src)

        # Check the shapes of the middle-level output arrays.
        n_bdry = 16 * q
        n_interface = 16 * q
        assert S_arr_lst[1].shape == (4, n_interface, n_bdry)
        assert g_tilde_lst[1].shape == (4, n_interface, n_src)

        # Check the shapes of the top-level output arrays.
        n_root_bdry = t.domain.boundary_points.shape[0]
        n_root_interface = n_root_bdry
        assert S_arr_lst[2].shape == (1, n_root_interface, n_root_bdry)
        assert g_tilde_lst[2].shape == (1, n_root_interface, n_src)


class Test__uniform_quad_merge_ItI:
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
        print("test_0: T_a shape: ", T_a.shape)
        print("test_0: v_prime_a shape: ", v_prime_a.shape)
        S, R, h, f = _uniform_quad_merge_ItI(
            T_a, T_b, T_c, T_d, v_prime_a, v_prime_b, v_prime_c, v_prime_d
        )

        assert S.shape == (8 * n_bdry_int, 4 * n_bdry_ext)
        assert R.shape == (4 * n_bdry_ext, 4 * n_bdry_ext)
        assert h.shape == (4 * n_bdry_ext,)
        assert f.shape == (8 * n_bdry_int,)
