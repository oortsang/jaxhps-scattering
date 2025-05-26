import numpy as np
import jax.numpy as jnp
import jax

from jaxhps.local_solve._uniform_2D_ItI import (
    local_solve_stage_uniform_2D_ItI,
    get_ItI,
)
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._domain import Domain
from jaxhps._pdeproblem import PDEProblem


class Test__local_solve_stage_2D_ItI:
    def test_0(self) -> None:
        """Tests the solve_stage function returns without error and returns the correct shape."""

        p = 16
        q = 14
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

        Y_arr, R_arr, v_arr, g_arr = local_solve_stage_uniform_2D_ItI(
            pde_problem=t
        )

        assert Y_arr.shape == (n_leaves, p**2, 4 * q)
        assert R_arr.shape == (n_leaves, 4 * q, 4 * q)
        assert v_arr.shape == (n_leaves, p**2)
        assert g_arr.shape == (n_leaves, 4 * q)
        jax.clear_caches()

    def test_1(self) -> None:
        """Tests the solve_stage function returns without error and returns the correct shape nsrc = 3."""

        p = 16
        q = 14
        l = 3
        nsrc = 3
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
        source_term = np.random.normal(size=(n_leaves, p**2, nsrc))
        print("test_0: d_xx_coeffs = ", d_xx_coeffs.shape)
        print("test_0: source_term = ", source_term.shape)

        t = PDEProblem(
            domain=domain,
            source=source_term,
            D_xx_coefficients=d_xx_coeffs,
            use_ItI=True,
            eta=eta,
        )

        Y_arr, R_arr, v_arr, g_arr = local_solve_stage_uniform_2D_ItI(
            pde_problem=t
        )

        assert Y_arr.shape == (n_leaves, p**2, 4 * q)
        assert R_arr.shape == (n_leaves, 4 * q, 4 * q)
        assert v_arr.shape == (n_leaves, p**2, nsrc)
        assert g_arr.shape == (n_leaves, 4 * q, nsrc)
        jax.clear_caches()


class Test_get_ItI:
    def test_0(self) -> None:
        """Makes sure shapes are correct."""
        p = 4
        q = 3
        n_cheby = p**2
        n_cheby_bdry = 4 * (p - 1)

        source_term = jnp.array(
            np.random.normal(size=(n_cheby,)).astype(np.float64)
        )
        diff_operator = jnp.array(
            np.random.normal(size=(n_cheby, n_cheby)).astype(np.float64)
        )
        P = jnp.array(
            np.random.normal(size=(n_cheby_bdry, 4 * q)).astype(np.float64)
        )
        QH = jnp.array(
            np.random.normal(size=(4 * q, n_cheby)).astype(np.float64)
        )

        G = jnp.array(
            np.random.normal(size=(n_cheby_bdry, n_cheby)).astype(
                np.complex128
            )
        )

        R, Y, g_part, part_soln = get_ItI(
            diff_operator=diff_operator,
            source_term=source_term,
            P=P,
            QH=QH,
            G=G,
        )

        assert Y.shape == (n_cheby, 4 * q)
        assert R.shape == (4 * q, 4 * q)
        assert g_part.shape == (4 * q,)
        assert part_soln.shape == (n_cheby,)
        jax.clear_caches()

    def test_1(self) -> None:
        """Makes sure shapes are correct when using multiple source terms."""
        p = 4
        q = 3
        n_cheby = p**2
        n_cheby_bdry = 4 * (p - 1)
        n_src = 7

        diff_operator = jnp.array(
            np.random.normal(size=(n_cheby, n_cheby)).astype(np.float64)
        )

        source_term = jnp.array(
            np.random.normal(size=(n_cheby, n_src)).astype(np.float64)
        )
        P = jnp.array(
            np.random.normal(size=(n_cheby_bdry, 4 * q)).astype(np.float64)
        )
        QH = jnp.array(
            np.random.normal(size=(4 * q, n_cheby)).astype(np.float64)
        )

        G = jnp.array(
            np.random.normal(size=(n_cheby_bdry, n_cheby)).astype(
                np.complex128
            )
        )
        R, Y, g_part, part_soln = get_ItI(
            diff_operator=diff_operator,
            source_term=source_term,
            P=P,
            QH=QH,
            G=G,
        )

        assert Y.shape == (n_cheby, 4 * q)
        assert R.shape == (4 * q, 4 * q)
        assert g_part.shape == (4 * q, n_src)
        assert part_soln.shape == (n_cheby, n_src)

        jax.clear_caches()

    # def test_2(self) -> None:
    #     """Checks the accuracy of the ItI operator when solving a Laplace problem with zero source and
    #     polynomial impedance data.

    #     In the notes this is test case 2D.ItI.a
    #     """
    #     p = 8
    #     q = 6

    #     north = jnp.pi / 2
    #     south = -jnp.pi / 2
    #     east = jnp.pi / 2
    #     west = -jnp.pi / 2
    #     half_side_len = jnp.pi / 2
    #     root = Node(xmin=west, xmax=east, ymin=south, ymax=north)

    #     # Set up the GL boundary points
    #     bdry_pts = get_all_boundary_gauss_legendre_points(q, root)
    #     cheby_pts = get_all_leaf_2d_cheby_points(p, root)[0]

    #     # Precompute differential operators
    #     d_x, d_y, d_xx, d_yy, d_xy = precompute_diff_operators(
    #         p, half_side_len
    #     )
    #     N = precompute_N_matrix(d_x, d_y, p)
    #     N_tilde = precompute_N_tilde_matrix(d_x, d_y, p)

    #     eta = 4.0
    #     F = precompute_F_matrix(N_tilde, p, eta)
    #     G = precompute_G_matrix(N, p, eta)

    #     # Precompute interpolation operators
    #     I_P_0 = precompute_I_P_0_matrix(p, q)
    #     Q_I = precompute_Q_I_matrix(p, q)

    #     # Stack the differential operators into a single array
    #     stacked_diff_ops = jnp.stack(
    #         [d_xx, d_xy, d_yy, d_x, d_y, jnp.eye(p**2)]
    #     )

    #     # Make Laplacian coefficients
    #     lap_coeffs = jnp.ones((2, p**2), dtype=jnp.float64)
    #     which_coeffs = jnp.array([True, False, True, False, False, False])
    #     diff_operator = assemble_diff_operator(
    #         coeffs_arr=lap_coeffs,
    #         which_coeffs=which_coeffs,
    #         diff_ops=stacked_diff_ops,
    #     )
    #     source_term = jnp.zeros((p**2,), dtype=jnp.float64)

    #     # Compute ItI map
    #     R, Y, g_part, part_soln = get_ItI(
    #         diff_operator=diff_operator,
    #         source_term=source_term,
    #         I_P_0=I_P_0,
    #         Q_I=Q_I,
    #         F=F,
    #         G=G,
    #     )

    #     def f(x: jnp.array) -> jnp.array:
    #         # f(x,y) = x^2 - y^2
    #         return x[..., 0] ** 2 - x[..., 1] ** 2

    #     def dfdx(x: jnp.array) -> jnp.array:
    #         # df/dx = 2x
    #         return 2 * x[..., 0]

    #     def dfdy(x: jnp.array) -> jnp.array:
    #         # df/dy = -2y
    #         return -2 * x[..., 1]

    #     # Evaluate f on the boundary points
    #     f_vals = f(bdry_pts)

    #     f_normal_vals = jnp.concatenate(
    #         [
    #             -dfdy(bdry_pts[0:q]),
    #             dfdx(bdry_pts[q : 2 * q]),
    #             dfdy(bdry_pts[2 * q : 3 * q]),
    #             -dfdx(bdry_pts[3 * q :]),
    #         ]
    #     )

    #     incoming_impedance_data = f_normal_vals + 1j * eta * f_vals
    #     outgoing_impedance_data = R @ incoming_impedance_data
    #     outgoing_expected = f_normal_vals - 1j * eta * f_vals

    #     assert jnp.allclose(outgoing_impedance_data, outgoing_expected)
    #     assert jnp.allclose(part_soln, jnp.zeros_like(part_soln))
    #     assert jnp.allclose(g_part, jnp.zeros_like(g_part))

    #     expected_homog_soln = f(cheby_pts)
    #     computed_homog_soln = Y @ incoming_impedance_data
    #     assert jnp.allclose(expected_homog_soln, computed_homog_soln)
