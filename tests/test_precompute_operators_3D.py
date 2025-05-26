import numpy as np
import jax.numpy as jnp

from jaxhps.quadrature import chebyshev_points, affine_transform
from jaxhps._grid_creation_3D import (
    bounds_to_cheby_points_3D,
    compute_boundary_Gauss_points_adaptive_3D,
    compute_interior_Chebyshev_points_adaptive_3D,
)
from jaxhps._discretization_tree import DiscretizationNode3D
from jaxhps._discretization_tree_operations_3D import add_eight_children
from jaxhps._precompute_operators_3D import (
    precompute_diff_operators_3D,
    precompute_P_3D_DtN,
    precompute_Q_3D_DtN,
    get_face_1_idxes,
    precompute_projection_ops_3D,
    precompute_L_8f1,
    indexing_for_refinement_operator,
)


class Test_precompute_diff_operators_3D:
    def test_0(self) -> None:
        """Makes sure the outputs are the right shape."""
        p = 5
        half_side_len = 1 / 2
        du_dx, du_dy, du_dz, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz = (
            precompute_diff_operators_3D(p, half_side_len)
        )

        assert du_dx.shape == (p**3, p**3)
        assert du_dy.shape == (p**3, p**3)
        assert du_dz.shape == (p**3, p**3)
        assert d_xx.shape == (p**3, p**3)
        assert d_yy.shape == (p**3, p**3)
        assert d_zz.shape == (p**3, p**3)
        assert d_xy.shape == (p**3, p**3)
        assert d_xz.shape == (p**3, p**3)
        assert d_yz.shape == (p**3, p**3)

    def test_1(self) -> None:
        """Tests du_dx, du_dy, du_dz work as expected with f(x, y, z) = x + y + z."""

        p = 5
        half_side_len = np.pi / 2
        # corners = np.array(
        #     [
        #         [-np.pi / 2, -np.pi / 2, -np.pi / 2],
        #         [np.pi / 2, np.pi / 2, np.pi / 2],
        #     ]
        # )
        bounds = jnp.array(
            [
                -np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                np.pi / 2,
                -np.pi / 2,
                np.pi / 2,
            ]
        )
        pp = chebyshev_points(p)
        cheby_pts = bounds_to_cheby_points_3D(bounds, pp)

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return x[:, 0] + x[:, 1] + x[:, 2]

        f_evals = f(cheby_pts)

        du_dx, du_dy, du_dz, _, _, _, _, _, _ = precompute_diff_operators_3D(
            p, half_side_len
        )

        du_dx_f = du_dx @ f_evals
        du_dy_f = du_dy @ f_evals
        du_dz_f = du_dz @ f_evals

        assert jnp.allclose(du_dx_f, 1)
        assert jnp.allclose(du_dy_f, 1)
        assert jnp.allclose(du_dz_f, 1)

    def test_2(self) -> None:
        """Tests du_dx, du_dy, du_dz work as expected with f(x, y, z) = sin(x) + sin(y) + sin(z).
        Needs to have a relatively high p to fit the non-polynomial data."""

        p = 10
        half_side_len = 1 / 2
        bounds = jnp.array([0, 1, 0, 1, 0, 1])
        pp = chebyshev_points(p)
        cheby_pts = bounds_to_cheby_points_3D(bounds, pp)

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return jnp.sin(x[:, 0]) + jnp.sin(x[:, 1]) + jnp.sin(x[:, 2])

        f_evals = f(cheby_pts)

        du_dx, du_dy, du_dz, _, _, _, _, _, _ = precompute_diff_operators_3D(
            p, half_side_len
        )

        d_xx_f = du_dx @ f_evals
        d_yy_f = du_dy @ f_evals
        d_zz_f = du_dz @ f_evals

        assert jnp.allclose(d_xx_f, jnp.cos(cheby_pts[:, 0])), jnp.max(
            jnp.abs(d_xx_f - jnp.cos(cheby_pts[:, 0]))
        )
        assert jnp.allclose(d_yy_f, jnp.cos(cheby_pts[:, 1]))
        assert jnp.allclose(d_zz_f, jnp.cos(cheby_pts[:, 2]))

    def test_3(self) -> None:
        """Tests du_dxx, du_dyy, du_dzz work as expected with f(x, y, z) = x^2 + y^2 + z^2."""
        p = 5
        half_side_len = 1 / 2
        bounds = jnp.array([0, 1, 0, 1, 0, 1])
        pp = chebyshev_points(p)
        cheby_pts = bounds_to_cheby_points_3D(bounds, pp)

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return x[:, 0] ** 2 + x[:, 1] ** 2 + x[:, 2] ** 2

        f_evals = f(cheby_pts)
        _, _, _, d_xx, d_yy, d_zz, d_xy, d_xz, d_yz = (
            precompute_diff_operators_3D(p, half_side_len)
        )

        d_xx_f = d_xx @ f_evals
        d_yy_f = d_yy @ f_evals
        d_zz_f = d_zz @ f_evals
        d_xy_f = d_xy @ f_evals
        d_xz_f = d_xz @ f_evals
        d_yz_f = d_yz @ f_evals

        assert jnp.allclose(d_xx_f, 2)
        assert jnp.allclose(d_yy_f, 2)
        assert jnp.allclose(d_zz_f, 2)
        assert jnp.allclose(d_xy_f, 0)
        assert jnp.allclose(d_xz_f, 0)
        assert jnp.allclose(d_yz_f, 0)

    def test_4(self) -> None:
        """Tests du_dxy, du_dxz, du_dyz work as expected with f(x, y, z) = xyz"""

        p = 5
        half_side_len = 1 / 2
        bounds = jnp.array([0, 1, 0, 1, 0, 1])
        pp = chebyshev_points(p)
        cheby_pts = bounds_to_cheby_points_3D(bounds, pp)

        def f(x: jnp.ndarray) -> jnp.ndarray:
            return x[:, 0] * x[:, 1] * x[:, 2]

        f_evals = f(cheby_pts)
        _, _, _, _, _, _, d_xy, d_xz, d_yz = precompute_diff_operators_3D(
            p, half_side_len
        )

        d_xy_f = d_xy @ f_evals
        d_xz_f = d_xz @ f_evals
        d_yz_f = d_yz @ f_evals

        assert jnp.allclose(d_xy_f, cheby_pts[:, 2])
        assert jnp.allclose(d_xz_f, cheby_pts[:, 1])
        assert jnp.allclose(d_yz_f, cheby_pts[:, 0])


class Test_precompute_P_3D_DtN:
    def test_0(self) -> None:
        # Test case 1
        p = 2
        q = 3
        o = precompute_P_3D_DtN(p, q)
        expected_shape = (p**3 - (p - 2) ** 3, 6 * q**2)
        assert o.shape == expected_shape

    def test_1(self) -> None:
        """Tests accuracy of interpolation on f(x,y,z) = x + y + z."""

        def f(x) -> jnp.ndarray:
            return x[:, 0] + x[:, 1] + x[:, 2]

        p = 4
        q = 2
        mat = precompute_P_3D_DtN(p, q)

        # corners = jnp.array(
        #     [
        #         [-1, -1, -1],
        #         [1, 1, 1],
        #     ]
        # )
        bounds = jnp.array([-1, 1, -1, 1, -1, 1])
        n_cheby_bdry = p**3 - (p - 2) ** 3
        cheby_pts = chebyshev_points(p)
        cheby_pts_lst = bounds_to_cheby_points_3D(bounds, cheby_pts)[
            :n_cheby_bdry
        ]
        # Create gauss_pts_lst by defining a DiscretizationNode3D and calling the
        # compute_boundary_Gauss_points_adaptive_3D function.
        root = DiscretizationNode3D(
            xmin=bounds[0],
            xmax=bounds[1],
            ymin=bounds[2],
            ymax=bounds[3],
            zmin=bounds[4],
            zmax=bounds[5],
        )
        gauss_pts_lst = compute_boundary_Gauss_points_adaptive_3D(root, q)

        f_gauss_evals = f(gauss_pts_lst)
        f_cheby_evals = f(cheby_pts_lst)

        f_interp = mat @ f_gauss_evals

        print("test_1: f_interp shape: ", f_interp.shape)

        # Check accuracy on the corners
        corner_idxes = jnp.array(
            [
                0,
                p - 1,
                p**2 - p,
                p**2 - 1,
                p**2,
                p**2 + p - 1,
                2 * p**2 - p,
                2 * p**2 - 1,
            ]
        )
        f_interp_corners = f_interp[corner_idxes]
        f_cheby_corners = f_cheby_evals[corner_idxes]
        print("test_1: f_cheby_corners: ", f_cheby_corners)
        print("test_1: f_interp_corners: ", f_interp_corners)

        assert jnp.allclose(f_interp, f_cheby_evals)

        # Check accuracy on face 1.
        face_1_idxes = get_face_1_idxes(p)
        f_interp_face_1 = f_interp[face_1_idxes]
        f_cheby_face_1 = f_cheby_evals[face_1_idxes]
        print("test_1: f_interp_face_1: ", f_interp_face_1)
        print("test_1: f_cheby_face_1: ", f_cheby_face_1)
        print("test_1: diffs: ", f_interp_face_1 - f_cheby_face_1)
        assert jnp.allclose(f_interp_face_1, f_cheby_face_1)

        # Check accuracy on face 2.
        face_1_idxes = get_face_1_idxes(p)
        f_interp_face_1 = f_interp[face_1_idxes]
        f_cheby_face_1 = f_cheby_evals[face_1_idxes]
        print("test_1: f_interp_face_1: ", f_interp_face_1)
        print("test_1: f_cheby_face_1: ", f_cheby_face_1)
        print("test_1: diffs: ", f_interp_face_1 - f_cheby_face_1)
        assert jnp.allclose(f_interp_face_1, f_cheby_face_1)


class Test_precompute_Q_3D_DtN:
    def test_0(self) -> None:
        p = 2
        q = 3
        du_dx = jnp.ones((p**3, p**3))
        du_dy = jnp.ones((p**3, p**3))
        du_dz = jnp.ones((p**3, p**3))
        o = precompute_Q_3D_DtN(p, q, du_dx, du_dy, du_dz)
        expected_shape = (6 * q**2, p**3)
        assert o.shape == expected_shape

    def test_1(self) -> None:
        """Check normal derivative accuracy on f(x,y,z) = x + y + z."""

        def f(x) -> jnp.ndarray:
            return x[:, 0] + x[:, 1] + x[:, 2]

        p = 4
        q = 2
        half_side_len = jnp.pi / 2
        du_dx, du_dy, du_dz, _, _, _, _, _, _ = precompute_diff_operators_3D(
            p, half_side_len=half_side_len
        )
        mat = precompute_Q_3D_DtN(p, q, du_dx, du_dy, du_dz)

        # corners = (jnp.pi / 2) * jnp.array([[-1, -1, -1], [1, 1, 1]])
        bounds = (jnp.pi / 2) * jnp.array([-1, 1, -1, 1, -1, 1])
        cheby_pts = chebyshev_points(p)
        cheby_pts_lst = bounds_to_cheby_points_3D(bounds, cheby_pts)
        # gauss_pts_lst = corners_to_gauss_points_lst(q, corners)

        f_cheby_evals = f(cheby_pts_lst)
        df_dn_gauss_interp = mat @ f_cheby_evals
        df_dn_gauss_expected = jnp.concatenate(
            [
                -1 * jnp.ones(q**2),  # Face 1
                jnp.ones(q**2),  # Face 2
                -1 * jnp.ones(q**2),  # Face 3
                jnp.ones(q**2),  # Face 4
                -1 * jnp.ones(q**2),  # Face 5
                jnp.ones(q**2),  # Face 6
            ]
        )

        assert jnp.allclose(df_dn_gauss_interp, df_dn_gauss_expected)

    def test_2(self) -> None:
        """Check normal derivative accuracy on f(x,y,z) = x**2 + 3y - xz."""

        def f(x) -> jnp.ndarray:
            # f(x,y,z) = x**2 + 3y - xz
            return jnp.square(x[:, 0]) + 3 * x[:, 1] - x[:, 0] * x[:, 2]

        p = 6
        q = 4
        half_side_len = jnp.pi / 2
        du_dx, du_dy, du_dz, _, _, _, _, _, _ = precompute_diff_operators_3D(
            p, half_side_len=half_side_len
        )
        mat = precompute_Q_3D_DtN(p, q, du_dx, du_dy, du_dz)

        # corners = (jnp.pi / 2) * jnp.array([[-1, -1, -1], [1, 1, 1]])
        bounds = (jnp.pi / 2) * jnp.array([-1, 1, -1, 1, -1, 1])
        cheby_pts = chebyshev_points(p)
        cheby_pts_lst = bounds_to_cheby_points_3D(bounds, cheby_pts)

        root = DiscretizationNode3D(
            xmin=bounds[0],
            xmax=bounds[1],
            ymin=bounds[2],
            ymax=bounds[3],
            zmin=bounds[4],
            zmax=bounds[5],
        )
        gauss_pts_lst = compute_boundary_Gauss_points_adaptive_3D(root, q)

        f_cheby_evals = f(cheby_pts_lst)
        df_dn_gauss_interp = mat @ f_cheby_evals

        # Test the answer face-by-face

        # Face 1
        face_1_expected = -1 * (
            2 * gauss_pts_lst[: q**2, 0] - gauss_pts_lst[: q**2, 2]
        )
        face_1_interp = df_dn_gauss_interp[: q**2]
        print("test_2: face_1_interp: ", face_1_interp)
        print("test_2: face_1_expected: ", face_1_expected)
        print("test_2: diffs: ", face_1_interp - face_1_expected)
        assert jnp.allclose(face_1_interp, face_1_expected)

        # Face 2
        face_2_expected = (
            2 * gauss_pts_lst[q**2 : 2 * q**2, 0] - gauss_pts_lst[: q**2, 2]
        )
        face_2_interp = df_dn_gauss_interp[q**2 : 2 * q**2]
        assert jnp.allclose(face_2_interp, face_2_expected)

        # Face 3
        face_3_expected = -1 * 3 * jnp.ones(q**2)
        face_3_interp = df_dn_gauss_interp[2 * q**2 : 3 * q**2]
        assert jnp.allclose(face_3_interp, face_3_expected)

        # Face 4
        face_4_expected = 3 * jnp.ones(q**2)
        face_4_interp = df_dn_gauss_interp[3 * q**2 : 4 * q**2]
        assert jnp.allclose(face_4_interp, face_4_expected)

        # Face 5
        face_5_expected = gauss_pts_lst[4 * q**2 : 5 * q**2, 0]
        face_5_interp = df_dn_gauss_interp[4 * q**2 : 5 * q**2]
        assert jnp.allclose(face_5_interp, face_5_expected)

        # Face 6
        face_6_expected = -1 * gauss_pts_lst[5 * q**2 :, 0]
        face_6_interp = df_dn_gauss_interp[5 * q**2 :]
        assert jnp.allclose(face_6_interp, face_6_expected)
        # df_dn_gauss_expected = jnp.concatenate(
        #     [
        #         -1
        #         * (
        #             2 * gauss_pts_lst[: q**2, 0] + gauss_pts_lst[: q**2, 2]
        #         ),  # Face 1 -dudx
        #         2 * gauss_pts_lst[: q**2, 0] + gauss_pts_lst[: q**2, 2],  # Face 2 dudx
        #         -1 * 3 * jnp.ones(q**2),  # Face 3 -dudy
        #         3 * jnp.ones(q**2),  # Face 4 dudy
        #         -1 * gauss_pts_lst[4 * q**2 : 5 * q**2, 0],  # Face 5 -dudz
        #         gauss_pts_lst[5 * q**2 :, 0],  # Face 6 dudz
        #     ]
        # )

        # assert jnp.allclose(df_dn_gauss_interp, df_dn_gauss_expected)


class Test_precompute_refining_coarsening_op:
    def test_0(self) -> None:
        q = 4

        L_2f1, L_1f2 = precompute_projection_ops_3D(q)

        assert L_2f1.shape == (4 * q**2, q**2)
        assert L_1f2.shape == (q**2, 4 * q**2)

        assert not jnp.any(jnp.isnan(L_2f1))
        assert not jnp.any(jnp.isnan(L_1f2))

        assert not jnp.any(jnp.isinf(L_2f1))
        assert not jnp.any(jnp.isinf(L_1f2))

    def test_1(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation on face 5."""
        q = 4

        L_2f1, L_1f2 = precompute_projection_ops_3D(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )

        root_1 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (x,y) plane farthest in the +z direction.
        root_0_pts = compute_boundary_Gauss_points_adaptive_3D(root_0, q)[
            4 * n_per_face_0 : 5 * n_per_face_0
        ]
        root_1_pts = compute_boundary_Gauss_points_adaptive_3D(root_1, q)[
            4 * n_per_face_1 : 5 * n_per_face_1
        ]

        # Check to make sure all of the points have x_vals equal to 0.0
        assert jnp.allclose(root_0_pts[:, 2], 0.0)
        assert jnp.allclose(root_1_pts[:, 2], 0.0)

        # Define a low-degree polynomial function
        def f(x: jnp.array) -> jnp.ndarray:
            # f(x,y,z) = 3 y + y^2 - 4x
            return 3 * x[:, 1] + x[:, 1] ** 2 - 4 * x[:, 0]

        root_0_evals = f(root_0_pts)
        root_1_evals = f(root_1_pts)

        root_1_interp = L_2f1 @ root_0_evals

        assert jnp.allclose(root_1_interp, root_1_evals)

        root_0_interp = L_1f2 @ root_1_evals

        print("test_1: root_0_interp: ", root_0_interp)
        print("test_1: root_0_evals: ", root_0_evals)
        print("test_1: diffs: ", root_0_interp - root_0_evals)

        assert jnp.allclose(root_0_interp, root_0_evals)

    def test_2(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation on face 4."""
        q = 4

        L_2f1, L_1f2 = precompute_projection_ops_3D(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )

        root_1 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (x,y) plane farthest in the +z direction.
        root_0_pts = compute_boundary_Gauss_points_adaptive_3D(root_0, q)[
            5 * n_per_face_0 : 6 * n_per_face_0
        ]
        root_1_pts = compute_boundary_Gauss_points_adaptive_3D(root_1, q)[
            5 * n_per_face_1 : 6 * n_per_face_1
        ]

        # Check to make sure all of the points have x_vals equal to max_val
        assert jnp.allclose(root_0_pts[:, 2], max_val)
        assert jnp.allclose(root_1_pts[:, 2], max_val)

        # Define a low-degree polynomial function
        def f(x: jnp.array) -> jnp.ndarray:
            # f(x,y,z) = 3 y + y^2 - 4x
            return 3 * x[:, 1] + x[:, 1] ** 2 - 4 * x[:, 0]

        root_0_evals = f(root_0_pts)
        root_1_evals = f(root_1_pts)

        root_1_interp = L_2f1 @ root_0_evals

        assert jnp.allclose(root_1_interp, root_1_evals)

        root_0_interp = L_1f2 @ root_1_evals

        print("test_1: root_0_interp: ", root_0_interp)
        print("test_1: root_0_evals: ", root_0_evals)
        print("test_1: diffs: ", root_0_interp - root_0_evals)

        assert jnp.allclose(root_0_interp, root_0_evals)

    def test_3(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation in the face 3"""
        q = 4

        L_2f1, L_1f2 = precompute_projection_ops_3D(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )

        root_1 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (x,z) plane farthest in the +y direction.
        root_0_pts = compute_boundary_Gauss_points_adaptive_3D(root_0, q)[
            3 * n_per_face_0 : 4 * n_per_face_0
        ]
        root_1_pts = compute_boundary_Gauss_points_adaptive_3D(root_1, q)[
            3 * n_per_face_1 : 4 * n_per_face_1
        ]

        # Check to make sure all of the points have y_vals equal to 1.0
        assert jnp.allclose(root_0_pts[:, 1], max_val)
        assert jnp.allclose(root_1_pts[:, 1], max_val)

        # Define a low-degree polynomial function
        def f(x: jnp.array) -> jnp.ndarray:
            # f(x,y,z) = 3 z + z^2 - 4x
            return 3 * x[:, 2] + x[:, 2] ** 2 - 4 * x[:, 0]

        root_0_evals = f(root_0_pts)
        root_1_evals = f(root_1_pts)

        root_1_interp = L_2f1 @ root_0_evals

        print("test_2: root_1_interp: ", root_1_interp)
        print("test_2: root_1_evals: ", root_1_evals)
        print("test_2: diffs: ", root_1_interp - root_1_evals)

        assert jnp.allclose(root_1_interp, root_1_evals), "Refining op failed"

        root_0_interp = L_1f2 @ root_1_evals

        print("test_2: root_0_interp: ", root_0_interp)
        print("test_2: root_0_evals: ", root_0_evals)
        print("test_2: diffs: ", root_0_interp - root_0_evals)

        assert jnp.allclose(root_0_interp, root_0_evals), (
            "Coarsening op failed"
        )

    def test_4(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation in the face 2"""
        q = 4

        L_2f1, L_1f2 = precompute_projection_ops_3D(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )

        root_1 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (x,z) plane farthest in the +y direction.
        root_0_pts = compute_boundary_Gauss_points_adaptive_3D(root_0, q)[
            2 * n_per_face_0 : 3 * n_per_face_0
        ]
        root_1_pts = compute_boundary_Gauss_points_adaptive_3D(root_1, q)[
            2 * n_per_face_1 : 3 * n_per_face_1
        ]

        # Check to make sure all of the points have y_vals equal to 0.0
        assert jnp.allclose(root_0_pts[:, 1], min_val)
        assert jnp.allclose(root_1_pts[:, 1], min_val)

        # Define a low-degree polynomial function
        def f(x: jnp.array) -> jnp.ndarray:
            # f(x,y,z) = 3 z + z^2 - 4x
            return 3 * x[:, 2] + x[:, 2] ** 2 - 4 * x[:, 0]

        root_0_evals = f(root_0_pts)
        root_1_evals = f(root_1_pts)

        root_1_interp = L_2f1 @ root_0_evals

        print("test_2: root_1_interp: ", root_1_interp)
        print("test_2: root_1_evals: ", root_1_evals)
        print("test_2: diffs: ", root_1_interp - root_1_evals)

        assert jnp.allclose(root_1_interp, root_1_evals), "Refining op failed"

        root_0_interp = L_1f2 @ root_1_evals

        print("test_2: root_0_interp: ", root_0_interp)
        print("test_2: root_0_evals: ", root_0_evals)
        print("test_2: diffs: ", root_0_interp - root_0_evals)

        assert jnp.allclose(root_0_interp, root_0_evals), (
            "Coarsening op failed"
        )

    def test_5(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation in face 1"""
        q = 4

        L_2f1, L_1f2 = precompute_projection_ops_3D(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )

        root_1 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (y,z) plane farthest in the +z direction.
        root_0_pts = compute_boundary_Gauss_points_adaptive_3D(root_0, q)[
            1 * n_per_face_0 : 2 * n_per_face_0
        ]
        root_1_pts = compute_boundary_Gauss_points_adaptive_3D(root_1, q)[
            1 * n_per_face_1 : 2 * n_per_face_1
        ]

        # Check to make sure all of the points have x_vals equal to 1.0
        assert jnp.allclose(root_0_pts[:, 0], max_val)
        assert jnp.allclose(root_1_pts[:, 0], max_val)

        # Define a low-degree polynomial function
        def f(x: jnp.array) -> jnp.ndarray:
            # f(x,y,z) = 3 z + z^2 - 4y
            return 3 * x[:, 2] + x[:, 2] ** 2 - 4 * x[:, 1]

        root_0_evals = f(root_0_pts)
        root_1_evals = f(root_1_pts)

        root_1_interp = L_2f1 @ root_0_evals

        assert jnp.allclose(root_1_interp, root_1_evals), "Refining op failed"

        root_0_interp = L_1f2 @ root_1_evals

        print("test_2: root_0_interp: ", root_0_interp)
        print("test_2: root_0_evals: ", root_0_evals)
        print("test_2: diffs: ", root_0_interp - root_0_evals)

        assert jnp.allclose(root_0_interp, root_0_evals), (
            "Coarsening op failed"
        )

    def test_6(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation in face 0"""
        q = 4

        L_2f1, L_1f2 = precompute_projection_ops_3D(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )

        root_1 = DiscretizationNode3D(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (y,z) plane farthest in the +z direction.
        root_0_pts = compute_boundary_Gauss_points_adaptive_3D(root_0, q)[
            :n_per_face_0
        ]
        root_1_pts = compute_boundary_Gauss_points_adaptive_3D(root_1, q)[
            :n_per_face_1
        ]

        # Check to make sure all of the points have x_vals equal to 0.0
        assert jnp.allclose(root_0_pts[:, 0], min_val)
        assert jnp.allclose(root_1_pts[:, 0], min_val)

        # Define a low-degree polynomial function
        def f(x: jnp.array) -> jnp.ndarray:
            # f(x,y,z) = 3 z + z^2 - 4y
            return 3 * x[:, 2] + x[:, 2] ** 2 - 4 * x[:, 1]

        root_0_evals = f(root_0_pts)
        root_1_evals = f(root_1_pts)

        root_1_interp = L_2f1 @ root_0_evals

        assert jnp.allclose(root_1_interp, root_1_evals), "Refining op failed"

        root_0_interp = L_1f2 @ root_1_evals

        print("test_2: root_0_interp: ", root_0_interp)
        print("test_2: root_0_evals: ", root_0_evals)
        print("test_2: diffs: ", root_0_interp - root_0_evals)

        assert jnp.allclose(root_0_interp, root_0_evals), (
            "Coarsening op failed"
        )


class Test_L_8f1:
    def test_0(self) -> None:
        # Test to make sure the shapes are right.
        p = 2
        x = precompute_L_8f1(p)

        print("test_0: x", x)
        assert x.shape == (8 * p**3, p**3)
        assert not jnp.any(jnp.isnan(x))

    def test_1(self) -> None:
        """Refinement should be exact on low-degree polynomials"""
        p = 5
        x = precompute_L_8f1(p)
        assert not jnp.any(jnp.isnan(x))
        root_0 = DiscretizationNode3D(
            xmin=-1.0,
            xmax=1.0,
            ymin=-1.0,
            ymax=1.0,
            zmin=-1.0,
            zmax=1.0,
        )
        root_1 = DiscretizationNode3D(
            xmin=-1.0,
            xmax=1.0,
            ymin=-1.0,
            ymax=1.0,
            zmin=-1.0,
            zmax=1.0,
        )
        add_eight_children(root_1, root=root_1, q=p - 2)

        # These are the Chebyshev points, ordered to be exterior points, then interior points.
        pts_0 = compute_interior_Chebyshev_points_adaptive_3D(
            root_0, p
        ).reshape((-1, 3))

        # These are four copies of the Cheby points
        pts_1 = compute_interior_Chebyshev_points_adaptive_3D(root_1, p)
        print("test_1: pts_1.shape: ", pts_1.shape)
        pts_1 = pts_1.reshape((-1, 3))

        def f(x: jnp.array) -> jnp.array:
            """f(x,y,z) = x^2 + 3y - xz"""
            return x[..., 0] ** 2 + 3 * x[..., 1] - x[..., 0] * x[..., 2]

        # def f(x: jnp.array) -> jnp.array:
        #     # f(x,y,z) = y
        #     return x[..., 1]

        f_0 = f(pts_0)
        f_1 = f(pts_1)
        f_interp = x @ f_0

        diffs = f_1 - f_interp

        # plt.plot(f_interp, label="f_interp")
        # plt.plot(f_1, label="f_1")
        # plt.plot(diffs, label="diffs")
        # plt.legend()
        # plt.grid()
        # plt.show()

        # print("test_1: x: ", x)

        print("test_1: f_0: ", f_0)
        print("test_1: f_1: ", f_1)
        print("test_1: f_interp: ", f_interp)
        print("test_1: diffs: ", diffs)

        assert jnp.allclose(f_1, f_interp)


class Test_indexing_for_refinement_operator:
    def test_0(self) -> None:
        p = 4
        row_idxes, col_idxes = indexing_for_refinement_operator(p)

        assert col_idxes.shape == (p**3,)
        assert row_idxes.shape == (8 * p**3,)

    def test_1(self) -> None:
        p = 4
        q = 2
        cheby_pts_1d = chebyshev_points(p)
        cheby_pts_refined = jnp.concatenate(
            [
                affine_transform(cheby_pts_1d, jnp.array([-1, 0])),
                affine_transform(cheby_pts_1d, jnp.array([0, 1])),
            ]
        )

        col_x, col_y, col_z = jnp.meshgrid(
            cheby_pts_refined,
            cheby_pts_refined,
            cheby_pts_refined,
            indexing="ij",
        )
        col_pts = jnp.stack([col_x, col_y, col_z], axis=-1).reshape(-1, 3)
        print("test_1: col_pts", col_pts)

        root = DiscretizationNode3D(
            xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, depth=0
        )
        add_eight_children(root, root=root, q=q)

        # These are eight copies of the Chebyshev points
        pts_1 = compute_interior_Chebyshev_points_adaptive_3D(root, p).reshape(
            (-1, 3)
        )

        row_idxes, col_idxes = indexing_for_refinement_operator(p)

        col_pts_rearranged = col_pts[row_idxes]
        print("test_1: col_pts_rearranged", col_pts_rearranged.shape)

        pts_1_trunc = pts_1[: col_pts_rearranged.shape[0]]
        print("test_1: pts_1", pts_1.shape)

        assert jnp.allclose(col_pts_rearranged, pts_1_trunc)

        assert jnp.allclose(col_pts_rearranged, pts_1)
