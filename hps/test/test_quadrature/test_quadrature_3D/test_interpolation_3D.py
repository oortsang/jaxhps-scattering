import pytest
import jax.numpy as jnp
import numpy as np
from hps.src.quadrature.quad_3D.interpolation import (
    precompute_P_matrix,
    precompute_Q_D_matrix,
    refinement_operator,
    interp_operator_to_uniform,
    precompute_refining_coarsening_ops,
)
from hps.src.quadrature.quad_3D.indexing import (
    get_face_1_idxes,
    get_face_2_idxes,
    get_face_3_idxes,
    get_face_4_idxes,
    get_face_5_idxes,
    get_face_6_idxes,
)
from hps.src.quadrature.quad_3D.differentiation import precompute_diff_operators
from hps.src.quadrature.quadrature_utils import (
    chebyshev_points,
    barycentric_lagrange_2d_interpolation_matrix,
)
from hps.src.quadrature.quad_3D.grid_creation import (
    corners_to_cheby_points_lst,
    corners_to_gauss_points_lst,
    get_all_leaf_3d_cheby_points,
    get_all_boundary_gauss_legendre_points,
)
from hps.src.quadrature.trees import Node, add_eight_children
from hps.src.utils import meshgrid_to_lst_of_pts
import matplotlib.pyplot as plt


class Test_interp_operator_to_uniform:
    def test_0(self) -> None:
        """Make sure things work OK on non-uniform grid."""
        p = 8

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0)
        add_eight_children(root)
        add_eight_children(root.children[0])

        leaves = get_all_leaf_3d_cheby_points(p, root)
        n_pts = leaves.shape[0] * leaves.shape[1]

        to_x = np.linspace(root.xmin, root.xmax, p, endpoint=False)
        to_y = np.linspace(root.ymin, root.ymax, p, endpoint=False)
        to_z = np.linspace(root.zmin, root.zmax, p, endpoint=False)

        x = interp_operator_to_uniform(root, leaves, to_x, to_y, to_z, p)

        assert x.shape == (p**3, n_pts)

    def test_1(self) -> None:
        """One node and one output point"""
        p = 4

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0)

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = x"""
            return x[..., 0] ** 2 - x[..., 1] ** 2

        leaves = get_all_leaf_3d_cheby_points(p, root)

        to_x = np.array([0.5])
        to_y = np.array([0.5])
        to_z = np.array([0.5])
        X, Y, Z = np.meshgrid(to_x, to_y, to_z, indexing="ij")
        to_pts = jnp.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)

        A = interp_operator_to_uniform(root, leaves, to_x, to_y, to_z, p)
        f_vals = f(to_pts)
        f_vals_nonuniform = f(leaves.reshape((-1, 3)))

        f_interp = A @ f_vals_nonuniform

        assert f_interp.shape == f_vals.shape
        assert jnp.allclose(f_interp, f_vals)

    def test_2(self) -> None:
        """Multiple nodes and multiple output points"""
        p = 4

        root = Node(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0, zmin=0.0, zmax=1.0)
        add_eight_children(root)
        add_eight_children(root.children[0])

        def f(x: jnp.array) -> jnp.array:
            """f(x,y,z) = x^2 - y^2 + 2z^2"""
            return x[..., 1] ** 2 - x[..., 0] ** 2 + 2 * x[..., 2] ** 2

        leaves = get_all_leaf_3d_cheby_points(p, root)

        to_x = np.linspace(root.xmin, root.xmax, p, endpoint=False)
        to_y = np.linspace(root.ymin, root.ymax, p, endpoint=False)
        to_z = np.linspace(root.zmin, root.zmax, p, endpoint=False)
        X, Y, Z = jnp.meshgrid(to_x, to_y, to_z, indexing="ij")
        to_pts = jnp.stack((X.flatten(), Y.flatten(), Z.flatten()), axis=-1)
        print("test_2: to_pts = ", to_pts)

        A = interp_operator_to_uniform(root, leaves, to_x, to_y, to_z, p)
        f_vals = f(to_pts)
        f_vals_nonuniform_a = f(leaves.reshape((-1, 3)))
        print("test_2: f_vals_nonuniform_a = ", f_vals_nonuniform_a)

        f_interp = A @ f_vals_nonuniform_a

        assert f_interp.shape == f_vals.shape

        # plt.plot(f_vals, label="f_vals")
        # plt.plot(f_interp, label="f_interp")
        # plt.legend()
        # plt.show()

        assert jnp.allclose(f_interp, f_vals)


class Test_refinement_operator:
    def test_0(self) -> None:
        # Test to make sure the shapes are right.
        p = 2
        x = refinement_operator(p)

        print("test_0: x", x)
        assert x.shape == (8 * p**3, p**3)
        assert not jnp.any(jnp.isnan(x))

    def test_1(self) -> None:
        """Refinement should be exact on low-degree polynomials"""
        p = 5
        x = refinement_operator(p)
        assert not jnp.any(jnp.isnan(x))
        corners = jnp.array([[-1, -1, -1], [1, 1, 1]], dtype=jnp.float64)
        root_0 = Node(
            xmin=-1.0,
            xmax=1.0,
            ymin=-1.0,
            ymax=1.0,
            zmin=-1.0,
            zmax=1.0,
            depth=0,
        )
        root_1 = Node(
            xmin=-1.0,
            xmax=1.0,
            ymin=-1.0,
            ymax=1.0,
            zmin=-1.0,
            zmax=1.0,
            depth=0,
        )
        add_eight_children(root_1, root=root_1, q=p - 2)

        # These are the Chebyshev points, ordered to be exterior points, then interior points.
        pts_0 = get_all_leaf_3d_cheby_points(p, root_0).reshape((-1, 3))

        # These are four copies of the Cheby points
        pts_1 = get_all_leaf_3d_cheby_points(p, root_1)
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


class Test_precompute_P_matrix:
    def test_0(self) -> None:
        # Test case 1
        p = 2
        q = 3
        o = precompute_P_matrix(p, q)
        expected_shape = (p**3 - (p - 2) ** 3, 6 * q**2)
        assert o.shape == expected_shape

    def test_1(self) -> None:
        """Tests accuracy of interpolation on f(x,y,z) = x + y + z."""

        def f(x) -> jnp.ndarray:
            return x[:, 0] + x[:, 1] + x[:, 2]

        p = 4
        q = 2
        mat = precompute_P_matrix(p, q)

        corners = jnp.array(
            [
                [-1, -1, -1],
                [1, 1, 1],
            ]
        )
        n_cheby_bdry = p**3 - (p - 2) ** 3
        cheby_pts = chebyshev_points(p)[0]
        cheby_pts_lst = corners_to_cheby_points_lst(corners, cheby_pts)[:n_cheby_bdry]
        gauss_pts_lst = corners_to_gauss_points_lst(q, corners)

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


class Test_precompute_Q_D_matrix:
    def test_0(self) -> None:
        p = 2
        q = 3
        du_dx = jnp.ones((p**3, p**3))
        du_dy = jnp.ones((p**3, p**3))
        du_dz = jnp.ones((p**3, p**3))
        o = precompute_Q_D_matrix(p, q, du_dx, du_dy, du_dz)
        expected_shape = (6 * q**2, p**3)
        assert o.shape == expected_shape

    def test_1(self) -> None:
        """Check normal derivative accuracy on f(x,y,z) = x + y + z."""

        def f(x) -> jnp.ndarray:
            return x[:, 0] + x[:, 1] + x[:, 2]

        p = 4
        q = 2
        half_side_len = jnp.pi / 2
        du_dx, du_dy, du_dz, _, _, _, _, _, _ = precompute_diff_operators(
            p, half_side_len=half_side_len
        )
        mat = precompute_Q_D_matrix(p, q, du_dx, du_dy, du_dz)

        corners = (jnp.pi / 2) * jnp.array([[-1, -1, -1], [1, 1, 1]])
        cheby_pts = chebyshev_points(p)[0]
        cheby_pts_lst = corners_to_cheby_points_lst(corners, cheby_pts)
        gauss_pts_lst = corners_to_gauss_points_lst(q, corners)

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
        du_dx, du_dy, du_dz, _, _, _, _, _, _ = precompute_diff_operators(
            p, half_side_len=half_side_len
        )
        mat = precompute_Q_D_matrix(p, q, du_dx, du_dy, du_dz)

        corners = (jnp.pi / 2) * jnp.array([[-1, -1, -1], [1, 1, 1]])
        cheby_pts = chebyshev_points(p)[0]
        cheby_pts_lst = corners_to_cheby_points_lst(corners, cheby_pts)
        gauss_pts_lst = corners_to_gauss_points_lst(q, corners)

        f_cheby_evals = f(cheby_pts_lst)
        df_dn_gauss_interp = mat @ f_cheby_evals

        # Test the answer face-by-face

        # Face 1
        face_1_expected = -1 * (2 * gauss_pts_lst[: q**2, 0] - gauss_pts_lst[: q**2, 2])
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

        L_2f1, L_1f2 = precompute_refining_coarsening_ops(q)

        assert L_2f1.shape == (4 * q**2, q**2)
        assert L_1f2.shape == (q**2, 4 * q**2)

        assert not jnp.any(jnp.isnan(L_2f1))
        assert not jnp.any(jnp.isnan(L_1f2))

        assert not jnp.any(jnp.isinf(L_2f1))
        assert not jnp.any(jnp.isinf(L_1f2))

    def test_1(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation on face 5."""
        q = 4

        L_2f1, L_1f2 = precompute_refining_coarsening_ops(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )

        root_1 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (x,y) plane farthest in the +z direction.
        root_0_pts = get_all_boundary_gauss_legendre_points(q, root_0)[
            4 * n_per_face_0 : 5 * n_per_face_0
        ]
        root_1_pts = get_all_boundary_gauss_legendre_points(q, root_1)[
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

        L_2f1, L_1f2 = precompute_refining_coarsening_ops(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )

        root_1 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (x,y) plane farthest in the +z direction.
        root_0_pts = get_all_boundary_gauss_legendre_points(q, root_0)[
            5 * n_per_face_0 : 6 * n_per_face_0
        ]
        root_1_pts = get_all_boundary_gauss_legendre_points(q, root_1)[
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

        L_2f1, L_1f2 = precompute_refining_coarsening_ops(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )

        root_1 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (x,z) plane farthest in the +y direction.
        root_0_pts = get_all_boundary_gauss_legendre_points(q, root_0)[
            3 * n_per_face_0 : 4 * n_per_face_0
        ]
        root_1_pts = get_all_boundary_gauss_legendre_points(q, root_1)[
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

        assert jnp.allclose(root_0_interp, root_0_evals), "Coarsening op failed"

    def test_4(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation in the face 2"""
        q = 4

        L_2f1, L_1f2 = precompute_refining_coarsening_ops(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )

        root_1 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (x,z) plane farthest in the +y direction.
        root_0_pts = get_all_boundary_gauss_legendre_points(q, root_0)[
            2 * n_per_face_0 : 3 * n_per_face_0
        ]
        root_1_pts = get_all_boundary_gauss_legendre_points(q, root_1)[
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

        assert jnp.allclose(root_0_interp, root_0_evals), "Coarsening op failed"

    def test_5(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation in face 1"""
        q = 4

        L_2f1, L_1f2 = precompute_refining_coarsening_ops(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )

        root_1 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (y,z) plane farthest in the +z direction.
        root_0_pts = get_all_boundary_gauss_legendre_points(q, root_0)[
            1 * n_per_face_0 : 2 * n_per_face_0
        ]
        root_1_pts = get_all_boundary_gauss_legendre_points(q, root_1)[
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

        assert jnp.allclose(root_0_interp, root_0_evals), "Coarsening op failed"

    def test_6(self) -> None:
        """Make sure things are accurate with low-degree polynomial interpolation in face 0"""
        q = 4

        L_2f1, L_1f2 = precompute_refining_coarsening_ops(q)

        max_val = 1.0
        min_val = 0.0

        root_0 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )

        root_1 = Node(
            xmin=min_val,
            xmax=max_val,
            ymin=min_val,
            ymax=max_val,
            zmin=min_val,
            zmax=max_val,
            depth=0,
        )
        add_eight_children(root_1)

        n_per_face_0 = q**2
        n_per_face_1 = 4 * q**2

        # Get the boundary points for the first face, which is the face in the (y,z) plane farthest in the +z direction.
        root_0_pts = get_all_boundary_gauss_legendre_points(q, root_0)[:n_per_face_0]
        root_1_pts = get_all_boundary_gauss_legendre_points(q, root_1)[:n_per_face_1]

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

        assert jnp.allclose(root_0_interp, root_0_evals), "Coarsening op failed"


if __name__ == "__main__":
    pytest.main()
