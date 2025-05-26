import numpy as np
import jax.numpy as jnp
import jax
import logging
from jaxhps.merge._utils_adaptive_3D_DtN import (
    get_a_submatrices,
    get_b_submatrices,
    get_c_submatrices,
    get_d_submatrices,
    get_e_submatrices,
    get_f_submatrices,
    get_g_submatrices,
    get_h_submatrices,
    get_rearrange_indices,
)
from jaxhps._grid_creation_3D import (
    compute_boundary_Gauss_points_adaptive_3D,
    bounds_for_oct_subdivision,
)
from jaxhps._precompute_operators_3D import precompute_projection_ops_3D
from jaxhps._discretization_tree import (
    DiscretizationNode3D,
)
from jaxhps._discretization_tree_operations_3D import add_eight_children


class Test_get_a_submatrices:
    def test_0(self) -> None:
        # Make sure the function returns correct shapes. No interpolation

        q = 4
        n_per_side = q**2
        n_gauss_bdry = 6 * n_per_side

        T = np.random.normal(size=(n_gauss_bdry, n_gauss_bdry))
        v = np.random.normal(size=(n_gauss_bdry))
        L_2f1 = np.random.normal(size=(2 * n_per_side, n_per_side))
        L_1f2 = np.random.normal(size=(n_per_side, 2 * n_per_side))
        need_interp_9 = jnp.array([False])
        need_interp_12 = jnp.array([False])
        need_interp_17 = jnp.array([False])
        expected_n_points_1 = 3 * n_per_side
        expected_n_points_9 = n_per_side
        expected_n_points_12 = n_per_side
        expected_n_points_17 = n_per_side
        ee = [
            expected_n_points_1,
            expected_n_points_9,
            expected_n_points_12,
            expected_n_points_17,
        ]
        return_tuple = get_a_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_9,
            need_interp_12,
            need_interp_17,
            n_0=n_per_side,
            n_1=n_per_side,
            n_2=n_per_side,
            n_3=n_per_side,
            n_4=n_per_side,
            n_5=n_per_side,
        )
        return_mats = return_tuple[:16]
        return_vecs = return_tuple[16:]

        for i, mat in enumerate(return_mats):
            expected_n_cols = ee[i % 4]
            expected_n_rows = ee[i // 4]
            assert mat.shape == (expected_n_rows, expected_n_cols), mat.shape

        for i, vec in enumerate(return_vecs):
            expected_n_rows = ee[i]
            assert vec.shape == (expected_n_rows,), vec.shape

    def test_1(self) -> None:
        """Tests that quad_pts are correctly assigned to the correct faces of the cube.
        No interpolation.
        """

        q = 5
        n_per_face = q**2

        xmin = -1
        ymin = -1
        zmin = -1
        xmax = 1
        ymax = 1
        zmax = 1
        root = DiscretizationNode3D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
        )
        quad_pts = compute_boundary_Gauss_points_adaptive_3D(root, q)

        need_interp_9 = jnp.array([False])
        need_interp_12 = jnp.array([False])
        need_interp_17 = jnp.array([False])

        T = np.random.normal(size=(6 * q**2, 6 * q**2))
        L_2f1 = np.random.normal(size=(2 * n_per_face, n_per_face))
        L_1f2 = np.random.normal(size=(n_per_face, 2 * n_per_face))
        # First test x direction
        v = quad_pts[:, 0]
        return_tuple = get_a_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_9,
            need_interp_12,
            need_interp_17,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_a_1 = return_tuple[16]
        assert jnp.all(v_a_1[:n_per_face] == xmin)
        assert jnp.all(v_a_1[n_per_face:] != xmin)

        # Second test y direction
        v = quad_pts[:, 1]
        return_tuple = get_a_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_9,
            need_interp_12,
            need_interp_17,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_a_1 = return_tuple[16]
        assert jnp.all(v_a_1[n_per_face : 2 * n_per_face] == ymin)
        assert jnp.all(v_a_1[:n_per_face] != ymin)
        assert jnp.all(v_a_1[2 * n_per_face :] != ymin)

        # Third test z direction
        v = quad_pts[:, 2]
        return_tuple = get_a_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_9,
            need_interp_12,
            need_interp_17,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_a_1 = return_tuple[16]
        assert jnp.all(v_a_1[2 * n_per_face : 3 * n_per_face] == zmax)
        assert jnp.all(v_a_1[: 2 * n_per_face] != zmax)

    def test_2(self) -> None:
        """
        Re-run test_1 with interpolation this time.
        """

        q = 4
        n_per_face = q**2
        n_per_face_refined = 4 * n_per_face

        xmin = -1
        ymin = -1
        zmin = -1
        xmax = 1
        ymax = 1
        zmax = 1
        root = DiscretizationNode3D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
        )

        add_eight_children(root)
        quad_pts = compute_boundary_Gauss_points_adaptive_3D(root, q)

        # root_0 = Node(xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, depth=0)
        # quad_pts_0 = get_all_boundary_gauss_legendre_points(q, root_0)

        need_interp_9 = jnp.array([True, True, True, True])
        need_interp_12 = jnp.array([False])
        need_interp_17 = jnp.array([False])

        T = np.random.normal(
            size=(6 * n_per_face_refined, 6 * n_per_face_refined)
        )
        L_2f1, L_1f2 = precompute_projection_ops_3D(q)
        print("test_2: L_2f1 shape: ", L_2f1.shape)
        print("test_2: L_1f2 shape: ", L_1f2.shape)
        # First test x direction
        v = quad_pts[:, 0]
        return_tuple = get_a_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_9,
            need_interp_12,
            need_interp_17,
            n_0=n_per_face_refined,
            n_1=n_per_face_refined,
            n_2=n_per_face_refined,
            n_3=n_per_face_refined,
            n_4=n_per_face_refined,
            n_5=n_per_face_refined,
        )
        v_a_1 = return_tuple[16]
        assert jnp.all(v_a_1[:n_per_face_refined] == xmin)
        assert jnp.all(v_a_1[n_per_face_refined:] != xmin)

        # Second test y direction
        v = quad_pts[:, 1]
        return_tuple = get_a_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_9,
            need_interp_12,
            need_interp_17,
            n_0=n_per_face_refined,
            n_1=n_per_face_refined,
            n_2=n_per_face_refined,
            n_3=n_per_face_refined,
            n_4=n_per_face_refined,
            n_5=n_per_face_refined,
        )
        v_a_1 = return_tuple[16]
        print("test_2: v_a_1 = ", v_a_1)
        assert jnp.all(v_a_1[:n_per_face_refined] != ymin)
        assert jnp.all(
            v_a_1[n_per_face_refined : 2 * n_per_face_refined] == ymin
        )
        assert jnp.all(v_a_1[2 * n_per_face_refined :] != ymin)

        # Third test z direction
        v = quad_pts[:, 2]
        return_tuple = get_a_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_9,
            need_interp_12,
            need_interp_17,
            n_0=n_per_face_refined,
            n_1=n_per_face_refined,
            n_2=n_per_face_refined,
            n_3=n_per_face_refined,
            n_4=n_per_face_refined,
            n_5=n_per_face_refined,
        )
        v_a_1 = return_tuple[16]
        assert jnp.all(v_a_1[: 2 * n_per_face_refined] != zmax)
        assert jnp.all(v_a_1[2 * n_per_face_refined :] == zmax)
        jax.clear_caches()


class Test_get_c_submatrices:
    def test_0(self) -> None:
        q = 5
        n_per_face = q**2

        xmin = -1
        ymin = -1
        zmin = -1
        xmax = 1
        ymax = 1
        zmax = 1
        root = DiscretizationNode3D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
        )
        quad_pts = compute_boundary_Gauss_points_adaptive_3D(root, q)

        L_1f2 = np.random.normal(size=(n_per_face, 4 * n_per_face))
        L_2f1 = np.random.normal(size=(4 * n_per_face, n_per_face))

        need_interp_10 = jnp.array([False])
        need_interp_11 = jnp.array([False])
        need_interp_19 = jnp.array([False])

        T = np.random.normal(size=(6 * q**2, 6 * q**2))
        # First test x direction
        v = quad_pts[:, 0]
        return_tuple = get_c_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_10,
            need_interp_11,
            need_interp_19,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_c_3 = return_tuple[16]
        assert jnp.all(v_c_3[:n_per_face] == xmax)
        assert jnp.all(v_c_3[n_per_face:] != xmax)

        # Second test y direction
        v = quad_pts[:, 1]
        return_tuple = get_c_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_10,
            need_interp_11,
            need_interp_19,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_c_3 = return_tuple[16]
        assert jnp.all(v_c_3[n_per_face : 2 * n_per_face] == ymax)
        assert jnp.all(v_c_3[:n_per_face] != ymax)
        assert jnp.all(v_c_3[2 * n_per_face :] != ymax)

        # Third test z direction
        v = quad_pts[:, 2]
        return_tuple = get_c_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_10,
            need_interp_11,
            need_interp_19,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_c_3 = return_tuple[16]
        assert jnp.all(v_c_3[2 * n_per_face : 3 * n_per_face] == zmax)
        assert jnp.all(v_c_3[: 2 * n_per_face] != zmax)
        jax.clear_caches()


class Test_get_d_submatrices:
    def test_0(self) -> None:
        q = 5
        n_per_face = q**2

        xmin = -1
        ymin = -1
        zmin = -1
        xmax = 1
        ymax = 1
        zmax = 1
        root = DiscretizationNode3D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
        )
        quad_pts = compute_boundary_Gauss_points_adaptive_3D(root, q)
        L_1f2 = np.random.normal(size=(n_per_face, 4 * n_per_face))
        L_2f1 = np.random.normal(size=(4 * n_per_face, n_per_face))
        T = np.random.normal(size=(6 * q**2, 6 * q**2))

        need_interp_11 = jnp.array([False])
        need_interp_12 = jnp.array([False])
        need_interp_20 = jnp.array([False])
        # First test x direction
        v = quad_pts[:, 0]
        return_tuple = get_d_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_11,
            need_interp_12,
            need_interp_20,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_d_4 = return_tuple[16]
        assert jnp.all(v_d_4[:n_per_face] == xmin)
        assert jnp.all(v_d_4[n_per_face:] != xmin)

        # Second test y direction
        v = quad_pts[:, 1]
        return_tuple = get_d_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_11,
            need_interp_12,
            need_interp_20,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_d_4 = return_tuple[16]
        assert jnp.all(v_d_4[n_per_face : 2 * n_per_face] == ymax)
        assert jnp.all(v_d_4[:n_per_face] != ymax)
        assert jnp.all(v_d_4[2 * n_per_face :] != ymax)

        # Third test z direction
        v = quad_pts[:, 2]
        return_tuple = get_d_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_11,
            need_interp_12,
            need_interp_20,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_d_4 = return_tuple[16]
        assert jnp.all(v_d_4[2 * n_per_face : 3 * n_per_face] == zmax)
        assert jnp.all(v_d_4[: 2 * n_per_face] != zmax)
        jax.clear_caches()


class Test_get_h_submatrices:
    def test_0(self) -> None:
        q = 5
        n_per_face = q**2

        xmin = -1
        ymin = -1
        zmin = -1
        xmax = 1
        ymax = 1
        zmax = 1
        root = DiscretizationNode3D(
            xmin=xmin,
            xmax=xmax,
            ymin=ymin,
            ymax=ymax,
            zmin=zmin,
            zmax=zmax,
        )
        quad_pts = compute_boundary_Gauss_points_adaptive_3D(root, q)
        L_1f2 = np.random.normal(size=(n_per_face, 4 * n_per_face))
        L_2f1 = np.random.normal(size=(4 * n_per_face, n_per_face))
        need_interp_15 = jnp.array([False])
        need_interp_16 = jnp.array([False])
        need_interp_20 = jnp.array([False])
        T = np.random.normal(size=(6 * q**2, 6 * q**2))
        # First test x direction
        v = quad_pts[:, 0]
        return_tuple = get_h_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_15,
            need_interp_16,
            need_interp_20,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_h_8 = return_tuple[16]
        assert jnp.all(v_h_8[:n_per_face] == xmin)
        assert jnp.all(v_h_8[n_per_face:] != xmin)

        # Second test y direction
        v = quad_pts[:, 1]
        return_tuple = get_h_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_15,
            need_interp_16,
            need_interp_20,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_h_8 = return_tuple[16]
        assert jnp.all(v_h_8[n_per_face : 2 * n_per_face] == ymax)
        assert jnp.all(v_h_8[:n_per_face] != ymax)
        assert jnp.all(v_h_8[2 * n_per_face :] != ymax)

        # Third test z direction
        v = quad_pts[:, 2]
        return_tuple = get_h_submatrices(
            T,
            v,
            L_2f1,
            L_1f2,
            need_interp_15,
            need_interp_16,
            need_interp_20,
            n_0=n_per_face,
            n_1=n_per_face,
            n_2=n_per_face,
            n_3=n_per_face,
            n_4=n_per_face,
            n_5=n_per_face,
        )
        v_h_8 = return_tuple[16]
        assert jnp.all(v_h_8[2 * n_per_face : 3 * n_per_face] == zmin)
        assert jnp.all(v_h_8[: 2 * n_per_face] != zmin)
        jax.clear_caches()


class Test_get_rearrange_indices:
    def test_0(self) -> None:
        q = 4
        q_squared = q**2
        idxes = np.arange(24 * q**2)
        out = get_rearrange_indices(
            idxes,
            n_a_0=q_squared,
            n_a_2=q_squared,
            n_a_5=q_squared,
            n_b_1=q_squared,
            n_b_2=q_squared,
            n_b_5=q_squared,
            n_c_1=q_squared,
            n_c_3=q_squared,
            n_c_5=q_squared,
            n_d_0=q_squared,
            n_d_3=q_squared,
            n_d_5=q_squared,
            n_e_0=q_squared,
            n_e_2=q_squared,
            n_e_4=q_squared,
            n_f_1=q_squared,
            n_f_2=q_squared,
            n_f_4=q_squared,
            n_g_1=q_squared,
            n_g_3=q_squared,
            n_g_4=q_squared,
            n_h_0=q_squared,
            n_h_3=q_squared,
            n_h_4=q_squared,
        )
        assert out.shape == (24 * q**2,)
        assert np.unique(out).shape == (24 * q**2,)

    def test_1(self, caplog) -> None:
        """This function tests that the get_rearrange_indices function successfully rearranges the indices in the uniform refinement case.
        Steps:
        1. Generate a quadrature of a cube with l=1 using get_all_boundary_gauss_legendre_points.
        2. Generate quadratures of all eight sub-cubes using bounds_for_oct_subdivision and corners_to_gauss_points_lst.
        3. For each sub-cube, make a vector of the (x, y, z) values of the quadrature points.
        4. Use these vectors and the get_*_submatrices functions to assemble a vector [v_1, ..., v_8].
        5. Use get_rearrange_indices to rearrange the vector.
        6. Check that the rearranged vector is in the correct order by matching it with the quadrature computed in step 1.
        """
        caplog.set_level(logging.DEBUG)
        # Setup:
        q = 4
        bounds = jnp.array([-1, 1, -1, 1, -1, 1])

        root = DiscretizationNode3D(
            xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, depth=0
        )
        add_eight_children(root, root, q=q)

        # Step 1:
        gauss_pts = compute_boundary_Gauss_points_adaptive_3D(root, q)

        # Step 2:
        root_1 = DiscretizationNode3D(
            xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1
        )
        # Manually add l=2 levels of refinement
        add_eight_children(root_1, root_1, q=q)
        for c in root_1.children:
            add_eight_children(c, root_1, q=q)
        oct_bounds = bounds_for_oct_subdivision(bounds)
        quad_pts_lst = []
        for b in oct_bounds:
            node_b = DiscretizationNode3D(
                xmin=b[0],
                xmax=b[1],
                ymin=b[2],
                ymax=b[3],
                zmin=b[4],
                zmax=b[5],
            )
            quad_pts = compute_boundary_Gauss_points_adaptive_3D(node_b, q)
            quad_pts_lst.append(quad_pts)

        # logging.debug(
        #     "test_1: quad_pts_lst = %s",
        #     quad_pts_lst.shape,
        # )
        # Step 3:
        # loop over x,y,z dimensions to check all of them
        for i in range(3):
            print("test_1: i = ", i)
            in_vecs = [q[:, i] for q in quad_pts_lst]

            # Step 4:
            L_1f2 = np.random.normal(size=(q**2, 4 * q**2))
            L_2f1 = np.random.normal(size=(4 * q**2, q**2))
            T = np.random.normal(size=(6 * q**2, 6 * q**2))
            need_interp_lst = jnp.array([False])
            return_tuple_a = get_a_submatrices(
                T,
                in_vecs[0],
                L_2f1,
                L_1f2,
                need_interp_lst,
                need_interp_lst,
                need_interp_lst,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_a = return_tuple_a[16]
            return_tuple_b = get_b_submatrices(
                T,
                in_vecs[1],
                L_2f1,
                L_1f2,
                need_interp_lst,
                need_interp_lst,
                need_interp_lst,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_b = return_tuple_b[16]
            return_tuple_c = get_c_submatrices(
                T,
                in_vecs[2],
                L_2f1,
                L_1f2,
                need_interp_lst,
                need_interp_lst,
                need_interp_lst,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_c = return_tuple_c[16]
            return_tuple_d = get_d_submatrices(
                T,
                in_vecs[3],
                L_2f1,
                L_1f2,
                need_interp_lst,
                need_interp_lst,
                need_interp_lst,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_d = return_tuple_d[16]
            return_tuple_e = get_e_submatrices(
                T,
                in_vecs[4],
                L_2f1,
                L_1f2,
                need_interp_lst,
                need_interp_lst,
                need_interp_lst,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_e = return_tuple_e[16]
            return_tuple_f = get_f_submatrices(
                T,
                in_vecs[5],
                L_2f1,
                L_1f2,
                need_interp_lst,
                need_interp_lst,
                need_interp_lst,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_f = return_tuple_f[16]
            return_tuple_g = get_g_submatrices(
                T,
                in_vecs[6],
                L_2f1,
                L_1f2,
                need_interp_lst,
                need_interp_lst,
                need_interp_lst,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_g = return_tuple_g[16]
            return_tuple_h = get_h_submatrices(
                T,
                in_vecs[7],
                L_2f1,
                L_1f2,
                need_interp_lst,
                need_interp_lst,
                need_interp_lst,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_h = return_tuple_h[16]

            v_all = jnp.concatenate([v_a, v_b, v_c, v_d, v_e, v_f, v_g, v_h])

            # Step 5:
            idxes = np.arange(v_all.shape[0])
            q_squared = q**2
            rearranged_idxes = get_rearrange_indices(
                idxes,
                n_a_0=q_squared,
                n_a_2=q_squared,
                n_a_5=q_squared,
                n_b_1=q_squared,
                n_b_2=q_squared,
                n_b_5=q_squared,
                n_c_1=q_squared,
                n_c_3=q_squared,
                n_c_5=q_squared,
                n_d_0=q_squared,
                n_d_3=q_squared,
                n_d_5=q_squared,
                n_e_0=q_squared,
                n_e_2=q_squared,
                n_e_4=q_squared,
                n_f_1=q_squared,
                n_f_2=q_squared,
                n_f_4=q_squared,
                n_g_1=q_squared,
                n_g_3=q_squared,
                n_g_4=q_squared,
                n_h_0=q_squared,
                n_h_3=q_squared,
                n_h_4=q_squared,
            )
            v_all_rearranged = v_all[rearranged_idxes]

            # Step 6:
            # Check each face in order
            n_per_face = idxes.shape[0] // 6
            for j in range(6):
                expected_face_j = gauss_pts[
                    j * n_per_face : (j + 1) * n_per_face, i
                ]
                computed_face_j = v_all_rearranged[
                    j * n_per_face : (j + 1) * n_per_face
                ]

                v = jnp.allclose(expected_face_j, computed_face_j)
                if not v:
                    print("####### test_1: face_j = ", j + 1, " v = ", v)
                    print("test_1: expected_face_j = ", expected_face_j)
                    print("test_1: computed_face_j = ", computed_face_j)
                    print("test_1: diffs: ", expected_face_j - computed_face_j)
                assert v
        # assert False

    def test_2(self) -> None:
        """This function tests that the get_rearrange_indices function successfully rearranges the
        indices in the non-uniform refinement case.
        Steps:
        1. Generate a quadrature of a cube with l=1 using get_all_boundary_gauss_legendre_points.
        2. Generate quadratures of all eight children using get_all_boundary_gauss_legendre_points.
        3. For each child, make a vector of the (x, y, z) values of the quadrature points.
        4. Use these vectors and the get_*_submatrices functions to assemble a vector [v_1, ..., v_8].
        5. Use get_rearrange_indices to rearrange the vector.
        6. Check that the rearranged vector is in the correct order by matching it with the quadrature computed in step 1.
        """
        # Setup:
        q = 4
        # corners = jnp.array([[-1, -1, -1], [1, 1, 1]])

        root = DiscretizationNode3D(
            xmin=-1, xmax=1, ymin=-1, ymax=1, zmin=-1, zmax=1, depth=0
        )
        add_eight_children(root, root=root, q=q)
        add_eight_children(root.children[0], root=root, q=q)

        # Step 1:
        gauss_pts = compute_boundary_Gauss_points_adaptive_3D(root, q)

        # Step 2:
        quad_pts_lst = []
        for child in root.children:
            quad_pts = compute_boundary_Gauss_points_adaptive_3D(child, q)
            quad_pts_lst.append(quad_pts)

        # Step 3:
        # loop over x,y,z dimensions to check all of them
        for i in range(3):
            print("################# test_1: Testing new dimension")
            print("################# test_1: i = ", i)
            in_vecs = [q[:, i] for q in quad_pts_lst]

            # Step 4:
            L_1f2 = np.random.normal(size=(q**2, 4 * q**2))
            L_2f1 = np.random.normal(size=(4 * q**2, q**2))
            T = np.random.normal(size=(6 * q**2, 6 * q**2))

            need_interps_a = jnp.array([True, True, True, True])
            need_interps_not_a = jnp.array([False])

            # a is larger than the other children so we need a different dummy DtN matrix
            T_a = np.random.normal(size=(4 * 6 * q**2, 4 * 6 * q**2))
            return_tuple_a = get_a_submatrices(
                T_a,
                in_vecs[0],
                L_2f1,
                L_1f2,
                need_interps_a,
                need_interps_a,
                need_interps_a,
                n_0=4 * q**2,
                n_1=4 * q**2,
                n_2=4 * q**2,
                n_3=4 * q**2,
                n_4=4 * q**2,
                n_5=4 * q**2,
            )
            v_a = return_tuple_a[16]
            return_tuple_b = get_b_submatrices(
                T,
                in_vecs[1],
                L_2f1,
                L_1f2,
                need_interps_not_a,
                need_interps_not_a,
                need_interps_not_a,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_b = return_tuple_b[16]
            return_tuple_c = get_c_submatrices(
                T,
                in_vecs[2],
                L_2f1,
                L_1f2,
                need_interps_not_a,
                need_interps_not_a,
                need_interps_not_a,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_c = return_tuple_c[16]
            return_tuple_d = get_d_submatrices(
                T,
                in_vecs[3],
                L_2f1,
                L_1f2,
                need_interps_not_a,
                need_interps_not_a,
                need_interps_not_a,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_d = return_tuple_d[16]
            return_tuple_e = get_e_submatrices(
                T,
                in_vecs[4],
                L_2f1,
                L_1f2,
                need_interps_not_a,
                need_interps_not_a,
                need_interps_not_a,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_e = return_tuple_e[16]
            return_tuple_f = get_f_submatrices(
                T,
                in_vecs[5],
                L_2f1,
                L_1f2,
                need_interps_not_a,
                need_interps_not_a,
                need_interps_not_a,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_f = return_tuple_f[16]
            return_tuple_g = get_g_submatrices(
                T,
                in_vecs[6],
                L_2f1,
                L_1f2,
                need_interps_not_a,
                need_interps_not_a,
                need_interps_not_a,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_g = return_tuple_g[16]
            return_tuple_h = get_h_submatrices(
                T,
                in_vecs[7],
                L_2f1,
                L_1f2,
                need_interps_not_a,
                need_interps_not_a,
                need_interps_not_a,
                n_0=q**2,
                n_1=q**2,
                n_2=q**2,
                n_3=q**2,
                n_4=q**2,
                n_5=q**2,
            )
            v_h = return_tuple_h[16]

            v_all = jnp.concatenate([v_a, v_b, v_c, v_d, v_e, v_f, v_g, v_h])

            # Step 5:
            idxes = np.arange(v_all.shape[0])
            rearranged_idxes = get_rearrange_indices(
                idxes,
                n_a_0=root.children[0].n_0,
                n_a_2=root.children[0].n_2,
                n_a_5=root.children[0].n_5,
                n_b_1=root.children[1].n_1,
                n_b_2=root.children[1].n_2,
                n_b_5=root.children[1].n_5,
                n_c_1=root.children[2].n_1,
                n_c_3=root.children[2].n_3,
                n_c_5=root.children[2].n_5,
                n_d_0=root.children[3].n_0,
                n_d_3=root.children[3].n_3,
                n_d_5=root.children[3].n_5,
                n_e_0=root.children[4].n_0,
                n_e_2=root.children[4].n_2,
                n_e_4=root.children[4].n_4,
                n_f_1=root.children[5].n_1,
                n_f_2=root.children[5].n_2,
                n_f_4=root.children[5].n_4,
                n_g_1=root.children[6].n_1,
                n_g_3=root.children[6].n_3,
                n_g_4=root.children[6].n_4,
                n_h_0=root.children[7].n_0,
                n_h_3=root.children[7].n_3,
                n_h_4=root.children[7].n_4,
            )
            v_all_rearranged = v_all[rearranged_idxes]

            # Step 6:
            # Check each face in order
            face_idxes = [
                0,
                root.n_0,
                root.n_0 + root.n_1,
                root.n_0 + root.n_1 + root.n_2,
                root.n_0 + root.n_1 + root.n_2 + root.n_3,
                root.n_0 + root.n_1 + root.n_2 + root.n_3 + root.n_4,
                root.n_0
                + root.n_1
                + root.n_2
                + root.n_3
                + root.n_4
                + root.n_5,
            ]
            for j in range(6):
                expected_face_j = gauss_pts[
                    face_idxes[j] : face_idxes[j + 1], i
                ]
                computed_face_j = v_all_rearranged[
                    face_idxes[j] : face_idxes[j + 1]
                ]

                v = jnp.allclose(expected_face_j, computed_face_j)
                if not v:
                    print("####### test_1: face_j = ", j + 1, " v = ", v)
                    print("test_1: expected_face_j = ", expected_face_j)
                    print("test_1: computed_face_j = ", computed_face_j)
                    print("test_1: diffs: ", expected_face_j - computed_face_j)
                assert v
        # assert False
        jax.clear_caches()
