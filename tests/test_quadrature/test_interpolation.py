from jaxhps.quadrature import (
    barycentric_lagrange_interpolation_matrix_1D,
    barycentric_lagrange_interpolation_matrix_2D,
    barycentric_lagrange_interpolation_matrix_3D,
    chebyshev_points,
    gauss_points,
    meshgrid_to_lst_of_pts,
)

import jax.numpy as jnp


class Test_barycentrix_lagrange_interpolation_matrix_1D:
    def test_0(self) -> None:
        from_pts = jnp.array([-0.5, 0.5])

        to_pts = jnp.array(
            [
                0.0,
            ]
        )
        m = barycentric_lagrange_interpolation_matrix_1D(from_pts, to_pts)
        expected_m = jnp.array([[0.5, 0.5]])
        assert jnp.allclose(m, expected_m)

    def test_1(self) -> None:
        """Checks the interpolation matrix is non-infinite and non-nan when target points overlap with source points"""
        p = 5
        from_pts = chebyshev_points(p)

        to_pts = jnp.array([-1.0, 0.5, 1.0])

        def f(x: jnp.array) -> jnp.array:
            # f(x) = x - x^2
            return x - x**2

        interp_mat = barycentric_lagrange_interpolation_matrix_1D(
            from_pts, to_pts
        )
        assert not jnp.any(jnp.isnan(interp_mat))
        assert not jnp.any(jnp.isinf(interp_mat))

        f_evals = f(to_pts)
        f_evals_source = f(from_pts)
        f_interp = interp_mat @ f_evals_source
        assert jnp.allclose(f_interp, f_evals)


class Test_barycentric_lagrange_interpolation_matrix_2D:
    def test_0(self) -> None:
        p = 5
        q = 3

        cheby_pts = chebyshev_points(p)
        gauss_pts = gauss_points(q)

        mat = barycentric_lagrange_interpolation_matrix_2D(
            cheby_pts, cheby_pts, gauss_pts, gauss_pts
        )
        assert not jnp.any(jnp.isnan(mat))
        assert mat.shape == (q**2, p**2)

        mat_2 = barycentric_lagrange_interpolation_matrix_2D(
            gauss_pts, gauss_pts, cheby_pts, cheby_pts
        )
        assert mat_2.shape == (p**2, q**2)

    def test_1(self):
        """Tests accuracy of interpolation on the polynomial function f(x,y) = x^2 + y^2.
        Different numbers of x and y from points"""
        p_x = 5
        p_y = 4
        q_x = 3
        q_y = 6

        cheby_pts_x = chebyshev_points(p_x)
        cheby_pts_y = chebyshev_points(p_y)
        gauss_pts_x = gauss_points(q_x)
        gauss_pts_y = gauss_points(q_y)

        mat = barycentric_lagrange_interpolation_matrix_2D(
            cheby_pts_x, cheby_pts_y, gauss_pts_x, gauss_pts_y
        )

        from_X, from_Y = jnp.meshgrid(cheby_pts_x, cheby_pts_y, indexing="ij")
        from_pts = jnp.stack((from_X.flatten(), from_Y.flatten()), axis=-1)
        # print("test_1: from_pts = ", from_pts)

        to_X, to_Y = jnp.meshgrid(gauss_pts_x, gauss_pts_y, indexing="ij")
        to_pts = jnp.stack((to_X.flatten(), to_Y.flatten()), axis=-1)
        # print("test_1: to_pts = ", to_pts)

        def f(x):
            return x[:, 0] ** 2 + x[:, 1] ** 2

        f_evals_cheby = f(from_pts)
        f_interp = mat @ f_evals_cheby
        f_evals_gauss = f(to_pts)

        print("test_1: f_interp = ", f_interp)
        print("test_1: f_evals_gauss = ", f_evals_gauss)
        print("test_1: diffs: ", f_interp - f_evals_gauss)

        assert jnp.allclose(f_interp, f_evals_gauss)

    def test_2(self) -> None:
        """Tests accuracy when there is a single output point."""
        q = 5

        cheby_pts = chebyshev_points(q)
        X, Y = jnp.meshgrid(cheby_pts, cheby_pts, indexing="ij")
        cheby_pts_2d = meshgrid_to_lst_of_pts(X, Y)

        out_pt_x = jnp.array([0.25])
        out_pt_y = jnp.array([0.25])

        mat = barycentric_lagrange_interpolation_matrix_2D(
            cheby_pts, cheby_pts, out_pt_x, out_pt_y
        )
        assert not jnp.any(jnp.isnan(mat))
        assert mat.shape == (1, q**2)

        out_pts = jnp.stack((out_pt_x, out_pt_y), axis=-1)

        def f(x):
            return x[:, 0] ** 2 + x[:, 1] ** 2

        f_evals_cheby = f(cheby_pts_2d)
        f_interp = mat @ f_evals_cheby
        f_evals_out = f(out_pts)

        assert jnp.allclose(f_interp, f_evals_out)

    def test_3(self) -> None:
        """Tests things are still finite when using target points at the convex hull of the source points."""
        p = 5
        source_pts = chebyshev_points(p)

        target_pts = jnp.array([-1.0, 0.0, 0.5, 1.0])

        mat = barycentric_lagrange_interpolation_matrix_2D(
            source_pts, source_pts, target_pts, target_pts
        )

        assert not jnp.any(jnp.isnan(mat))
        assert not jnp.any(jnp.isinf(mat))

        from_X, from_Y = jnp.meshgrid(source_pts, source_pts, indexing="ij")
        from_pts = jnp.stack((from_X.flatten(), from_Y.flatten()), axis=-1)
        # print("test_1: from_pts = ", from_pts)

        to_X, to_Y = jnp.meshgrid(target_pts, target_pts, indexing="ij")
        to_pts = jnp.stack((to_X.flatten(), to_Y.flatten()), axis=-1)
        # print("test_1: to_pts = ", to_pts)

        def f(x):
            return x[:, 0]  # ** 2 + x[:, 1] ** 2

        f_evals_source = f(from_pts)
        f_interp = mat @ f_evals_source
        f_evals_target = f(to_pts)

        print("test_2: diffs: ", f_interp - f_evals_target)
        assert jnp.allclose(f_interp, f_evals_target)

    # @pytest.mark.skip(reason="Not yet supported.")
    def test_4(self) -> None:
        """Tests that shapes are correct when from_x and from_y have different lengths."""
        n_x = 5
        n_y = 7
        from_x = jnp.linspace(0.0, 1.0, n_x)
        from_y = jnp.linspace(0.0, 1.0, n_y)
        to_x = jnp.array([0.0, 0.5, 1.0])
        to_y = jnp.array([0.0, 0.5, 1.0])

        mat = barycentric_lagrange_interpolation_matrix_2D(
            from_x, from_y, to_x, to_y
        )
        assert not jnp.any(jnp.isnan(mat))
        assert not jnp.any(jnp.isinf(mat))
        assert mat.shape == (9, n_x * n_y)


class Test_barycentric_lagrange_interpolation_matrix_3D:
    def test_0(self) -> None:
        """Checks that things are working properly."""
        p = 3
        q = 2

        cheby_pts = chebyshev_points(p)
        gauss_pts = gauss_points(q)

        mat = barycentric_lagrange_interpolation_matrix_3D(
            cheby_pts, cheby_pts, cheby_pts, gauss_pts, gauss_pts, gauss_pts
        )

        assert not jnp.any(jnp.isnan(mat))
        assert mat.shape == (q**3, p**3)

        mat_2 = barycentric_lagrange_interpolation_matrix_3D(
            gauss_pts, gauss_pts, gauss_pts, cheby_pts, cheby_pts, cheby_pts
        )
        assert mat_2.shape == (p**3, q**3)

    def test_1(self) -> None:
        """Tests accuracy of interpolation on the polynomial function f(x,y,z) = x^2 + y^2 + 3z"""
        p_x = 5
        p_y = 4
        p_z = 3
        q_x = 7
        q_y = 6
        q_z = 5

        cheby_pts_x = chebyshev_points(p_x)
        cheby_pts_y = chebyshev_points(p_y)
        cheby_pts_z = chebyshev_points(p_z)
        gauss_pts_x = gauss_points(q_x)
        gauss_pts_y = gauss_points(q_y)
        gauss_pts_z = gauss_points(q_z)

        mat = barycentric_lagrange_interpolation_matrix_3D(
            cheby_pts_x,
            cheby_pts_y,
            cheby_pts_z,
            gauss_pts_x,
            gauss_pts_y,
            gauss_pts_z,
        )
        print("test_1: mat = ", mat)

        from_X, from_Y, from_Z = jnp.meshgrid(
            cheby_pts_x, cheby_pts_y, cheby_pts_z, indexing="ij"
        )
        from_pts = jnp.stack(
            (from_X.flatten(), from_Y.flatten(), from_Z.flatten()), axis=-1
        )
        # print("test_1: from_pts = ", from_pts)

        to_X, to_Y, to_Z = jnp.meshgrid(
            gauss_pts_x, gauss_pts_y, gauss_pts_z, indexing="ij"
        )
        to_pts = jnp.stack(
            (to_X.flatten(), to_Y.flatten(), to_Z.flatten()), axis=-1
        )
        # print("test_1: to_pts = ", to_pts)

        def f(x):
            return x[:, 0] ** 2 + x[:, 1] ** 2 + 3 * x[:, 2]

        f_evals_cheby = f(from_pts)
        f_interp = mat @ f_evals_cheby
        f_evals_gauss = f(to_pts)

        diffs = f_interp - f_evals_gauss
        print("test_1: f_cheby", f_evals_cheby)
        print("test_1: f_interp", f_interp)
        print("test_1: f_evals_gauss", f_evals_gauss)
        print("test_1: diffs", diffs)

        assert jnp.allclose(f_interp, f_evals_gauss)

    def test_2(self) -> None:
        """Tests whether the interpolation matrix is scale-invariant. That is, if we scale the
        input points, the interpolation matrix should be the same."""
        p = 5
        q = 3

        cheby_pts = chebyshev_points(p)
        gauss_pts = gauss_points(q)

        mat = barycentric_lagrange_interpolation_matrix_3D(
            cheby_pts, cheby_pts, cheby_pts, gauss_pts, gauss_pts, gauss_pts
        )

        scaled_cheby = jnp.pi * cheby_pts
        scaled_gauss = jnp.pi * gauss_pts
        scaled_mat = barycentric_lagrange_interpolation_matrix_3D(
            scaled_cheby,
            scaled_cheby,
            scaled_cheby,
            scaled_gauss,
            scaled_gauss,
            scaled_gauss,
        )

        assert jnp.allclose(mat, scaled_mat)

    def test_3(self) -> None:
        """Tests the case where some of the input and output points are the same."""

        p1 = 3
        p2 = 2

        cheby1 = chebyshev_points(p1)
        cheby2 = chebyshev_points(p2)

        x = barycentric_lagrange_interpolation_matrix_3D(
            cheby1, cheby1, cheby1, cheby2, cheby2, cheby2
        )
        print("test_3: x = ", x)
        assert not jnp.any(jnp.isnan(x))
        assert not jnp.any(jnp.isinf(x))
