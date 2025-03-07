import jax.numpy as jnp
import jax
import numpy as np
import pytest

from hps.src.quadrature.quadrature_utils import (
    differentiation_matrix_1d,
    chebyshev_points,
    barycentric_lagrange_interpolation_matrix,
    affine_transform,
    barycentric_lagrange_2d_interpolation_matrix,
    barycentric_lagrange_3d_interpolation_matrix,
    chebyshev_weights,
)
from hps.src.quadrature.quad_3D.grid_creation import (
    corners_to_cheby_points_lst,
    corners_to_gauss_points_lst,
)
from hps.src.quadrature.quad_3D.indexing import get_face_1_idxes
from hps.src.test_utils import check_arrays_close
from hps.src.utils import meshgrid_to_lst_of_pts


class Test_differentiation_matrix_1d:
    def test_0(self) -> None:
        p = jnp.arange(10)
        o = differentiation_matrix_1d(p)
        assert o.shape == (10, 10), o.shape

    def test_1(self) -> None:
        """Check n=2 against
        example from MATLAB textbook (T00)
        """

        p, _ = chebyshev_points(2)

        d = differentiation_matrix_1d(p)

        expected_d = np.array([[-1 / 2, 1 / 2], [-1 / 2, 1 / 2]])

        print(d)
        print(expected_d)

        check_arrays_close(d, expected_d)

    def test_2(self) -> None:
        """Check n=3 against
        example from MATLAB textbook (T00)
        """

        p, _ = chebyshev_points(3)

        d = differentiation_matrix_1d(p)

        expected_d = np.array(
            [[-3 / 2, 2, -1 / 2], [-1 / 2, 0, 1 / 2], [1 / 2, -2, 3 / 2]]
        )

        print(d)
        print(expected_d)
        print(d - expected_d)

        check_arrays_close(d, expected_d, rtol=1e-05, atol=1e-05)

    def test_3(self) -> None:
        """Check n=4 against example from MATLAB textbook (T00)"""

        p, _ = chebyshev_points(4)

        d = differentiation_matrix_1d(p)

        expected_d = np.array(
            [
                [19 / 6, -4, 4 / 3, -1 / 2],
                [1, -1 / 3, -1, 1 / 3],
                [-1 / 3, 1, 1 / 3, -1],
                [1 / 2, -4 / 3, 4, -19 / 6],
            ]
        )
        expected_d = -1 * expected_d

        print(d)
        print(expected_d)
        print(d - expected_d)

        check_arrays_close(d, expected_d)

    def test_4(self) -> None:
        """Checks the matrix differentiates f(x) = x^2 correctly."""

        n = 7
        p, _ = chebyshev_points(n)
        print("test_4: p: ", p)
        d = differentiation_matrix_1d(p)
        f = p**2
        df = d @ f
        expected = 2 * p
        print("test_4: df: ", df)
        print("test_4: expected: ", expected)
        assert jnp.allclose(df, expected)


class Test_chebyshev_weights:
    def test_0(self) -> None:
        """Checks to make sure things run correctly."""
        p = 6
        bounds = np.array([-1, 1])
        w = chebyshev_weights(p, bounds)
        assert w.shape == (p,)
        assert not jnp.any(jnp.isnan(w))
        assert not jnp.any(jnp.isinf(w))

        p = 7
        bounds = np.array([-1, 1])
        w = chebyshev_weights(p, bounds)
        assert w.shape == (p,)
        assert not jnp.any(jnp.isnan(w))
        assert not jnp.any(jnp.isinf(w))

    def test_1(self) -> None:
        """Checks that a constant function has the expected integral."""

        p = 4
        bounds = np.array([-1, 1])
        w = chebyshev_weights(p, bounds)
        assert w.shape == (p,)
        print("test_1: w = ", w)
        f = np.ones(p)
        integral = w @ f
        print("test_1: integral = ", integral)
        expected = 2
        assert jnp.allclose(integral, expected)

        p = 5
        bounds = np.array([-1, 1])
        w = chebyshev_weights(p, bounds)
        assert w.shape == (p,)
        print("test_1: w = ", w)
        f = np.ones(p)
        integral = w @ f
        print("test_1: integral = ", integral)
        expected = 2
        assert jnp.allclose(integral, expected)

    def test_2(self) -> None:
        """Checks a nonzero polynomial over [0,1]"""
        p = 6
        bounds = np.array([0, 1])
        w = chebyshev_weights(p, bounds)
        print("test_2: w = ", w)
        assert w.shape == (p,)

        def f(x):
            return x**3

        pts = affine_transform(chebyshev_points(p)[0], bounds)
        f_evals = f(pts)
        integral = w @ f_evals
        expected = 1 / 4

        print("test_2: integral = ", integral)
        print("test_2: expected = ", expected)

        assert jnp.allclose(integral, expected)

        p = 7
        bounds = np.array([0, 1])
        w = chebyshev_weights(p, bounds)
        print("test_2: w = ", w)
        assert w.shape == (p,)

        def f(x):
            return x**3

        pts = affine_transform(chebyshev_points(p)[0], bounds)
        f_evals = f(pts)
        integral = w @ f_evals
        expected = 1 / 4

        print("test_2: integral = ", integral)
        print("test_2: expected = ", expected)

        assert jnp.allclose(integral, expected)

    def test_3(self) -> None:
        """Observed this test case in the wild."""
        p = 16
        bounds = np.array([0.5, 0.625])
        w = chebyshev_weights(p, bounds)
        # Make sure not nan or inf
        print("test_3: w = ", w)
        assert not jnp.any(jnp.isnan(w))
        assert not jnp.any(jnp.isinf(w))


class Test_chebyshev_points:
    def test_0(self) -> None:
        n = 2
        p, w = chebyshev_points(n)
        expected_p = np.array([-1, 1.0])
        check_arrays_close(p, expected_p)

    def test_1(self) -> None:
        n = 3
        p, w = chebyshev_points(n)
        expected_p = np.array([-1, 0.0, 1.0])
        print(p)
        check_arrays_close(p, expected_p, atol=1e-06)


class Test_barycentrix_lagrange_interpolation_matrix:
    def test_0(self) -> None:

        from_pts = np.array([-0.5, 0.5])

        to_pts = np.array(
            [
                0.0,
            ]
        )
        m = barycentric_lagrange_interpolation_matrix(from_pts, to_pts)
        expected_m = np.array([[0.5, 0.5]])
        check_arrays_close(m, expected_m)

    def test_1(self) -> None:
        """Checks the interpolation matrix is non-infinite and non-nan when target points overlap with source points"""
        p = 5
        from_pts = chebyshev_points(p)[0]

        to_pts = jnp.array([-1.0, 0.5, 1.0])

        def f(x: jnp.array) -> jnp.array:
            # f(x) = x - x^2
            return x - x**2

        interp_mat = barycentric_lagrange_interpolation_matrix(from_pts, to_pts)
        assert not jnp.any(jnp.isnan(interp_mat))
        assert not jnp.any(jnp.isinf(interp_mat))

        f_evals = f(to_pts)
        f_evals_source = f(from_pts)
        f_interp = interp_mat @ f_evals_source
        assert jnp.allclose(f_interp, f_evals)


class Test_barycentric_lagrange_2d_interp_matrix:
    def test_0(self) -> None:
        p = 5
        q = 3

        cheby_pts = chebyshev_points(p)[0]
        gauss_pts = np.polynomial.legendre.leggauss(q)[0]

        mat = barycentric_lagrange_2d_interpolation_matrix(
            cheby_pts, cheby_pts, gauss_pts, gauss_pts
        )
        assert not jnp.any(jnp.isnan(mat))
        assert mat.shape == (q**2, p**2)

        mat_2 = barycentric_lagrange_2d_interpolation_matrix(
            gauss_pts, gauss_pts, cheby_pts, cheby_pts
        )
        assert mat_2.shape == (p**2, q**2)

    def test_1(self):
        """Tests accuracy of interpolation on the polynomial function f(x,y) = x^2 + y^2."""
        p = 5
        q = 3

        cheby_pts = chebyshev_points(p)[0]
        gauss_pts = np.polynomial.legendre.leggauss(q)[0]

        mat = barycentric_lagrange_2d_interpolation_matrix(
            cheby_pts, cheby_pts, gauss_pts, gauss_pts
        )

        from_X, from_Y = jnp.meshgrid(cheby_pts, cheby_pts, indexing="ij")
        from_pts = jnp.stack((from_X.flatten(), from_Y.flatten()), axis=-1)
        # print("test_1: from_pts = ", from_pts)

        to_X, to_Y = jnp.meshgrid(gauss_pts, gauss_pts, indexing="ij")
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

        cheby_pts = chebyshev_points(q)[0]
        X, Y = jnp.meshgrid(cheby_pts, cheby_pts, indexing="ij")
        cheby_pts_2d = meshgrid_to_lst_of_pts(X, Y)

        out_pt_x = np.array([0.25])
        out_pt_y = np.array([0.25])

        mat = barycentric_lagrange_2d_interpolation_matrix(
            cheby_pts, cheby_pts, out_pt_x, out_pt_y
        )
        assert not jnp.any(jnp.isnan(mat))
        assert mat.shape == (1, q**2)

        out_pts = np.stack((out_pt_x, out_pt_y), axis=-1)

        def f(x):
            return x[:, 0] ** 2 + x[:, 1] ** 2

        f_evals_cheby = f(cheby_pts_2d)
        f_interp = mat @ f_evals_cheby
        f_evals_out = f(out_pts)

        assert jnp.allclose(f_interp, f_evals_out)

    def test_3(self) -> None:
        """Tests things are still finite when using target points at the convex hull of the source points."""
        p = 5
        source_pts = chebyshev_points(p)[0]

        target_pts = jnp.array([-1.0, 0.0, 0.5, 1.0])

        mat = barycentric_lagrange_2d_interpolation_matrix(
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


class Test_barycentric_lagrange_3d_interp_matrix:
    def test_0(self) -> None:
        """Checks that things are working properly."""
        p = 3
        q = 2

        cheby_pts = chebyshev_points(p)[0]
        gauss_pts = np.polynomial.legendre.leggauss(q)[0]

        mat = barycentric_lagrange_3d_interpolation_matrix(
            cheby_pts, cheby_pts, cheby_pts, gauss_pts, gauss_pts, gauss_pts
        )

        assert not jnp.any(jnp.isnan(mat))
        assert mat.shape == (q**3, p**3)

        mat_2 = barycentric_lagrange_3d_interpolation_matrix(
            gauss_pts, gauss_pts, gauss_pts, cheby_pts, cheby_pts, cheby_pts
        )
        assert mat_2.shape == (p**3, q**3)

    def test_1(self) -> None:
        """Tests accuracy of interpolation on the polynomial function f(x,y,z) = x^2 + y^2 + 3z"""
        p = 5
        q = 4

        cheby_pts = chebyshev_points(p)[0]
        gauss_pts = np.polynomial.legendre.leggauss(q)[0]

        mat = barycentric_lagrange_3d_interpolation_matrix(
            cheby_pts, cheby_pts, cheby_pts, gauss_pts, gauss_pts, gauss_pts
        )
        print("test_1: mat = ", mat)

        from_X, from_Y, from_Z = jnp.meshgrid(
            cheby_pts, cheby_pts, cheby_pts, indexing="ij"
        )
        from_pts = jnp.stack(
            (from_X.flatten(), from_Y.flatten(), from_Z.flatten()), axis=-1
        )
        # print("test_1: from_pts = ", from_pts)

        to_X, to_Y, to_Z = jnp.meshgrid(gauss_pts, gauss_pts, gauss_pts, indexing="ij")
        to_pts = jnp.stack((to_X.flatten(), to_Y.flatten(), to_Z.flatten()), axis=-1)
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

        cheby_pts = chebyshev_points(p)[0]
        gauss_pts = np.polynomial.legendre.leggauss(q)[0]

        mat = barycentric_lagrange_3d_interpolation_matrix(
            cheby_pts, cheby_pts, cheby_pts, gauss_pts, gauss_pts, gauss_pts
        )

        scaled_cheby = jnp.pi * cheby_pts
        scaled_gauss = jnp.pi * gauss_pts
        scaled_mat = barycentric_lagrange_3d_interpolation_matrix(
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

        cheby1 = chebyshev_points(p1)[0]
        cheby2 = chebyshev_points(p2)[0]

        x = barycentric_lagrange_3d_interpolation_matrix(
            cheby1, cheby1, cheby1, cheby2, cheby2, cheby2
        )
        print("test_3: x = ", x)
        assert not jnp.any(jnp.isnan(x))
        assert not jnp.any(jnp.isinf(x))


class Test_affine_transform:
    def test_0(self) -> None:
        """Check that the affine transform works as expected."""
        n = 10
        x = chebyshev_points(n)[0]
        ab = jnp.array([0, 1])
        y = affine_transform(x, ab)
        expected_y = x * 0.5 + 0.5
        check_arrays_close(y, expected_y)


if __name__ == "__main__":

    pytest.main()
