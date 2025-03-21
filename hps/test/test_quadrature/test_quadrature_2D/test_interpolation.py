import pytest
import jax.numpy as jnp
import numpy as np
from hps.src.quadrature.quadrature_utils import (
    affine_transform,
    chebyshev_points,
)
from hps.src.quadrature.quad_2D.interpolation import (
    refinement_operator,
    precompute_refining_coarsening_ops,
    _interp_to_point,
    precompute_I_P_0_matrix,
    precompute_Q_I_matrix,
    interp_from_nonuniform_hps_to_regular_grid,
)
from hps.src.quadrature.trees import (
    Node,
    add_uniform_levels,
    # _corners_for_quad_subdivision,
)
from hps.src.quadrature.quad_2D.grid_creation import (
    corners_to_cheby_points_lst,
    get_all_boundary_gauss_legendre_points,
    get_all_leaf_2d_cheby_points,
    get_all_leaf_2d_cheby_points_uniform_refinement,
)


class Test_refinement_operator:
    def test_0(self) -> None:
        # Test to make sure the shapes are right.
        p = 2
        x = refinement_operator(p)

        print("test_0: x", x)
        assert x.shape == (4 * p**2, p**2)
        assert not jnp.any(jnp.isnan(x))
        assert not jnp.any(jnp.isinf(x))

    def test_1(self) -> None:
        """Refinement should be exact on low-degree polynomials"""
        p = 4
        x = refinement_operator(p)
        assert not jnp.any(jnp.isnan(x))
        north = 1
        south = -1
        east = 1
        west = -1
        corners = jnp.array(
            [[west, south], [east, south], [east, north], [west, north]],
            dtype=jnp.float64,
        )

        # These are the Chebyshev points, ordered to be exterior points, then interior points.
        pts_0 = get_all_leaf_2d_cheby_points_uniform_refinement(
            p, 0, corners
        ).reshape((-1, 2))

        # These are four copies of the Che
        pts_1 = get_all_leaf_2d_cheby_points_uniform_refinement(
            p, 1, corners
        ).reshape((-1, 2))

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + 3x**2 - 20"""
            return x[..., 1] + 3 * x[..., 0] ** 2 - 20.0

        f_0 = f(pts_0)
        f_1 = f(pts_1)
        f_interp = x @ f_0

        # print("test_1: x: ", x)
        print("test_1: pts_1: ", pts_1)

        print("test_1: f_0: ", f_0)
        print("test_1: f_1: ", f_1)
        print("test_1: f_interp: ", f_interp)
        print("test_1: diffs: ", f_1 - f_interp)

        assert jnp.allclose(f_1, f_interp)


class Test_refining_coarsening_ops:
    def test_0(self) -> None:
        q = 12

        ref, coarse = precompute_refining_coarsening_ops(q)

        assert ref.shape == (2 * q, q)
        assert coarse.shape == (q, 2 * q)

        assert not jnp.any(jnp.isnan(ref))
        assert not jnp.any(jnp.isinf(ref))
        assert not jnp.any(jnp.isnan(coarse))
        assert not jnp.any(jnp.isinf(coarse))

    def test_1(self) -> None:
        q = 12
        ref, coarse = precompute_refining_coarsening_ops(q)

        def f(x: jnp.array) -> jnp.array:
            """f(x) = 3 + 4x - 5x**2"""
            return 3 + 4 * x - 5 * x**2

        gauss_panel = np.polynomial.legendre.leggauss(q)[0]
        double_gauss_panel = np.concatenate(
            [
                affine_transform(gauss_panel, [-1.0, 0.0]),
                affine_transform(gauss_panel, [0.0, 1.0]),
            ]
        )
        f_vals = f(gauss_panel)
        f_ref = ref @ f_vals
        f_ref_expected = f(double_gauss_panel)

        assert jnp.allclose(f_ref, f_ref_expected)

        f_coarse = coarse @ f_ref_expected
        f_coarse_expected = f_vals
        assert jnp.allclose(f_coarse, f_coarse_expected)


class Test_precompute_I_P_0_matrix:
    def test_0(self) -> None:
        """Makes sure the output is correct shape"""
        p = 8
        q = 6

        I_P = precompute_I_P_0_matrix(p, q)

        assert I_P.shape == (4 * (p - 1), 4 * q)

    def test_1(self) -> None:
        """Makes sure low-degree polynomial interpolation is exact."""
        p = 8
        q = 6

        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2

        root = Node(xmin=west, xmax=east, ymin=south, ymax=north)

        bdry_pts = get_all_boundary_gauss_legendre_points(q, root)
        cheby_pts = get_all_leaf_2d_cheby_points(p, root)
        interior_pts = cheby_pts[0, : 4 * (p - 1)]

        print("test_1: bdry_pts shape: ", bdry_pts.shape)
        print("test_1: cheby_pts shape: ", cheby_pts.shape)
        print("test_1: interior_pts shape: ", interior_pts.shape)

        def f(x):
            # f(x,y) = 3x - y**2
            return 3 * x[..., 0] - x[..., 1] ** 2

        # Compute the function values at the Gauss boundary points
        f_vals = f(bdry_pts)

        I_P_0 = precompute_I_P_0_matrix(p, q)

        f_interp = I_P_0 @ f_vals

        f_expected = f(interior_pts)

        print("test_1: f_interp shape: ", f_interp.shape)
        print("test_1: f_expected shape: ", f_expected.shape)
        assert jnp.allclose(f_interp, f_expected)


class Test_precompute_Q_I_matrix:
    def test_0(self) -> None:
        """Makes sure the output is correct shape"""
        p = 8
        q = 6

        Q_I = precompute_Q_I_matrix(p, q)

        assert Q_I.shape == (4 * q, 4 * p)

    def test_1(self) -> None:
        """Makes sure low-degree polynomial interpolation is exact."""
        p = 8
        q = 6

        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2

        root = Node(xmin=west, xmax=east, ymin=south, ymax=north)

        bdry_pts = get_all_boundary_gauss_legendre_points(q, root)
        cheby_pts = get_all_leaf_2d_cheby_points(p, root)
        exterior_pts = cheby_pts[0, : 4 * (p - 1)]

        def f(x):
            # f(x,y) = 3x - y**2
            return 3 * x[..., 0] - x[..., 1] ** 2

        f_vals = jnp.concatenate(
            [
                f(exterior_pts[:p]),
                f(exterior_pts[p - 1 : 2 * p - 1]),
                f(exterior_pts[2 * p - 2 : 3 * p - 2]),
                f(exterior_pts[3 * p - 3 :]),
                f(exterior_pts[0]).reshape(1),
            ]
        )

        # Compute the function values at the Gauss boundary points
        print("test_1: f_vals shape: ", f_vals.shape)
        Q_I = precompute_Q_I_matrix(p, q)

        f_interp = Q_I @ f_vals

        f_expected = f(bdry_pts)

        print("test_1: f_interp shape: ", f_interp.shape)
        print("test_1: f_expected shape: ", f_expected.shape)

        assert jnp.allclose(f_interp, f_expected)


class Test_interp_from_nonuniform_hps_to_regular_grid:
    def test_0(self) -> None:
        # Make sure returns correct shape.
        l = 2
        p = 10
        n_x = 7

        xmin = -1.0
        xmax = 1.0
        ymin = 3.0
        ymax = 4.0

        root = Node(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

        f_evals = jnp.ones((4**l, p**2))

        out_0, out_1 = interp_from_nonuniform_hps_to_regular_grid(
            root, p, f_evals, n_x
        )

        assert out_0.shape == (n_x, n_x)
        assert not jnp.any(jnp.isnan(out_0))
        assert not jnp.any(jnp.isinf(out_0))

        assert out_1.shape == (n_x, n_x, 2)
        assert jnp.all(out_1[..., 0] >= xmin)
        assert jnp.all(out_1[..., 0] <= xmax)
        assert jnp.all(out_1[..., 1] >= ymin)
        assert jnp.all(out_1[..., 1] <= ymax)

    def test_1(self) -> None:
        """Check that low-degree polynomial interpolation is exact."""
        l = 3
        p = 8
        n_x = 3

        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0

        root = Node(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        add_uniform_levels(root, l, p - 2)

        # Generate chebyshev grid

        hps_grid_pts = get_all_leaf_2d_cheby_points(p, root)

        def f(x: jnp.array) -> jnp.array:
            # x has shape (..., 2)
            # f(x,y) = 3y - 4x^2
            return 3 * x[..., 1] - 4 * x[..., 0] ** 2

        f_evals = f(hps_grid_pts)

        interp_vals, target_pts = interp_from_nonuniform_hps_to_regular_grid(
            root, p, f_evals, n_x
        )

        f_target = f(target_pts)

        print("test_1: f_target: ", f_target)
        print("test_1: interp_vals: ", interp_vals)
        diffs = interp_vals - f_target

        print("test_1: diffs : ", diffs)
        assert jnp.allclose(interp_vals, f_target)

    @pytest.mark.skip("Super slow test.")
    def test_2(self) -> None:
        """Check the accuracy of interpolating a plane wave with wavenumber 100 over-resolved to 50 points per wavelength, and
        then interpolated to a 500x500 grid."""
        l = 8
        p = 20
        n_x = 500

        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0

        root = Node(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        add_uniform_levels(root, l, p - 2)

        # Generate chebyshev grid

        hps_grid_pts = get_all_leaf_2d_cheby_points(p, root)
        k = 100.0

        def f(x: jnp.array) -> jnp.array:
            # x has shape (..., 2)
            # f(x,y) = exp(i * k * 2 pi * x)
            exponent = 1j * k * 2 * jnp.pi * x[..., 0]
            return jnp.exp(exponent)

        f_evals = f(hps_grid_pts)

        interp_vals, target_pts = interp_from_nonuniform_hps_to_regular_grid(
            root, p, f_evals, n_x
        )

        f_target = f(target_pts)

        # print("test_1: f_target: ", f_target)
        # print("test_1: interp_vals: ", interp_vals)
        diffs = interp_vals - f_target

        print("test_1: max diffs : ", jnp.max(jnp.abs(diffs)))
        assert jnp.allclose(interp_vals, f_target)


class Test__interp_to_point:
    def test_0(self) -> None:
        # l = 3
        corners = jnp.array([[-1, -1], [1.0, -1.0], [1.0, 1], [-1, 1]])

        # xval = jnp.array([-0.99])
        # yval = jnp.array([-0.99])
        xval = jnp.array([0.0])
        yval = jnp.array([0.0])

        p = 2
        f = np.random.normal(size=(p**2))

        out = _interp_to_point(xval, yval, corners, f, p)

        assert out.shape == (1,)

    def test_1(self) -> None:
        """Check that low-degree polynomial interpolation is exact."""
        corners = jnp.array([[-1, -1], [1.0, -1.0], [1.0, 1], [-1, 1]])
        xval = jnp.array([0.250])
        yval = jnp.array([-1.0])

        xy = jnp.array([xval, yval]).T
        print("test_1: xy: ", xy)
        print("test_1: xy shape: ", xy.shape)

        p = 8
        cheby_pts_1d = chebyshev_points(p)[0]

        # Get the 2D Cheby grid of this size
        cheby_pts_grid = corners_to_cheby_points_lst(corners, cheby_pts_1d)

        def f(x: jnp.array) -> jnp.array:
            # x has shape (..., 2)
            # f(x,y) = x
            return x[..., 1] ** 3 - 4 * x[..., 0]

        f_evals = f(cheby_pts_grid)
        interp_val = _interp_to_point(xval, yval, corners, f_evals, p)
        expected = f(xy)

        print("test_1: interp_val: ", interp_val)
        print("test_1: expected: ", expected)

        assert jnp.allclose(interp_val, expected)


if __name__ == "__main__":
    pytest.main()
