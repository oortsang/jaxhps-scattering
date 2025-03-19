import jax.numpy as jnp
import jax
import numpy as np
import pytest

import matplotlib.pyplot as plt

from hps.src.quadrature.quad_2D.grid_creation import (
    chebyshev_points,
    corners_to_cheby_points_lst,
    vmapped_corners,
    vmapped_corners_to_cheby_points_lst,
    get_all_boundary_gauss_legendre_points,
    get_all_leaf_2d_cheby_points,
)
from hps.src.test_utils import check_arrays_close
from hps.src.quadrature.quad_2D.interpolation import (
    precompute_P_matrix,
    precompute_Q_D_matrix,
    precompute_I_P_0_matrix,
)
from hps.src.quadrature.quad_2D.differentiation import (
    precompute_diff_operators,
    precompute_N_matrix,
    precompute_N_tilde_matrix,
    precompute_G_matrix,
    precompute_F_matrix,
)
from hps.src.quadrature.trees import Node, _corners_for_quad_subdivision


class Test_precompute_N_matrix:
    def test_0(self) -> None:
        """Check the shape of the output."""
        p = 8
        q = 6
        du_dx, du_dy, _, _, _ = precompute_diff_operators(p, 1.0)
        out = precompute_N_matrix(du_dx, du_dy, p)
        assert out.shape == (4 * p, p**2)

    def test_1(self) -> None:
        """Check the output is correct on low-degree polynomials."""
        p = 8
        q = 6
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2

        root = Node(xmin=west, xmax=east, ymin=south, ymax=north)
        half_side_len = jnp.pi / 2
        du_dx, du_dy, _, _, _ = precompute_diff_operators(p, half_side_len)
        out = precompute_N_matrix(du_dx, du_dy, p)

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -3
            return -3 * jnp.ones_like(x[..., 1])

        # Set up the Chebyshev points.
        cheby_pts = get_all_leaf_2d_cheby_points(p, root)
        interior_pts = cheby_pts[0, 4 * (p - 1) :]
        exterior_pts = cheby_pts[0, : 4 * (p - 1)]
        all_pts = cheby_pts[0]

        # Compute the function values at the Chebyshev points.
        f_vals = f(all_pts)

        computed_normals = out @ f_vals

        expected_normals = jnp.concatenate(
            [
                -1 * dfdy(exterior_pts[:p]),
                dfdx(exterior_pts[p - 1 : 2 * p - 1]),
                dfdy(exterior_pts[2 * p - 2 : 3 * p - 2]),
                -1 * dfdx(exterior_pts[3 * p - 3 :]),
                -1 * dfdx(exterior_pts[0]).reshape(1),
            ]
        )
        print("test_1: computed_normals shape = ", computed_normals.shape)
        print("test_1: expected_normals shape = ", expected_normals.shape)

        # Check the computed normals against the expected normals. side-by-side.

        # S side
        print("test_1: computed_normals[:p] = ", computed_normals[:p])
        print("test_1: expected_normals[:p] = ", expected_normals[:p])
        assert jnp.allclose(computed_normals[:p], expected_normals[:p])

        # E side
        print("test_1: computed_normals[p:2*p] = ", computed_normals[p : 2 * p])
        print("test_1: expected_normals[p:2*p] = ", expected_normals[p : 2 * p])
        assert jnp.allclose(computed_normals[p : 2 * p], expected_normals[p : 2 * p])

        # N side
        print("test_1: computed_normals[2*p:3*p] = ", computed_normals[2 * p : 3 * p])
        print("test_1: expected_normals[2*p:3*p] = ", expected_normals[2 * p : 3 * p])
        assert jnp.allclose(
            computed_normals[2 * p : 3 * p], expected_normals[2 * p : 3 * p]
        )

        # W side
        print("test_1: computed_normals[3*p:] = ", computed_normals[3 * p :])
        print("test_1: expected_normals[3*p:] = ", expected_normals[3 * p :])
        assert jnp.allclose(computed_normals[3 * p :], expected_normals[3 * p :])

        assert jnp.allclose(computed_normals, expected_normals)


class Test_precompute_N_tilde_matrix:
    def test_0(self) -> None:
        """Check the shape of the output."""
        p = 8
        du_dx, du_dy, _, _, _ = precompute_diff_operators(p, 1.0)
        out = precompute_N_tilde_matrix(du_dx, du_dy, p)
        assert out.shape == (4 * (p - 1), p**2)

    def test_1(self) -> None:
        """Check that low-degree polynomials are handled correctly."""

        p = 8
        q = 6
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2
        root = Node(xmin=west, xmax=east, ymin=south, ymax=north)

        half_side_len = jnp.pi / 2
        du_dx, du_dy, _, _, _ = precompute_diff_operators(p, half_side_len)
        out = precompute_N_tilde_matrix(du_dx, du_dy, p)

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -3
            return -3 * jnp.ones_like(x[..., 1])

        # Set up the Chebyshev points.
        cheby_pts = get_all_leaf_2d_cheby_points(p, root)[0]
        cheby_bdry = cheby_pts[: 4 * (p - 1)]

        f_evals = f(cheby_pts)
        computed_normals = out @ f_evals

        expected_normals = jnp.concatenate(
            [
                -1 * dfdy(cheby_bdry[: p - 1]),
                dfdx(cheby_bdry[p - 1 : 2 * (p - 1)]),
                dfdy(cheby_bdry[2 * (p - 1) : 3 * (p - 1)]),
                -1 * dfdx(cheby_bdry[3 * (p - 1) :]),
            ]
        )

        print("test_1: computed_normals shape = ", computed_normals.shape)
        print("test_1: expected_normals shape = ", expected_normals.shape)
        assert jnp.allclose(computed_normals, expected_normals)


class Test_precompute_G_matrix:
    def test_0(self) -> None:
        """Check the shape of the output."""
        p = 8
        du_dx, du_dy, _, _, _ = precompute_diff_operators(p, 1.0)
        N = precompute_N_matrix(du_dx, du_dy, p)
        out = precompute_G_matrix(N, p, 4.0)
        assert out.shape == (4 * p, p**2)

    def test_1(self) -> None:
        """Check that low-degree polynomials are handled correctly."""
        p = 8
        q = 6
        north = jnp.pi / 2
        south = -jnp.pi / 2
        east = jnp.pi / 2
        west = -jnp.pi / 2
        root = Node(xmin=west, xmax=east, ymin=south, ymax=north)
        half_side_len = jnp.pi / 2
        du_dx, du_dy, _, _, _ = precompute_diff_operators(p, half_side_len)
        N = precompute_N_matrix(du_dx, du_dy, p)
        eta = 4.0
        out = precompute_G_matrix(N, p, eta)

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -3
            return -3 * jnp.ones_like(x[..., 1])

        # Set up the Chebyshev points.
        cheby_pts = get_all_leaf_2d_cheby_points(p, root)
        cheby_bdry = cheby_pts[0, : 4 * (p - 1)]
        all_pts = cheby_pts[0]

        f_evals = f(all_pts)
        computed_out_imp = out @ f_evals

        f_normals = jnp.concatenate(
            [
                -1 * dfdy(cheby_bdry[:p]),
                dfdx(cheby_bdry[p - 1 : 2 * p - 1]),
                dfdy(cheby_bdry[2 * p - 2 : 3 * p - 2]),
                -1 * dfdx(cheby_bdry[3 * p - 3 :]),
                -1 * dfdx(cheby_bdry[0]).reshape(1),
            ]
        )
        f_evals = jnp.concatenate(
            [
                f(cheby_bdry[:p]),
                f(cheby_bdry[p - 1 : 2 * p - 1]),
                f(cheby_bdry[2 * p - 2 : 3 * p - 2]),
                f(cheby_bdry[3 * p - 3 :]),
                f(cheby_bdry[0]).reshape(1),
            ]
        )

        expected_out_imp = f_normals - 1j * eta * f_evals

        assert jnp.allclose(computed_out_imp, expected_out_imp)


class Test_precompute_F_matrix:
    def test_0(self) -> None:
        """Check the shape of the output."""
        p = 8
        du_dx, du_dy, _, _, _ = precompute_diff_operators(p, 1.0)
        N_tilde = precompute_N_tilde_matrix(du_dx, du_dy, p)
        out = precompute_F_matrix(N_tilde, p, 4.0)
        assert out.shape == (4 * (p - 1), p**2)

    def test_1(self) -> None:
        """Checks correctness for low-degree polynomials."""
        p = 8
        print("test_1: p = ", p)
        print("test_1: 4(p-1) = ", 4 * (p - 1))
        du_dx, du_dy, _, _, _ = precompute_diff_operators(p, 0.5)
        N_tilde = precompute_N_tilde_matrix(du_dx, du_dy, p)

        # Corners for a square of side length 1.0
        north = 0.5
        south = -0.5
        east = 0.5
        west = -0.5
        root = Node(xmin=west, xmax=east, ymin=south, ymax=north)
        pts = get_all_leaf_2d_cheby_points(p, root)

        def f(x: jnp.array) -> jnp.array:
            # f(x,y) = x^2 - 3y
            return x[..., 0] ** 2 - 3 * x[..., 1]

        def dfdx(x: jnp.array) -> jnp.array:
            # df/dx = 2x
            return 2 * x[..., 0]

        def dfdy(x: jnp.array) -> jnp.array:
            # df/dy = -3
            return -3 * jnp.ones_like(x[..., 1])

        eta = 4.0

        f_evals = f(pts[0])
        print("test_1: f_evals shape = ", f_evals.shape)
        F = precompute_F_matrix(N_tilde, p, eta)
        print("test_1: F shape = ", F.shape)

        computed_inc_imp = F @ f_evals
        expected_bdry_normals = jnp.concatenate(
            [
                -1 * dfdy(pts[0][: p - 1]),
                dfdx(pts[0][p - 1 : 2 * p - 2]),
                dfdy(pts[0][2 * p - 2 : 3 * p - 3]),
                -1 * dfdx(pts[0][3 * p - 3 : 4 * (p - 1)]),
            ]
        )
        expected_bdry_f = f(pts[0][: 4 * (p - 1)])
        print("test_1: expected_bdry_normals shape = ", expected_bdry_normals.shape)
        print("test_1: expected_bdry_f shape = ", expected_bdry_f.shape)
        expected_inc_imp = expected_bdry_normals + 1j * eta * expected_bdry_f

        # plt.plot(computed_inc_imp.real, "o-", label="computed_inc_imp.real")
        # plt.plot(expected_inc_imp.real, "x-", label="expected_inc_imp.real")
        # plt.plot(computed_inc_imp.imag, "o-", label="computed_inc_imp.imag")
        # plt.plot(expected_inc_imp.imag, "x-", label="expected_inc_imp.imag")
        # plt.legend()
        # plt.show()

        assert jnp.allclose(computed_inc_imp, expected_inc_imp)


if __name__ == "__main__":
    pytest.main()
