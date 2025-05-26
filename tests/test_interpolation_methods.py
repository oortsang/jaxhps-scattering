import jax.numpy as jnp
import jax
import pytest
from jaxhps._interpolation_methods import (
    interp_from_hps_2D,
    interp_from_hps_3D,
    interp_to_single_Chebyshev_panel_2D,
    interp_to_single_Chebyshev_panel_3D,
)
from jaxhps._discretization_tree import (
    DiscretizationNode2D,
    DiscretizationNode3D,
    get_all_leaves,
)
from jaxhps._discretization_tree_operations_2D import add_four_children
from jaxhps._discretization_tree_operations_3D import add_eight_children
from jaxhps._grid_creation_2D import (
    compute_interior_Chebyshev_points_adaptive_2D,
    compute_interior_Chebyshev_points_uniform_2D,
    get_all_uniform_leaves_2D,
)
from jaxhps._grid_creation_3D import (
    compute_interior_Chebyshev_points_adaptive_3D,
    compute_interior_Chebyshev_points_uniform_3D,
    get_all_uniform_leaves_3D,
)
import logging

# Turn off jax logging
logging.getLogger("jax").setLevel(logging.WARNING)


class Test_interp_from_hps_2D:
    def test_0(self) -> None:
        # Make sure returns correct shape. Uniform grid with 2 levels of refinement.
        l = 2
        p = 10
        n_x = 7

        xmin = -1.0
        xmax = 1.0
        ymin = 3.0
        ymax = 4.0

        root = DiscretizationNode2D(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

        leaves = get_all_uniform_leaves_2D(root, l)

        x_vals = jnp.linspace(xmin, xmax, n_x)
        y_vals = jnp.linspace(ymin, ymax, n_x)

        f_evals = jnp.ones((4**l, p**2))

        out_0, out_1 = interp_from_hps_2D(leaves, p, f_evals, x_vals, y_vals)

        assert out_0.shape == (n_x, n_x)
        assert not jnp.any(jnp.isnan(out_0))
        assert not jnp.any(jnp.isinf(out_0))

        assert out_1.shape == (n_x, n_x, 2)
        assert jnp.all(out_1[..., 0] >= xmin)
        assert jnp.all(out_1[..., 0] <= xmax)
        assert jnp.all(out_1[..., 1] >= ymin)
        assert jnp.all(out_1[..., 1] <= ymax)

    def test_1(self) -> None:
        """Check that low-degree polynomial interpolation is exact on a non-uniform HPS grid."""
        p = 6
        q = 4
        n_x = 3

        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0

        x_vals = jnp.linspace(xmin, xmax, n_x)
        y_vals = jnp.linspace(ymin, ymax, n_x)

        expected_target_pts_X, expected_target_pts_Y = jnp.meshgrid(
            x_vals, y_vals
        )

        root = DiscretizationNode2D(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)
        add_four_children(root, root=root, q=q)
        for c in root.children:
            add_four_children(c, root=root, q=q)
            # for cc in c.children:
            #     add_four_children(cc, root=root, q=q)
        add_four_children(root.children[0].children[0], root=root, q=q)

        # Generate chebyshev grid

        hps_grid_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)

        def f(x: jnp.array) -> jnp.array:
            # x has shape (..., 2)
            # f(x,y) = 3y - 4x^2
            return 3 * x[..., 1] - 4 * x[..., 0] ** 2

        f_evals = f(hps_grid_pts)

        leaves = get_all_leaves(root)
        interp_vals, target_pts = interp_from_hps_2D(
            leaves=leaves, p=p, f_evals=f_evals, x_vals=x_vals, y_vals=y_vals
        )

        assert jnp.allclose(target_pts[..., 0], expected_target_pts_X)
        assert jnp.allclose(target_pts[..., 1], expected_target_pts_Y)

        f_target = f(target_pts)

        print("test_1: f_target: ", f_target)
        print("test_1: interp_vals: ", interp_vals)
        diffs = interp_vals - f_target

        print("test_1: diffs : ", diffs)
        assert jnp.allclose(interp_vals, f_target)

    def test_2(self) -> None:
        """Check that low-degree polynomial interpolation is exact on a uniform HPS grid."""
        p = 6
        l = 2
        n_x = 3

        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0

        x_vals = jnp.linspace(xmin, xmax, n_x)
        y_vals = jnp.linspace(ymin, ymax, n_x)

        expected_target_pts_X, expected_target_pts_Y = jnp.meshgrid(
            x_vals, y_vals
        )

        root = DiscretizationNode2D(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

        leaves = get_all_uniform_leaves_2D(root, L=l)

        # Generate chebyshev grid

        hps_grid_pts = compute_interior_Chebyshev_points_uniform_2D(root, l, p)

        def f(x: jnp.array) -> jnp.array:
            # x has shape (..., 2)
            # f(x,y) = 3y - 4x^2
            return 3 * x[..., 1] - 4 * x[..., 0] ** 2

        f_evals = f(hps_grid_pts)

        interp_vals, target_pts = interp_from_hps_2D(
            leaves=leaves, p=p, f_evals=f_evals, x_vals=x_vals, y_vals=y_vals
        )

        assert jnp.allclose(target_pts[..., 0], expected_target_pts_X)
        assert jnp.allclose(target_pts[..., 1], expected_target_pts_Y)

        f_target = f(target_pts)

        print("test_1: f_target: ", f_target)
        print("test_1: interp_vals: ", interp_vals)
        diffs = interp_vals - f_target

        print("test_1: diffs : ", diffs)
        assert jnp.allclose(interp_vals, f_target)

    @pytest.mark.skip("Super slow test.")
    def test_3(self) -> None:
        """Check the accuracy of interpolating a plane wave with wavenumber 100 over-resolved to 50 points per wavelength, and
        then interpolated to a 500x500 grid."""
        l = 8
        p = 20
        n_x = 500

        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0

        x_vals = jnp.linspace(xmin, xmax, n_x)
        y_vals = jnp.linspace(ymin, ymax, n_x)
        # expected_target_pts_X, expected_target_pts_Y = jnp.meshgrid(
        #     x_vals, y_vals
        # )

        root = DiscretizationNode2D(xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax)

        # Generate chebyshev grid

        hps_grid_pts = compute_interior_Chebyshev_points_uniform_2D(
            root=root, L=l, p=p
        )
        k = 100.0

        def f(x: jnp.array) -> jnp.array:
            # x has shape (..., 2)
            # f(x,y) = exp(i * k * 2 pi * x)
            exponent = 1j * k * 2 * jnp.pi * x[..., 0]
            return jnp.exp(exponent)

        f_evals = f(hps_grid_pts)

        leaves = get_all_uniform_leaves_2D(root, L=l)

        interp_vals, target_pts = interp_from_hps_2D(
            leaves=leaves, p=p, f_evals=f_evals, x_vals=x_vals, y_vals=y_vals
        )

        f_target = f(target_pts)

        # print("test_1: f_target: ", f_target)
        # print("test_1: interp_vals: ", interp_vals)
        diffs = interp_vals - f_target

        print("test_1: max diffs : ", jnp.max(jnp.abs(diffs)))
        assert jnp.allclose(interp_vals, f_target)


class Test_interp_from_hps_3D:
    def test_0(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )
        add_eight_children(root)

        leaves = get_all_leaves(root)
        p = 4
        hps_pts = compute_interior_Chebyshev_points_adaptive_3D(root, p)
        f_evals = jnp.ones_like(hps_pts[..., 0])

        n_x = 4
        x_vals = jnp.linspace(0.0, 1.0, n_x)
        y_vals = jnp.linspace(0.0, 1.0, n_x)
        z_vals = jnp.linspace(0.0, 1.0, n_x)
        out_0, out_1 = interp_from_hps_3D(
            leaves=leaves,
            p=p,
            f_evals=f_evals,
            x_vals=x_vals,
            y_vals=y_vals,
            z_vals=z_vals,
        )
        assert out_0.shape == (n_x, n_x, n_x)
        assert not jnp.any(jnp.isnan(out_0))
        assert not jnp.any(jnp.isinf(out_0))

        X, Y, Z = jnp.meshgrid(x_vals, y_vals, z_vals)
        logging.debug("test_0: X shape: %s", X.shape)
        logging.debug("test_0: out_1 shape: %s", out_1.shape)
        assert jnp.allclose(out_1[..., 0], X)
        assert jnp.allclose(out_1[..., 1], Y)
        assert jnp.allclose(out_1[..., 2], Z)

    def test_1(self, caplog) -> None:
        """Test low-degree polynomial interpolation is exact on a non-uniform grid."""

        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )
        add_eight_children(root)

        leaves = get_all_leaves(root)
        p = 6

        def f(x: jax.Array) -> jax.Array:
            # x has shape (..., 3)
            # f(x,y,z) = 3y - 4x^2 + z
            return 3 * x[..., 1] - 4 * x[..., 0] ** 2 + x[..., 2]

        # Generate chebyshev grid
        hps_points = compute_interior_Chebyshev_points_adaptive_3D(root, p)
        f_evals = f(hps_points)
        n_x = 3
        x_vals = jnp.linspace(0.0, 1.0, n_x)
        y_vals = jnp.linspace(0.0, 1.0, n_x)
        z_vals = jnp.linspace(0.0, 1.0, n_x)
        f_interp, target_pts = interp_from_hps_3D(
            leaves=leaves,
            p=p,
            f_evals=f_evals,
            x_vals=x_vals,
            y_vals=y_vals,
            z_vals=z_vals,
        )

        f_target = f(target_pts)
        assert jnp.allclose(f_interp, f_target)

    def test_2(self, caplog) -> None:
        """Test low-degree polynomial interpolation is exact on a uniform grid."""

        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )
        L = 2

        leaves = get_all_uniform_leaves_3D(root, L=L)
        p = 6

        def f(x: jax.Array) -> jax.Array:
            # x has shape (..., 3)
            # f(x,y,z) = 3y - 4x^2 + z
            return 3 * x[..., 1] - 4 * x[..., 0] ** 2 + x[..., 2]

        # Generate chebyshev grid
        hps_points = compute_interior_Chebyshev_points_uniform_3D(root, L, p)
        f_evals = f(hps_points)
        n_x = 3
        x_vals = jnp.linspace(0.0, 1.0, n_x)
        y_vals = jnp.linspace(0.0, 1.0, n_x)
        z_vals = jnp.linspace(0.0, 1.0, n_x)
        f_interp, target_pts = interp_from_hps_3D(
            leaves=leaves,
            p=p,
            f_evals=f_evals,
            x_vals=x_vals,
            y_vals=y_vals,
            z_vals=z_vals,
        )

        f_target = f(target_pts)
        assert jnp.allclose(f_interp, f_target)


class Test_interp_to_single_Chebyshev_panel_2D:
    def test_0(self) -> None:
        """Check that interpolation to a single Chebyshev panel is correct."""
        p = 4
        n_x = 5

        node_bounds = jnp.array([0.0, 1.0, 0.0, 1.0])
        samples = jnp.ones((n_x, n_x), dtype=jnp.float64)

        x_vals = jnp.linspace(0.0, 1.0, n_x)
        y_vals = jnp.linspace(0.0, 1.0, n_x)

        out = interp_to_single_Chebyshev_panel_2D(
            node_bounds=node_bounds,
            p=p,
            from_x=x_vals,
            from_y=y_vals,
            samples=samples,
        )
        assert out.shape == (p**2,)


class Test_interp_to_single_Chebyshev_panel_3D:
    def test_0(self) -> None:
        """Check that interpolation to a single Chebyshev panel is correct."""
        p = 4
        n_x = 5

        node_bounds = jnp.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
        samples = jnp.ones((n_x, n_x, n_x), dtype=jnp.float64)

        x_vals = jnp.linspace(0.0, 1.0, n_x)
        y_vals = jnp.linspace(0.0, 1.0, n_x)
        z_vals = jnp.linspace(0.0, 1.0, n_x)

        out = interp_to_single_Chebyshev_panel_3D(
            node_bounds=node_bounds,
            p=p,
            from_x=x_vals,
            from_y=y_vals,
            from_z=z_vals,
            samples=samples,
        )
        assert out.shape == (p**3,)
