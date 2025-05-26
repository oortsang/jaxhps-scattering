from jaxhps._grid_creation_2D import (
    compute_boundary_Gauss_points_uniform_2D,
    compute_boundary_Gauss_points_adaptive_2D,
    compute_interior_Chebyshev_points_adaptive_2D,
    vmapped_bounds_2D,
    bounds_to_cheby_points_2D,
)
from jaxhps._discretization_tree import (
    DiscretizationNode2D,
    get_all_leaves,
)
from jaxhps._discretization_tree_operations_2D import (
    add_four_children,
)
from jaxhps.quadrature import chebyshev_points
import jax.numpy as jnp


class Test_bounds_to_cheby_points_2D:
    def test_0(self) -> None:
        p = 16
        # q = 14
        xmin = 0.0
        xmax = 1.0
        ymin = 0.0
        ymax = 1.0
        bounds = jnp.array([xmin, xmax, ymin, ymax])

        cheby_nodes = chebyshev_points(p)

        x = bounds_to_cheby_points_2D(bounds, cheby_nodes)
        assert x.shape == (p**2, 2)

    def test_1(self) -> None:
        p = 16
        # q = 14
        xmin = -1.0
        xmax = 1.0
        ymin = -1.0
        ymax = 1.0
        bounds = jnp.array([xmin, xmax, ymin, ymax])

        cheby_nodes = chebyshev_points(p)

        x = bounds_to_cheby_points_2D(bounds, cheby_nodes)
        assert x.shape == (p**2, 2)
        n_cheby_bdry = 4 * (p - 1)
        # Check that the first n_cheby_bdry points are on the boundary
        assert jnp.max(jnp.abs(x[:n_cheby_bdry, 0])) == 1.0
        assert jnp.max(jnp.abs(x[:n_cheby_bdry, 1])) == 1.0

        # For the rest, check that the abs value is less than 1
        assert jnp.all(jnp.abs(x[n_cheby_bdry:, 0]) < 1)
        assert jnp.all(jnp.abs(x[n_cheby_bdry:, 1]) < 1)


class Test_vmapped_bounds_2D:
    def test_0(self) -> None:
        corners_0 = jnp.array([[-1.0, 0.0, -1.0, 0.0]])

        print("test_0: corners_0.shape: ", corners_0.shape)

        corners_1 = vmapped_bounds_2D(corners_0)
        assert corners_1.shape == (1, 4, 4)

        corners_2 = vmapped_bounds_2D(corners_1.reshape(-1, 4))
        assert corners_2.shape == (4, 4, 4)


class Test_compute_boundary_Gauss_points_uniform_2D:
    def test_0(self) -> None:
        """Checks output shapes are correct on uniform refinement of 2 levels."""
        # p = 16
        q = 8
        L = 2

        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        pts = compute_boundary_Gauss_points_uniform_2D(root, L, q)

        assert pts.shape == (16 * q, 2)

    def test_1(self) -> None:
        """Checks that outputs pass basic sanity checks on uniform refinement of 3 levels."""
        # p = 16
        q = 6
        L = 3
        corners = jnp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        west, south = corners[0]
        east, north = corners[2]
        root = DiscretizationNode2D(
            xmin=west,
            xmax=east,
            ymin=south,
            ymax=north,
        )
        y = compute_boundary_Gauss_points_uniform_2D(root, L, q)

        # Check that the gauss points are constant along the boundaries
        n_per_side = y.shape[0] // 4
        S = y[:n_per_side]
        E = y[n_per_side : 2 * n_per_side]
        N = y[2 * n_per_side : 3 * n_per_side]
        W = y[3 * n_per_side :]
        assert jnp.all(S[:, 1] == south)
        assert jnp.all(E[:, 0] == east)
        assert jnp.all(N[:, 1] == north)
        assert jnp.all(W[:, 0] == west)

        # Check that the gauss points are monotonically increasing/decreasing
        assert jnp.all(S[1:, 0] > S[:-1, 0])
        assert jnp.all(E[1:, 1] > E[:-1, 1])
        assert jnp.all(N[1:, 0] < N[:-1, 0])
        assert jnp.all(W[1:, 1] < W[:-1, 1])


class Test_compute_boundary_Gauss_points_adaptive_2D:
    def test_0(self) -> None:
        """Checks output shapes are correct on uniform refinement of 2 levels."""
        # p = 16
        q = 4
        root = DiscretizationNode2D(
            xmin=-1.0,
            xmax=1.0,
            ymin=-1.0,
            ymax=1.0,
        )

        add_four_children(root, root=root, q=q)
        for c in root.children:
            add_four_children(c, root=root, q=q)
        add_four_children(root.children[0].children[0], root=root, q=q)

        pts = compute_boundary_Gauss_points_adaptive_2D(root=root, q=q)

        n_expected_panels = 18
        assert pts.shape == (n_expected_panels * q, 2)

    def test_1(self) -> None:
        """Checks that outputs pass basic sanity checks on uniform refinement of 3 levels."""
        # p = 16
        q = 6
        west = -1.0
        east = 1.0
        south = -1.0
        north = 1.0
        root = DiscretizationNode2D(
            xmin=west,
            xmax=east,
            ymin=south,
            ymax=north,
        )
        add_four_children(root, root=root, q=q)
        for c in root.children:
            add_four_children(c, root=root, q=q)
            for gc in c.children:
                add_four_children(gc, root=root, q=q)

        y = compute_boundary_Gauss_points_adaptive_2D(root, q)

        # Check that the gauss points are constant along the boundaries
        n_per_side = y.shape[0] // 4
        S = y[:n_per_side]
        E = y[n_per_side : 2 * n_per_side]
        N = y[2 * n_per_side : 3 * n_per_side]
        W = y[3 * n_per_side :]
        assert jnp.all(S[:, 1] == south)
        assert jnp.all(E[:, 0] == east)
        assert jnp.all(N[:, 1] == north)
        assert jnp.all(W[:, 0] == west)

        # Check that the gauss points are monotonically increasing/decreasing
        assert jnp.all(S[1:, 0] > S[:-1, 0])
        assert jnp.all(E[1:, 1] > E[:-1, 1])
        assert jnp.all(N[1:, 0] < N[:-1, 0])
        assert jnp.all(W[1:, 1] < W[:-1, 1])


class Test_compute_interior_Chebyshev_points_adaptive_2D:
    def test_0(self) -> None:
        """Check that output shapes are correct when looking at 3 levels of uniform refinement."""
        p = 16
        q = p - 2
        west = -1
        south = -1
        east = 1
        north = 1
        root = DiscretizationNode2D(
            xmin=west,
            xmax=east,
            ymin=south,
            ymax=north,
        )
        add_four_children(root, root=root, q=q)
        for c in root.children:
            add_four_children(c, root=root, q=q)
            for gc in c.children:
                add_four_children(gc, root=root, q=q)

        x = compute_interior_Chebyshev_points_adaptive_2D(root, p)

        # Check the shape is correct
        assert x.shape == (4**3, p**2, 2)
        # Check that the cheby points lie inside the corners
        assert jnp.all(x[:, :, 0] >= west)
        assert jnp.all(x[:, :, 0] <= east)
        assert jnp.all(x[:, :, 1] >= south)
        assert jnp.all(x[:, :, 1] <= north)

    def test_1(self) -> None:
        """Check that the output array is ordered in the same way that get_all_leaves() orders leaves"""

        p = 8

        root = DiscretizationNode2D(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
        )
        add_four_children(root)
        add_four_children(root.children[0])

        x = compute_interior_Chebyshev_points_adaptive_2D(root, p)

        for i, leaf in enumerate(get_all_leaves(root)):
            assert jnp.all(x[i, :, 0] <= leaf.xmax)
            assert jnp.all(x[i, :, 0] >= leaf.xmin)
            assert jnp.all(x[i, :, 1] <= leaf.ymax)
            assert jnp.all(x[i, :, 1] >= leaf.ymin)
