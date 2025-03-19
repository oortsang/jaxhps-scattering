import jax.numpy as jnp
import jax
import numpy as np
import pytest

from hps.src.quadrature.quad_2D.grid_creation import (
    chebyshev_points,
    bounds_to_cheby_points_lst,
    vmapped_corners,
    vmapped_bounds_to_cheby_points_lst,
    get_all_boundary_gauss_legendre_points,
    get_all_leaf_2d_cheby_points,
    _get_next_S_boundary_node,
)
from hps.src.quadrature.trees import (
    Node,
    tree_equal,
    add_four_children,
    find_node_at_corner,
    get_all_leaves,
    _corners_for_quad_subdivision,

)
from hps.src.test_utils import check_arrays_close


class Test_get_all_boundary_gauss_legendre_points:
    def test_0(self) -> None:
        """Checks output shapes are correct on uniform refinement of 2 levels."""
        p = 16
        q = 14
        root = Node(
            xmin=-1.0, xmax=1.0, ymin=-1.0, ymax=1.0, zmin=None, zmax=None, depth=0
        )

        add_four_children(root)
        for c in root.children:
            add_four_children(c)

        pts = get_all_boundary_gauss_legendre_points(q, root)

        assert pts.shape == (16 * q, 2)

    def test_1(self) -> None:
        """Checks that outputs pass basic sanity checks on uniform refinement of 3 levels."""
        p = 16
        q = 14
        corners = jnp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        west, south = corners[0]
        east, north = corners[2]
        root = Node(
            xmin=west, xmax=east, ymin=south, ymax=north, depth=0, zmin=None, zmax=None
        )
        add_four_children(root)
        for c in root.children:
            add_four_children(c)
            for gc in c.children:
                add_four_children(gc)

        y = get_all_boundary_gauss_legendre_points(q, root)

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


class Test_get_all_leaf_2d_cheby_points:
    def test_0(self) -> None:
        """Check that output shapes are correct when looking at 3 levels of uniform refinement."""
        p = 16
        west = -1
        south = -1
        east = 1
        north = 1
        root = Node(
            xmin=west, xmax=east, ymin=south, ymax=north, depth=0, zmin=None, zmax=None
        )
        add_four_children(root)
        for c in root.children:
            add_four_children(c)
            for gc in c.children:
                add_four_children(gc)

        x = get_all_leaf_2d_cheby_points(p, root)

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

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=None, zmax=None, depth=0
        )
        add_four_children(root)
        add_four_children(root.children[0])

        x = get_all_leaf_2d_cheby_points(p, root)

        for i, leaf in enumerate(get_all_leaves(root)):
            assert jnp.all(x[i, :, 0] <= leaf.xmax)
            assert jnp.all(x[i, :, 0] >= leaf.xmin)
            assert jnp.all(x[i, :, 1] <= leaf.ymax)
            assert jnp.all(x[i, :, 1] >= leaf.ymin)


class Test_bounds_to_cheby_points_lst:
    def test_0(self) -> None:
        p = 16
        q = 14
        node = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=None, zmax=None, depth=0
        )
        bounds = jnp.array([node.xmin, node.xmax, node.ymin, node.ymax])

        cheby_nodes = chebyshev_points(p)[0]

        x = bounds_to_cheby_points_lst(bounds, cheby_nodes)
        assert x.shape == (p**2, 2)


class Test_vmapped_corners:
    def test_0(self) -> None:
        corners_0 = jnp.array([[-1, -1], [1, -1], [1, 1], [-1, 1]])
        corners_0 = jnp.expand_dims(corners_0, axis=0)

        print("test_0: corners_0.shape: ", corners_0.shape)

        corners_1 = vmapped_corners(corners_0)

        print("test_0: corners_1.shape: ", corners_1.shape)


class Test__corners_for_quad_subdivision:
    def test_0(self) -> None:
        corners = [(0, 0), (1, 0), (1, 1), (0, 1)]
        new_corners = _corners_for_quad_subdivision(corners)

        expected_corners = jnp.array(
            [
                # SW quadrant
                [(0, 0), (0.5, 0), (0.5, 0.5), (0, 0.5)],
                # SE quadrant
                [(0.5, 0), (1, 0), (1, 0.5), (0.5, 0.5)],
                # NE quadrant
                [(0.5, 0.5), (1, 0.5), (1, 1), (0.5, 1)],
                # NW quadrant
                [(0, 0.5), (0.5, 0.5), (0.5, 1), (0, 1)],
            ]
        )
        assert jnp.all(new_corners == expected_corners)


class Test__get_next_S_boundary_node:
    def test_0(self) -> None:
        node = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        add_four_children(node)
        add_four_children(node.children[0])
        add_four_children(node.children[0].children[0])

        corner = find_node_at_corner(node, xmin=node.xmin, ymin=node.ymin)
        s_bdry = _get_next_S_boundary_node(node, (corner,))

        expected_s_bdry = (
            node.children[0].children[0].children[0],
            node.children[0].children[0].children[1],
            node.children[0].children[1],
            node.children[1],
        )
        print("test_0: s_bdry len: ", len(s_bdry))
        print("test_0: expected_s_bdry len: ", len(expected_s_bdry))
        for s, e in zip(s_bdry, expected_s_bdry):
            assert tree_equal(s, e)


if __name__ == "__main__":

    pytest.main()
