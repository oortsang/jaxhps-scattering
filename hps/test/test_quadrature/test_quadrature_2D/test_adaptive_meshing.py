from copy import deepcopy
import jax.numpy as jnp
import numpy as np

from hps.src.quadrature.quad_2D.interpolation import refinement_operator
from hps.src.quadrature.quad_2D.grid_creation import get_all_leaf_2d_cheby_points
from hps.src.quadrature.quad_2D.adaptive_meshing import (
    generate_adaptive_mesh_l2,
    find_or_add_child,
    get_squared_l2_norm_single_panel,
    check_current_discretization_relative_global_l2_norm,
    node_corners_to_2d_corners,
)
from hps.src.quadrature.trees import (
    Node,
    get_all_leaves,
    add_four_children,
    tree_equal,
)


class Test_get_squared_l2_norm_single_panel:
    def test_0(self) -> None:
        """Make sure things run without error."""
        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=None, zmax=None, depth=0
        )

        p = 4

        f_evals = np.random.normal(size=(p**2))

        x = get_squared_l2_norm_single_panel(
            f_evals, node_corners_to_2d_corners(root), p
        )

        print("test_0: x: ", x)
        assert not np.isnan(x)
        assert not np.isinf(x)

    def test_1(self) -> None:
        """Make sure things are correct for a constant function."""
        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=None, zmax=None, depth=0
        )
        p = 4

        f_evals = 3 * np.ones((p**2))
        x = get_squared_l2_norm_single_panel(
            f_evals, node_corners_to_2d_corners(root), p
        )

        print("test_1: x: ", x)
        assert np.isclose(x, 9.0)

    def test_2(self) -> None:
        """Make sure things are correct for a low-degree polynomial."""
        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=None, zmax=None, depth=0
        )

        p = 16

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + x**2"""
            return x[..., 1] + x[..., 0] ** 2

        cheby_pts = get_all_leaf_2d_cheby_points(p, root)
        f_evals = f(cheby_pts).flatten()

        x = get_squared_l2_norm_single_panel(
            f_evals, node_corners_to_2d_corners(root), p
        )

        print("test_2: x: ", x)

        # Norm of f(x,y) = y + x**2 over [0,1]x[0,1] is sqrt(13/15)
        expected_x = 13 / 15
        print("test_2: expected_x: ", expected_x)
        assert np.isclose(x, expected_x)

    def test_3(self) -> None:
        """Constant function with side length pi"""
        root = Node(
            xmin=-np.pi / 2,
            xmax=np.pi / 2,
            ymin=-np.pi / 2,
            ymax=np.pi / 2,
            zmin=None,
            zmax=None,
            depth=0,
        )

        p = 16
        f_evals = 3 * np.ones((p**2))
        expected_x = 9 * np.pi**2
        x = get_squared_l2_norm_single_panel(
            f_evals, node_corners_to_2d_corners(root), p
        )
        print("test_3: x: ", x)
        print("test_3: expected_x: ", expected_x)
        assert np.isclose(x, expected_x)


class Test_generate_adaptive_mesh_l2:
    def test_0(self) -> None:
        """Makes sure the shapes are right"""

        p = 4
        q = 2
        refinement_op = refinement_operator(p)

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + x**2"""
            return x[..., 1] + x[..., 0] ** 2

        tol = 1e-3

        node = Node(
            xmin=0.0,
            xmax=1.0,
            ymin=0.0,
            ymax=1.0,
            zmin=None,
            zmax=None,
            depth=0,
        )
        add_four_children(add_to=node, root=node, q=q)

        generate_adaptive_mesh_l2(
            root=node, refinement_op=refinement_op, f_fn=f, tol=tol, p=p, q=q
        )
        leaves_iter = get_all_leaves(node)
        assert len(leaves_iter) == 4, len(leaves_iter)
        for leaf in leaves_iter:
            assert leaf.depth == 1, leaf.depth
        # The example is a low-degree polynomial, which should be exact for 1 level of refinement.


class Test_find_or_add_child:

    def test_0(self) -> None:
        """Test case from a bug observed while plotting"""
        xmin = -1
        xmax = -0.75
        ymin = -0.5
        ymax = -0.25

        q = 2

        node = Node(xmin=-1, xmax=0.0, ymin=-1, ymax=0.0, zmin=None, zmax=None, depth=0)
        add_four_children(add_to=node, root=node, q=2)
        add_four_children(add_to=node.children[0], root=node, q=2)
        node_cp = deepcopy(node)

        find_or_add_child(
            node=node, root=node, q=q, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax
        )
        print("test_0: node: ", node)
        leaves = get_all_leaves(node)
        print("test_0: leaves: ", leaves)
        assert not tree_equal(node, node_cp)


class Test_check_current_discretization_relative_global_l2_norm:
    def test_0(self) -> None:
        p = 4
        refinement_op = refinement_operator(p)

        f_evals = np.random.normal(size=(p**2))
        f_evals_refined = np.random.normal(size=(4 * p**2))
        tol = 1e-03
        global_l2_norm = 100.0
        node = Node(xmin=0, xmax=1, ymin=0, ymax=1, zmin=None, zmax=None, depth=0)
        z = check_current_discretization_relative_global_l2_norm(
            f_evals,
            f_evals_refined,
            refinement_op,
            tol,
            global_l2_norm,
            node_corners_to_2d_corners(node),
            p,
        )
        print("test_0: z: ", z)
