import pytest
import jax.numpy as jnp
import numpy as np

from hps.src.quadrature.quad_3D.adaptive_meshing import (
    generate_adaptive_mesh_level_restriction,
    get_squared_l2_norm_single_voxel,
    node_corners_to_3d_corners,
)

from hps.src.quadrature.trees import (
    Node,
    add_eight_children,
    get_all_leaves,
)
from hps.src.quadrature.quad_3D.interpolation import refinement_operator
from hps.src.quadrature.quad_3D.grid_creation import (
    get_all_leaf_3d_cheby_points,
)


class Test_generate_adaptive_mesh_level_restriction:
    def test_0(self) -> None:
        """Make sure things run without error"""

        p = 4
        tol = 1e-03
        refinement_op = refinement_operator(p)

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        add_eight_children(root)

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + x**2"""
            return x[..., 1] + x[..., 0] ** 2

        generate_adaptive_mesh_level_restriction(
            root, refinement_op, f, tol, p, q=p - 2
        )
        leaves_iter = get_all_leaves(root)

        # We don't expect the level restriction to add any nodes, because f is a low-
        # degree polynomial which should be resolved by our mesh.
        assert len(leaves_iter) == 8, len(leaves_iter)
        for leaf in leaves_iter:
            assert leaf.depth == 1, leaf.depth

    def test_1(self) -> None:
        """Make sure things run without error"""

        p = 6
        tol = 1e-03
        refinement_op = refinement_operator(p)

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        add_eight_children(root)

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + x**2"""
            return x[..., 1] + x[..., 0] ** 2

        generate_adaptive_mesh_level_restriction(
            root, refinement_op, f, tol, p, q=p - 2, l2_norm=True
        )
        leaves_iter = get_all_leaves(root)

        # We don't expect the level restriction to add any nodes, because f is a low-
        # degree polynomial which should be resolved by our mesh.
        assert len(leaves_iter) == 8, len(leaves_iter)
        for leaf in leaves_iter:
            assert leaf.depth == 1, leaf.depth


class Test_get_squared_l2_norm_single_voxel:
    def test_0(self) -> None:
        """Make sure things run without error."""
        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        p = 10

        f_evals = np.random.normal(size=(p**3))

        corners = jnp.array(
            [
                [root.xmin, root.ymin, root.zmin],
                [root.xmax, root.ymax, root.zmax],
            ]
        )

        x = get_squared_l2_norm_single_voxel(f_evals, corners, p)
        assert not np.isnan(x)
        assert not np.isinf(x)

    def test_1(self) -> None:
        """Constant function. f(x,y,z) = 3"""

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        p = 4
        f_evals = 3 * np.ones((p**3))
        x = get_squared_l2_norm_single_voxel(
            f_evals, node_corners_to_3d_corners(root), p
        )

        assert np.isclose(x, 9.0)

    def test_2(self) -> None:
        """Low-degree polynomial. f(x,y,z) = sqrt(x + y + z)
        Antiderivative of f^2 is 1/2 x^2 + 1/2 y^2 + 1/2 z^2
        Evaluating that from 0 to 1 gives 1.5
        """

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        p = 4
        pts = get_all_leaf_3d_cheby_points(p, root)
        f_evals = jnp.sqrt(pts[..., 0] + pts[..., 1] + pts[..., 2])
        print("test_2: f_evals shape: ", f_evals.shape)
        x = get_squared_l2_norm_single_voxel(
            f_evals, node_corners_to_3d_corners(root), p
        )
        assert np.isclose(x, 1.5)

    def test_3(self) -> None:
        """
        Low-degree polynomial. f(x,y,z) = x^2 + y

        Norm of f(x,y,z) = x^2 + y over [0,1]x[0,1]x[0,1] is sqrt(13/15)
        """

        root = Node(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        p = 6

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + x**2"""
            return x[..., 0] ** 2 + x[..., 1]

        pts = get_all_leaf_3d_cheby_points(p, root)
        f_evals = f(pts)
        x = get_squared_l2_norm_single_voxel(
            f_evals, node_corners_to_3d_corners(root), p
        )
        expected_x = 13 / 15
        print("test_3: x: ", x)
        print("test_3: expected_x: ", expected_x)
        assert np.isclose(x, expected_x)


if __name__ == "__main__":
    pytest.main()
