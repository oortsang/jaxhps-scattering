import jax.numpy as jnp

from jaxhps._adaptive_discretization_3D import (
    node_to_bounds,
    generate_adaptive_mesh_level_restriction,
    get_squared_l2_norm_single_voxel,
)
from jaxhps._grid_creation_3D import (
    compute_interior_Chebyshev_points_adaptive_3D,
)
from jaxhps._discretization_tree import DiscretizationNode3D, get_all_leaves
from jaxhps._discretization_tree_operations_3D import add_eight_children
import numpy as np


class Test_generate_adaptive_mesh_level_restriction:
    def test_0(self) -> None:
        """Make sure things run without error"""

        p = 4
        tol = 1e-03

        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        add_eight_children(root)

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + x**2"""
            return x[..., 1] + x[..., 0] ** 2

        generate_adaptive_mesh_level_restriction(root, f, tol, p, q=p - 2)
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

        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        add_eight_children(root)

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + x**2"""
            return x[..., 1] + x[..., 0] ** 2

        generate_adaptive_mesh_level_restriction(
            root, f, tol, p, q=p - 2, l2_norm=True
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
        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0
        )
        p = 10

        f_evals = np.random.normal(size=(p**3))

        bounds = node_to_bounds(root)

        x = get_squared_l2_norm_single_voxel(f_evals, bounds, p)
        assert not np.isnan(x)
        assert not np.isinf(x)

    def test_1(self) -> None:
        """Constant function. f(x,y,z) = 3"""

        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        p = 4
        f_evals = 3 * np.ones((p**3))
        x = get_squared_l2_norm_single_voxel(f_evals, node_to_bounds(root), p)

        assert np.isclose(x, 9.0)

    def test_2(self) -> None:
        """Low-degree polynomial. f(x,y,z) = sqrt(x + y + z)
        Antiderivative of f^2 is 1/2 x^2 + 1/2 y^2 + 1/2 z^2
        Evaluating that from 0 to 1 gives 1.5
        """

        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        p = 4
        pts = compute_interior_Chebyshev_points_adaptive_3D(root, p)
        f_evals = jnp.sqrt(pts[..., 0] + pts[..., 1] + pts[..., 2])
        print("test_2: f_evals shape: ", f_evals.shape)
        x = get_squared_l2_norm_single_voxel(f_evals, node_to_bounds(root), p)
        assert np.isclose(x, 1.5)

    def test_3(self) -> None:
        """
        Low-degree polynomial. f(x,y,z) = x^2 + y

        Norm of f(x,y,z) = x^2 + y over [0,1]x[0,1]x[0,1] is sqrt(13/15)
        """

        root = DiscretizationNode3D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, zmin=0.0, zmax=1.0, depth=0
        )
        p = 6

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + x**2"""
            return x[..., 0] ** 2 + x[..., 1]

        pts = compute_interior_Chebyshev_points_adaptive_3D(root, p)
        f_evals = f(pts)
        x = get_squared_l2_norm_single_voxel(f_evals, node_to_bounds(root), p)
        expected_x = 13 / 15
        print("test_3: x: ", x)
        print("test_3: expected_x: ", expected_x)
        assert np.isclose(x, expected_x)
