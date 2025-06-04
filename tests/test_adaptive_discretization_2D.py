import jax.numpy as jnp
import jax
from jaxhps._adaptive_discretization_2D import (
    node_to_bounds,
    generate_adaptive_mesh_level_restriction_2D,
    get_squared_l2_norm_single_panel,
)
from jaxhps._grid_creation_2D import (
    compute_interior_Chebyshev_points_adaptive_2D,
    get_all_leaves,
)
from jaxhps._discretization_tree import DiscretizationNode2D
from jaxhps._discretization_tree_operations_2D import add_four_children
import numpy as np
import logging


class Test_generate_adaptive_mesh_level_restriction_2D:
    def test_0(self, caplog) -> None:
        """TODO: This test is running slowly. I am not sure if that's because it's a problem that is too hard or beca"""
        caplog.set_level(logging.DEBUG)

        p = 4
        q = 2
        root = DiscretizationNode2D(
            xmin=0.0, xmax=2.0, ymin=0.0, ymax=2.0, depth=0
        )
        add_four_children(root, root=root, q=q)

        def f(x: jax.Array) -> jax.Array:
            """f(x,y) =sin(4y) + x**2"""
            return jnp.sin(3 * x[..., 1] ** 2) + x[..., 0] ** 2

        # Generate the adaptive mesh level restriction
        generate_adaptive_mesh_level_restriction_2D(
            root, f, tol=1e-2, p=p, q=q
        )
        all_leaves = get_all_leaves(root)
        logging.info("Number of leaves: %s", len(all_leaves))
        # Assert that the root node has been refined.
        assert root.children is not None
        assert len(root.children) == 4

    def test_1(self, caplog) -> None:
        caplog.set_level(logging.DEBUG)

        root_l2 = DiscretizationNode2D(
            xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0, depth=0
        )

        def f(x: jax.Array) -> jax.Array:
            """f(x,y) = y + x**2"""
            return jnp.sin(4 * x[..., 1]) + x[..., 0] ** 2

        generate_adaptive_mesh_level_restriction_2D(
            root_l2, f, tol=1e-3, p=4, q=2, l2_norm=True
        )


class Test_get_squared_l2_norm_single_panel:
    def test_0(self) -> None:
        """Make sure things run without error."""
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        p = 4

        f_evals = np.random.normal(size=(p**2))

        x = get_squared_l2_norm_single_panel(f_evals, node_to_bounds(root), p)

        print("test_0: x: ", x)
        assert not np.isnan(x)
        assert not np.isinf(x)

    def test_1(self) -> None:
        """Make sure things are correct for a constant function."""
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)
        p = 4

        f_evals = 3 * np.ones((p**2))
        x = get_squared_l2_norm_single_panel(f_evals, node_to_bounds(root), p)

        print("test_1: x: ", x)
        assert np.isclose(x, 9.0)

    def test_2(self) -> None:
        """Make sure things are correct for a low-degree polynomial."""
        root = DiscretizationNode2D(xmin=0.0, xmax=1.0, ymin=0.0, ymax=1.0)

        p = 16

        def f(x: jnp.array) -> jnp.array:
            """f(x,y) = y + x**2"""
            return x[..., 1] + x[..., 0] ** 2

        cheby_pts = compute_interior_Chebyshev_points_adaptive_2D(root, p)
        f_evals = f(cheby_pts).flatten()

        x = get_squared_l2_norm_single_panel(f_evals, node_to_bounds(root), p)

        print("test_2: x: ", x)

        # Norm of f(x,y) = y + x**2 over [0,1]x[0,1] is sqrt(13/15)
        expected_x = 13 / 15
        print("test_2: expected_x: ", expected_x)
        assert np.isclose(x, expected_x)

    def test_3(self) -> None:
        """Constant function with side length pi"""
        root = DiscretizationNode2D(
            xmin=-np.pi / 2, xmax=np.pi / 2, ymin=-np.pi / 2, ymax=np.pi / 2
        )

        p = 16
        f_evals = 3 * np.ones((p**2))
        expected_x = 9 * np.pi**2
        x = get_squared_l2_norm_single_panel(f_evals, node_to_bounds(root), p)
        print("test_3: x: ", x)
        print("test_3: expected_x: ", expected_x)
        assert np.isclose(x, expected_x)
